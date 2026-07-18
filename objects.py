"""Object-aware segmentation: SAM 2 "everything" masks fused with the depth map.

SAM 2 automatic mask generation segments every object (and often its parts). We
merge parts that overlap and sit at a similar depth back into whole objects, read
each object's depth from the heatmap so it keeps its place in the scene, and fill
whatever SAM missed with the depth-cluster regions from regions.py. The result is
one label map + scene.json in the same contract the editor already reads.
"""
from pathlib import Path

import cv2
import numpy as np

from regions import _auto_depth_levels, _kmeans_1d, _label_name, _normalize, build_regions, save_scene

DEFAULT_MODEL = "facebook/sam2.1-hiera-tiny"


def run_sam2_masks(image, model_id: str = DEFAULT_MODEL, points_per_batch: int = 64,
                   points_per_side: int = 32, pred_iou_thresh: float = 0.7,
                   stability_score_thresh: float = 0.85):
    """Run SAM 2 automatic mask generation. Returns a list of bool HxW masks.

    Denser `points_per_side` finds more (smaller) objects at higher CPU cost.
    """
    from transformers import pipeline

    generator = pipeline("mask-generation", model=model_id, device=-1)
    out = generator(
        image,
        points_per_batch=points_per_batch,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )
    masks = [np.asarray(m, dtype=bool) for m in out["masks"]]
    scores = [float(s) for s in out.get("scores", [1.0] * len(masks))]
    # order by SAM's own confidence; area/depth ordering happens later
    order = np.argsort(scores)[::-1]
    return [masks[i] for i in order], [scores[i] for i in order]


def _median_depth(depth: np.ndarray, mask: np.ndarray) -> float:
    vals = depth[mask]
    return float(np.median(vals)) if vals.size else 0.0


def build_layer_groups(
    depth01: np.ndarray,
    masks,
    n_groups=None,
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.85,
    smooth_sigma: float = 2.0,
):
    """Collapse the scene into a few depth-layer groups, snapping objects whole.

    The depth map is clustered into a small number of natural layers. Each SAM
    object is assigned entirely to the layer holding the majority of its pixels
    (so a person straddling two layers lands on one). Every pixel then belongs to
    exactly one layer, and each layer becomes a single selectable region.

    Returns (label_map uint8, regions list, centers).
    """
    h, w = depth01.shape
    depth = _normalize(depth01)
    smooth = cv2.GaussianBlur(depth, (0, 0), smooth_sigma)
    area = h * w

    if n_groups is not None:
        hist, _ = np.histogram(depth, bins=256, range=(0.0, 1.0))
        bin_centers = (np.arange(256) + 0.5) / 256.0
        centers, _ = _kmeans_1d(bin_centers, hist.astype(float), int(n_groups))
    else:
        centers, _ = _auto_depth_levels(depth)
    K = len(centers)

    # Per-pixel nearest layer, then snap each object onto its dominant layer.
    layer = np.argmin(np.abs(smooth[..., None] - centers[None, None, :]), axis=2).astype(np.int32)
    objs = [m for m in masks if min_area_frac * area <= int(m.sum()) <= max_area_frac * area]
    objs.sort(key=lambda m: int(m.sum()))  # small first, so big objects win overlaps
    for m in objs:
        vals = layer[m]
        if vals.size:
            layer[m] = int(np.bincount(vals, minlength=K).argmax())

    # One region per (non-empty) layer.
    label_map = np.zeros((h, w), dtype=np.int32)
    regions = []
    new_id = 0
    for lyr in range(K):
        mask = layer == lyr
        if not mask.any():
            continue
        new_id += 1
        label_map[mask] = new_id
        ys, xs = np.where(mask)
        d = depth[mask]
        cy, cx = float(ys.mean()), float(xs.mean())
        regions.append({
            "id": new_id,
            "label": f"layer {new_id} · " + _label_name(float(centers[lyr]), cy, cx, h, w),
            "source": "layer",
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "centroid": [round(cx, 1), round(cy, 1)],
            "area_frac": round(int(mask.sum()) / area, 4),
            "depth_min": round(float(d.min()), 3),
            "depth_max": round(float(d.max()), 3),
            "depth_mean": round(float(centers[lyr]), 3),
            "depth_mode": "layer",
        })
    return label_map.astype(np.uint8), regions, centers


def build_hybrid_regions(
    depth01: np.ndarray,
    masks,
    min_area_frac: float = 0.0015,
    max_area_frac: float = 0.85,
    contain_thresh: float = 0.75,
    depth_merge: float = 0.12,
    bg_min_area_frac: float = 0.012,
):
    """Fuse SAM object masks with depth into a single labelled region map.

    Returns (label_map uint8, regions list). Each region records whether it came
    from an object mask or the depth-cluster background, plus its depth stats.
    """
    h, w = depth01.shape
    depth = _normalize(depth01)
    area = h * w
    min_area, max_area = min_area_frac * area, max_area_frac * area

    # Keep object-sized masks; drop specks and full-frame background blobs.
    objs = [m for m in masks if min_area <= int(m.sum()) <= max_area]
    objs.sort(key=lambda m: int(m.sum()), reverse=True)  # whole objects before parts

    # Greedy merge: a smaller mask mostly inside a kept object at similar depth is
    # a part of that object -> union it in. Otherwise it's a new object.
    kept = []
    for m in objs:
        md = _median_depth(depth, m)
        m_area = int(m.sum())
        for k in kept:
            inter = int(np.logical_and(m, k["mask"]).sum())
            if inter / m_area > contain_thresh and abs(md - k["depth"]) < depth_merge:
                k["mask"] |= m
                k["depth"] = _median_depth(depth, k["mask"])
                break
        else:
            kept.append({"mask": m.copy(), "depth": md})

    # Paint far objects first so nearer ones occlude them (correct overlap).
    kept.sort(key=lambda k: k["depth"])
    label_map = np.zeros((h, w), dtype=np.int32)
    sources = {}
    next_id = 1
    for k in kept:
        label_map[k["mask"]] = next_id
        sources[next_id] = "object"
        next_id += 1

    # Fill whatever SAM didn't cover with depth-cluster regions.
    background = label_map == 0
    if background.any():
        bg_labels, _ = build_regions(depth01, min_area_frac=bg_min_area_frac)
        for bid in range(1, int(bg_labels.max()) + 1):
            piece = (bg_labels == bid) & background
            if int(piece.sum()) >= min_area:
                label_map[piece] = next_id
                sources[next_id] = "depth"
                next_id += 1

    # Absorb tiny leftovers into nearest region so there are no holes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for _ in range(8):
        holes = label_map == 0
        if not holes.any():
            break
        grown = cv2.dilate(label_map.astype(np.float32), kernel).astype(np.int32)
        label_map[holes] = grown[holes]

    # Relabel contiguously and build region metadata.
    regions = []
    old_ids = [i for i in np.unique(label_map) if i != 0]
    remap = {old: new for new, old in enumerate(old_ids, start=1)}
    relabeled = np.zeros_like(label_map)
    for old, new in remap.items():
        relabeled[label_map == old] = new
    label_map = relabeled

    for old, new in remap.items():
        mask = label_map == new
        ys, xs = np.where(mask)
        d = depth[mask]
        dmin, dmax, dmean = float(d.min()), float(d.max()), float(d.mean())
        cy, cx = float(ys.mean()), float(xs.mean())
        src = sources[old]
        regions.append({
            "id": new,
            "label": ("object " if src == "object" else "") + _label_name(dmean, cy, cx, h, w),
            "source": src,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "centroid": [round(cx, 1), round(cy, 1)],
            "area_frac": round(int(mask.sum()) / area, 4),
            "depth_min": round(dmin, 3),
            "depth_max": round(dmax, 3),
            "depth_mean": round(dmean, 3),
            "depth_mode": "object" if src == "object" else ("continuous" if dmax - dmin > 0.18 else "object"),
        })

    return label_map.astype(np.uint8), regions
