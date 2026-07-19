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

from regions import _auto_depth_levels, _kmeans_1d, _label_name, _normalize, save_scene
from step_3_build_regions import build_regions

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


def save_segments(masks, path):
    """把 SAM masks 拍平成一张分割图存盘,供网页「SAM 选物体」用。

    小物体画在上层(大 mask 先画、小 mask 后画覆盖),这样点小物体能选到它。
    用 RGB 编码 segment id:id = R + G*256(id 0 = 没有物体),支持上千个 mask。
    """
    from PIL import Image
    if not masks:
        return 0
    H, W = masks[0].shape
    seg = np.zeros((H, W), dtype=np.int32)
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)  # 大→小
    for rank, i in enumerate(order, start=1):        # 后画的小 mask 覆盖 → 小物体在上
        seg[masks[i]] = rank
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = (seg & 0xFF).astype(np.uint8)
    rgb[..., 1] = ((seg >> 8) & 0xFF).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(path)
    return len(masks)


def _median_depth(depth: np.ndarray, mask: np.ndarray) -> float:
    vals = depth[mask]
    return float(np.median(vals)) if vals.size else 0.0


def snap_objects_to_layers(level_map, masks, dominant_frac_thresh=0.9,
                           min_area_frac=0.002, max_area_frac=0.85):
    """用 SAM 边界清理层图:某物体 ≥thresh 的像素在同一层,就把整块归到那层。

    输入 level_map(每像素的层号,来自 step_3 的分层)和 SAM masks;对每个物体,
    统计它在各层的像素占比,若最高占比 ≥ dominant_frac_thresh,就把整块(含少数派)
    都改成那层号;不到阈值就不动(物体确实跨层,保留逐像素)。

    返回 (新 level_map, 被归整的物体数)。
    """
    level_map = np.asarray(level_map)
    out = level_map.astype(np.int32).copy()
    h, w = out.shape
    area = h * w
    K = int(out.max()) + 1

    objs = [m for m in masks if min_area_frac * area <= int(m.sum()) <= max_area_frac * area]
    objs.sort(key=lambda m: int(m.sum()))  # 小的先处理,大的后盖(重叠时大物体优先)
    snapped = 0
    for m in objs:
        vals = out[m]
        if not vals.size:
            continue
        counts = np.bincount(vals, minlength=K)
        dom = int(counts.argmax())
        if counts[dom] / vals.size >= dominant_frac_thresh:   # ≥90% → 整块归那层
            out[m] = dom
            snapped += 1
    return out, snapped


def extract_flat_objects(label_map, depth01, masks, min_area_frac=0.03,
                         max_depth_range=0.10, contained_ratio=1.5):
    """大而深度均匀的 SAM 物体,若被并在更大区域里,独立成一块(自成一层)。

    判定"平整整体":
      - 面积 ≥ min_area_frac(默认 3%)
      - 深度极差 p95-p5 ≤ max_depth_range(深度变化很少)
    判定"被包含在其他整体里":
      - 当前它主要落在某个区域内,而那区域面积 ≥ contained_ratio × 物体面积
    满足两者 → 把整块赋一个新区域 id 独立出来。

    返回 (新 label_map, 独立出来的物体数)。
    """
    depth = _normalize(np.asarray(depth01, np.float32))
    out = np.asarray(label_map).astype(np.int32).copy()
    h, w = out.shape
    area = h * w
    next_id = int(out.max()) + 1
    extracted = 0

    for m in sorted(masks, key=lambda m: -int(m.sum())):   # 大的先处理
        a = int(m.sum())
        if a < min_area_frac * area:                       # 不够大
            continue
        d = depth[m]
        if d.size == 0:
            continue
        if float(np.percentile(d, 95) - np.percentile(d, 5)) > max_depth_range:
            continue                                       # 深度变化大,不是平整整体
        vals = out[m]
        vals = vals[vals > 0]
        if vals.size == 0:
            continue
        dom = int(np.bincount(vals).argmax())              # 它当前主要属于哪个区域
        dom_area = int((out == dom).sum())
        if dom_area >= contained_ratio * a:                # 被包在更大的区域里 → 抠出来
            out[m] = next_id
            next_id += 1
            extracted += 1
    return out, extracted


def build_sam_valley_regions(depth01, masks, min_area_frac=0.002, smooth_sigma=2.0,
                             valley_min_gap=0.05, valley_min_prom=0.06,
                             dominant_frac_thresh=0.9, extract_area_frac=0.03,
                             extract_depth_range=0.10, morph=7, merge_same_layer=True):
    """notebook 的 §3→§4c 流水线(搬进生产):

      valley 分层 → SAM 物体 ≥thresh 归整层 →(同层合并/连通域拆)→ 抠出平整大整体 → 元数据。

    merge_same_layer=True(默认):**同一深度层合成一块**(= 一张 sprite),不按空间连通拆开
      → 每个深度层一张贴图,最干净。False 则按连通块拆(左右楼各自独立视差)。
    """
    from step_3_build_regions import (prepare_depth, valley_levels, assign_by_boundaries,
                                       split_regions, describe_regions)
    depth, smooth = prepare_depth(depth01, smooth_sigma)
    _, boundaries = valley_levels(depth, min_gap=valley_min_gap, min_prom=valley_min_prom)  # §3
    level_map = assign_by_boundaries(smooth, boundaries)
    level_map, _ = snap_objects_to_layers(level_map, masks, dominant_frac_thresh, min_area_frac)  # §4b

    if merge_same_layer:                                   # 同层并成一块(不拆连通域)
        label_map = np.zeros(level_map.shape, np.int32)
        for new_id, lvl in enumerate(np.unique(level_map), start=1):
            label_map[level_map == lvl] = new_id
    else:
        label_map = split_regions(level_map, min_area_frac=min_area_frac, morph=morph)

    label_map, _ = extract_flat_objects(label_map, depth01, masks,                        # §4c
                                        min_area_frac=extract_area_frac,
                                        max_depth_range=extract_depth_range)
    regions = describe_regions(label_map, depth)
    return label_map.astype(np.uint8), regions


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
