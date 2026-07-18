"""Depth-band region segmentation for the interactive depth editor.

Splits the painting interior into editable regions derived from the depth map,
using only OpenCV + NumPy (no skimage/scipy). Each region gets a stable integer
id encoded into a grayscale label map PNG, plus metadata in scene.json so the
browser editor can select, re-rank, ignore, or brush-edit it.
"""
import json
from pathlib import Path

import cv2
import numpy as np


def _normalize(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    lo, hi = float(depth.min()), float(depth.max())
    if hi - lo < 1e-6:
        return np.zeros_like(depth)
    return (depth - lo) / (hi - lo)


def _kmeans_1d(vals: np.ndarray, weights: np.ndarray, k: int, iters: int = 100):
    """Weighted 1D k-means over histogram bins. Returns sorted cluster centers."""
    active = vals[weights > 0]
    c = np.linspace(active.min(), active.max(), k)
    for _ in range(iters):
        assign = np.argmin(np.abs(vals[:, None] - c[None, :]), axis=1)
        new_c = np.array([
            (vals[assign == j] * weights[assign == j]).sum() / max(weights[assign == j].sum(), 1e-9)
            if weights[assign == j].sum() > 0 else c[j]
            for j in range(k)
        ])
        if np.allclose(new_c, c):
            break
        c = new_c
    assign = np.argmin(np.abs(vals[:, None] - c[None, :]), axis=1)
    sse = float((weights * (vals - c[assign]) ** 2).sum())
    return np.sort(c), sse


def _auto_depth_levels(depth: np.ndarray, k_min: int = 5, k_max: int = 8):
    """Pick the number of depth clusters from the data via the kneedle elbow.

    A framed painting's depth is often a smooth gradient with no crisp K, so we
    detect the elbow of the k-means distortion curve (on log-SSE, which tames the
    steep initial drop) and clamp it to a sane range. Returns cluster centers.
    """
    hist, _ = np.histogram(depth, bins=256, range=(0.0, 1.0))
    centers = (np.arange(256) + 0.5) / 256.0
    hist = hist.astype(float)

    ks = list(range(1, k_max + 2))
    sse = np.array([_kmeans_1d(centers, hist, k)[1] for k in ks])
    y = np.log(np.clip(sse, 1e-9, None))
    x = np.array(ks, dtype=float)
    xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-9)
    # distance of each point from the chord joining first and last
    dist = np.abs((yn[-1] - yn[0]) * xn - (xn[-1] - xn[0]) * yn + xn[-1] * yn[0] - yn[-1] * xn[0])
    k = int(ks[int(np.argmax(dist))])
    k = max(k_min, min(k_max, k))
    return _kmeans_1d(centers, hist, k)[0], k


def _label_name(mean_depth: float, cy: float, cx: float, h: int, w: int) -> str:
    tier = "near" if mean_depth > 0.66 else "far" if mean_depth < 0.33 else "mid"
    vert = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "middle"
    horiz = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
    return f"{tier} {vert}-{horiz}"


def build_regions(
    depth01: np.ndarray,
    n_clusters=None,
    n_bands=None,
    min_area_frac: float = 0.002,
    smooth_sigma: float = 2.0,
):
    """Segment a normalized (0..1) depth map into editable regions.

    Depth is grouped into a small number of levels, then each level is split into
    large spatial blobs. The number of levels is decided from the data:
      - n_bands given    -> legacy uniform bands (manual, fine-grained)
      - n_clusters given -> force that many depth clusters (k-means)
      - both None        -> auto: kneedle elbow picks K (default, "reasonable")

    Returns (label_map, regions):
      label_map : uint8 HxW, 0 = unassigned, 1..N = region id
      regions   : list of dicts with id/label/bbox/depth stats/mode
    """
    h, w = depth01.shape
    depth = _normalize(depth01)
    smooth = cv2.GaussianBlur(depth, (0, 0), smooth_sigma)

    label_map = np.zeros((h, w), dtype=np.int32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    min_area = min_area_frac * h * w
    next_id = 1

    # Build a per-pixel level map, then split each level into connected blobs.
    if n_bands is not None:
        edges = np.linspace(0.0, 1.0, n_bands + 1)
        level_map = np.clip(np.digitize(smooth, edges[1:-1]), 0, n_bands - 1)
        n_levels = n_bands
        centers = (edges[:-1] + edges[1:]) / 2
    else:
        if n_clusters is not None:
            hist, _ = np.histogram(depth, bins=256, range=(0.0, 1.0))
            bin_centers = (np.arange(256) + 0.5) / 256.0
            centers, _ = _kmeans_1d(bin_centers, hist.astype(float), int(n_clusters))
        else:
            centers, n_clusters = _auto_depth_levels(depth)
        n_levels = len(centers)
        level_map = np.argmin(np.abs(smooth[..., None] - centers[None, None, :]), axis=2)

    for lvl in range(n_levels):
        band = (level_map == lvl).astype(np.uint8)
        band = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)
        band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel)
        num, comp = cv2.connectedComponents(band, connectivity=8)
        for c in range(1, num):
            pixels = comp == c
            if int(pixels.sum()) < min_area:
                continue
            label_map[pixels] = next_id
            next_id += 1

    # Absorb small unassigned islands into the nearest region by dilating a few
    # steps; anything still unassigned stays id 0 (base depth, freely brushable).
    for _ in range(6):
        holes = label_map == 0
        if not holes.any():
            break
        grown = cv2.dilate(label_map.astype(np.float32), kernel).astype(np.int32)
        label_map[holes] = grown[holes]

    regions = []
    for rid in range(1, next_id):
        mask = label_map == rid
        area = int(mask.sum())
        if area == 0:
            continue
        ys, xs = np.where(mask)
        d = depth[mask]
        dmin, dmax, dmean = float(d.min()), float(d.max()), float(d.mean())
        cy, cx = float(ys.mean()), float(xs.mean())
        regions.append({
            "id": rid,
            "label": _label_name(dmean, cy, cx, h, w),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "centroid": [round(cx, 1), round(cy, 1)],
            "area_frac": round(area / (h * w), 4),
            "depth_min": round(dmin, 3),
            "depth_max": round(dmax, 3),
            "depth_mean": round(dmean, 3),
            "depth_mode": "continuous" if (dmax - dmin) > 0.18 else "object",
        })

    return label_map.astype(np.uint8), regions


def save_scene(
    label_map: np.ndarray,
    regions: list,
    output_dir: Path,
    image_name: str = "cropped_input.png",
    depth_name: str = "depth_map.png",
    labelmap_name: str = "region_labels.png",
    scene_name: str = "scene.json",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w = label_map.shape

    from PIL import Image
    Image.fromarray(label_map, mode="L").save(output_dir / labelmap_name)

    scene = {
        "image": image_name,
        "depth": depth_name,
        "labelmap": labelmap_name,
        "width": int(w),
        "height": int(h),
        "region_count": len(regions),
        "regions": regions,
    }
    with open(output_dir / scene_name, "w") as f:
        json.dump(scene, f, indent=2)
    return output_dir / scene_name
