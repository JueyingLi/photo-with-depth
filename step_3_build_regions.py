"""第 3 步:把深度图切成可编辑的区域(region)。

把「分区」拆成几个清晰、可单独调用的小步骤,方便在 notebook 里分别调参。
重点是控制"边界怎么生成"——每一步都能单独换掉、单独看效果:

    1. prepare_depth    : 归一化 + 平滑(平滑越大,边界越圆滑)
    2. choose_levels    : 决定分几层、每层的深度中心(← 边界"数量"主要由这步定)
    3. assign_levels    : 每个像素归到最近的层 → 层图 level_map
    4. split_regions    : 每层形态学清理 + 连通域 → 区域标签图 label_map(← 边界"形状")
    5. describe_regions : 算每个区域的元数据(深度/bbox/标签…)

`build_regions` 只是把这 5 步串起来,签名与原来一致,可直接替换。

底层小工具(1D k-means、肘部选层、命名等)仍复用 regions.py,避免重复。
"""
import cv2
import numpy as np

from regions import _auto_depth_levels, _kmeans_1d, _label_name, _normalize


def prepare_depth(depth01: np.ndarray, smooth_sigma: float = 2.0):
    """① 归一化到 0~1,再高斯平滑。返回 (norm, smooth)。

    smooth 用于分层(减少噪点导致的碎边界);norm 用于最后算区域深度统计。
    smooth_sigma 越大,边界越圆滑、越不容易被细节打碎。
    """
    depth = _normalize(depth01)
    smooth = cv2.GaussianBlur(depth, (0, 0), smooth_sigma) if smooth_sigma > 0 else depth
    return depth, smooth


def choose_levels(depth: np.ndarray, n_clusters=None, n_bands=None) -> np.ndarray:
    """② 决定"分几层、界在哪"——控制边界数量的关键一步,返回每层的深度中心。

      - n_bands   给定 → 均匀切带(等距,最可控)
      - n_clusters给定 → 强制 k 个深度聚类(k-means)
      - 两者都不给 → 自动用肘部法(kneedle)选 k
    """
    if n_bands is not None:
        edges = np.linspace(0.0, 1.0, n_bands + 1)
        return (edges[:-1] + edges[1:]) / 2                      # 每个带的中点
    if n_clusters is not None:
        hist, _ = np.histogram(depth, bins=256, range=(0.0, 1.0))
        bin_centers = (np.arange(256) + 0.5) / 256.0
        centers, _ = _kmeans_1d(bin_centers, hist.astype(float), int(n_clusters))
        return centers
    centers, _ = _auto_depth_levels(depth)                       # 自动选 k
    return centers


def valley_levels(depth: np.ndarray, bins: int = 256, smooth_sigma: float = 2.5,
                  min_gap: float = 0.05, min_prom: float = 0.06):
    """②(替代法)按深度直方图的形状分层:峰=中心,谷=分界。

    比 k-means 更贴合多峰分布——分界落在真正的"谷底"而非中心中点。
    - min_gap  : 两峰离得比这近就合并(留高的)→ 治"太近的中心并到一起"
    - min_prom : 两峰之间的谷不够深(没掉下去这么多)就合并 → 治"浅谷不算分界"

    返回 (centers 峰位置, boundaries 谷位置)。boundaries 交给 assign_by_boundaries 分带。
    """
    depth = _normalize(depth)
    hist, _ = np.histogram(depth.ravel(), bins=bins, range=(0.0, 1.0))
    x = (np.arange(bins) + 0.5) / bins
    h = cv2.GaussianBlur(hist.astype(np.float32).reshape(1, -1), (0, 0), smooth_sigma).ravel()
    h = h / max(float(h.max()), 1e-9)

    peaks = [i for i in range(1, bins - 1)
             if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 0.03]     # 局部极大
    merged = []                                                            # 合并挨太近的峰
    for i in peaks:
        if merged and x[i] - x[merged[-1]] < min_gap:
            if h[i] > h[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    peaks = merged
    changed = True                                                         # 合并浅谷
    while changed and len(peaks) > 1:
        changed = False
        for k in range(len(peaks) - 1):
            a, b = peaks[k], peaks[k + 1]
            if min(h[a], h[b]) - h[a:b + 1].min() < min_prom:
                peaks = peaks[:k] + [a if h[a] >= h[b] else b] + peaks[k + 2:]
                changed = True
                break

    centers = x[np.array(peaks)] if peaks else np.array([0.5])
    boundaries = np.array([x[a + int(np.argmin(h[a:b + 1]))]              # 谷底 = 相邻峰间最低点
                           for a, b in zip(peaks[:-1], peaks[1:])])
    return centers, boundaries


def assign_by_boundaries(smooth: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """③(配合 valley_levels)按谷底分界把像素切成带 → 层图(0..K)。"""
    return np.digitize(smooth, np.sort(np.asarray(boundaries, dtype=np.float32)))


def assign_levels(smooth: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """③ 每个像素归到最近的层中心 → 层图(每像素一个层号 0..K-1)。"""
    centers = np.asarray(centers, dtype=np.float32)
    return np.argmin(np.abs(smooth[..., None] - centers[None, None, :]), axis=2)


def split_regions(level_map: np.ndarray, min_area_frac: float = 0.002,
                  morph: int = 7, fill_iter: int = 6) -> np.ndarray:
    """④ 把每一层按空间连通拆成区域,返回 label_map(0=未分配, 1..N=区域 id)。

    对每层:形态学开(去毛刺)→ 闭(补小洞)→ 连通域;只保留够大的块。
    最后把零散未分配的小孤岛并进最近的区域。

    调参对边界的影响:
      - morph         越大 → 边界越平滑、小突起被抹掉
      - min_area_frac 越大 → 小块越容易被丢弃(区域更少更大)
    """
    h, w = level_map.shape
    label_map = np.zeros((h, w), dtype=np.int32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
    min_area = min_area_frac * h * w
    next_id = 1

    for lvl in range(int(level_map.max()) + 1):
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

    for _ in range(fill_iter):                                   # 未分配孤岛并进最近区域
        holes = label_map == 0
        if not holes.any():
            break
        grown = cv2.dilate(label_map.astype(np.float32), kernel).astype(np.int32)
        label_map[holes] = grown[holes]
    return label_map


def describe_regions(label_map: np.ndarray, depth: np.ndarray) -> list:
    """⑤ 为每个区域算元数据 → regions 列表(id/label/bbox/深度统计等)。"""
    h, w = label_map.shape
    regions = []
    for rid in range(1, int(label_map.max()) + 1):
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
    return regions


def build_regions(depth01: np.ndarray, n_clusters=None, n_bands=None,
                  min_area_frac: float = 0.002, smooth_sigma: float = 2.0,
                  method: str = "valley", valley_min_gap: float = 0.05,
                  valley_min_prom: float = 0.06):
    """把 5 步串起来。默认用 valley(按峰谷分层),更贴深度分布。

    分层方式(优先级):n_bands > n_clusters > method。
      - n_bands   给定 → 均匀切带
      - n_clusters给定 → 强制 k 个 k-means 聚类
      - method="valley"(默认)→ 峰做中心、谷做分界(valley_levels)
      - method="kmeans"        → 自动肘部 k-means

    Returns (label_map uint8, regions list)。
    """
    depth, smooth = prepare_depth(depth01, smooth_sigma)
    if n_bands is not None:
        centers = choose_levels(depth, n_bands=n_bands)
        level_map = assign_levels(smooth, centers)
    elif n_clusters is not None:
        centers = choose_levels(depth, n_clusters=n_clusters)
        level_map = assign_levels(smooth, centers)
    elif method == "valley":
        centers, boundaries = valley_levels(depth, min_gap=valley_min_gap, min_prom=valley_min_prom)
        level_map = assign_by_boundaries(smooth, boundaries)
    else:  # "kmeans" 自动肘部
        centers = choose_levels(depth)
        level_map = assign_levels(smooth, centers)
    label_map = split_regions(level_map, min_area_frac=min_area_frac)
    regions = describe_regions(label_map, depth)
    return label_map.astype(np.uint8), regions
