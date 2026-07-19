"""Inpaint an object-free background plate for the layered (ghost-free) renderer.

The farthest region is kept as the backdrop; every nearer region (a "mover") is
erased and filled from surrounding context. At render time the movers slide as
cutouts over this plate, so nothing of them is left behind to ghost.

    python build_background.py
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def _bleed_fill(img, mask, smooth=0):
    """Fill each masked pixel with the colour of the NEAREST surviving pixel.

    This bleeds the boundary colour inward (Voronoi of the edge), so a removed
    object's gap takes the colour of whatever it was touching — the faithful,
    artefact-free fill for the thin disocclusion slivers a parallax opens up.
    """
    hole = (mask > 0).astype(np.uint8) * 255          # 255 = to fill, 0 = known
    if not hole.any():
        return img.copy()
    _, labels = cv2.distanceTransformWithLabels(hole, cv2.DIST_L2, 3,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    known = hole == 0
    lut = np.zeros((int(labels.max()) + 1, 3), np.uint8)
    lut[labels[known]] = img[known]                    # label -> that boundary pixel's colour
    filled = lut[labels]                               # every pixel -> nearest known colour
    if smooth > 0:                                     # optional: soften Voronoi seams inside the hole
        blurred = cv2.GaussianBlur(filled, (0, 0), smooth)
        m = (mask > 0)[..., None]
        filled = np.where(m, blurred, filled)
    return filled


def bleed_from(img, source_mask):
    """Spread the colours of `source_mask` outward to every other pixel (nearest).

    Used to extend one layer's content into the holes left by the layers in front
    of it: each hole pixel takes the nearest source-layer colour.
    """
    src = np.where(source_mask, 0, 255).astype(np.uint8)
    if not source_mask.any():
        return img.copy()
    _, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros((int(labels.max()) + 1, 3), np.uint8)
    lut[labels[source_mask]] = img[source_mask]
    return lut[labels]


def bleed_from_mode(img, source_mask, window=41, levels=6):
    """Fill every non-source pixel with the MODE colour of nearby source pixels.

    Colours are quantised into `levels`^3 bins; for each pixel we take the most
    common bin among source pixels within a `window`-sized neighbourhood, then use
    that bin's average colour. Far more stable than nearest-pixel bleed — no single
    outlier can streak across the fill. Pixels with no source in reach fall back to
    the nearest-pixel bleed.
    """
    h, w = img.shape[:2]
    known = source_mask
    if not known.any():
        return img.copy()
    q = np.minimum((img.astype(np.int32) * levels) // 256, levels - 1)
    qidx = (q[..., 0] * levels + q[..., 1]) * levels + q[..., 2]          # 0 .. levels^3-1
    K = levels ** 3

    # representative colour per bin = mean of the source pixels in it
    palette = np.zeros((K, 3), np.float32)
    counts = np.zeros(K, np.float32)
    np.add.at(palette, qidx[known], img[known].astype(np.float32))
    np.add.at(counts, qidx[known], 1.0)
    nz = counts > 0
    palette[nz] /= counts[nz, None]

    # windowed mode over source pixels only
    best_cnt = np.zeros((h, w), np.float32)
    best_v = np.full((h, w), -1, np.int32)
    for v in np.where(nz)[0]:
        c = cv2.boxFilter(((qidx == v) & known).astype(np.float32), -1, (window, window),
                          normalize=False)
        m = c > best_cnt
        best_cnt[m] = c[m]
        best_v[m] = v

    out = img.copy()
    holes = ~known
    got = holes & (best_v >= 0)
    out[got] = palette[best_v[got]].astype(np.uint8)
    rem = holes & (best_v < 0)                       # nothing in reach -> nearest fallback
    if rem.any():
        nn = bleed_from(img, source_mask)
        out[rem] = nn[rem]
    return out


def bleed_from_pushpull(img, source_mask):
    """把 source_mask 的颜色以「近重远轻」的方式铺满其余像素(push-pull 金字塔)。

    Pull:逐级下采样,颜色按权重(是否已知)加权汇聚到粗层。
    Push:从粗到细双三次上采样,细层已知处保留原色,未知处用上一级(更远/更粗)
          的估计补——所以近处已知色主导,越远才由越粗的平均接管。平滑、无条纹无面片。
    """
    h, w = img.shape[:2]
    C = img.astype(np.float32) * source_mask[..., None]   # 预乘颜色
    W = source_mask.astype(np.float32)                    # 权重(1=已知)
    pyr_C, pyr_W = [C], [W]
    while min(pyr_C[-1].shape[:2]) > 1:                   # PULL:下采样
        nh, nw = (pyr_C[-1].shape[0] + 1) // 2, (pyr_C[-1].shape[1] + 1) // 2
        pyr_C.append(cv2.resize(pyr_C[-1], (nw, nh), interpolation=cv2.INTER_AREA))
        pyr_W.append(cv2.resize(pyr_W[-1], (nw, nh), interpolation=cv2.INTER_AREA))

    est = pyr_C[-1] / np.maximum(pyr_W[-1], 1e-5)[..., None]
    for L in range(len(pyr_C) - 2, -1, -1):              # PUSH:上采样融合
        Wl = np.clip(pyr_W[L], 0, 1)[..., None]
        col_l = pyr_C[L] / np.maximum(pyr_W[L], 1e-5)[..., None]
        up = cv2.resize(est, (pyr_W[L].shape[1], pyr_W[L].shape[0]), interpolation=cv2.INTER_CUBIC)
        est = Wl * col_l + (1 - Wl) * up                 # 已知处保留,未知处用更粗的估计
    return np.clip(est, 0, 255).astype(np.uint8)


def bleed_from_harmonic(img, source_mask, smooth_iters=40, locality_px=60.0):
    """Laplace/harmonic 填充:内部 = 局部边界色的平滑插值(像肥皂膜)。

    邻居白就更白、邻居黑就更黑,只有离所有边界都很远才慢慢过渡到平均——不会像
    push-pull 那样塌成一片全局平均。用多重网格(coarse 解当初值 + 每级 Jacobi 精修)
    加速求解 ∇²u=0。

    locality_px:让局部色保持更久。>0 时按"离边界的距离"把结果往最近边界色拉,
    距离 < 约 locality_px 的地方更贴局部色,越深越回到平滑 harmonic;越大越局部。
    """
    src_img = img
    img = img.astype(np.float32)

    def solve(im, kn):
        h, w = kn.shape
        unknown = ~kn
        if not unknown.any():
            return im
        if min(h, w) <= 4:                                   # 最粗:多迭代几次
            out = im.copy()
            for _ in range(400):
                out[unknown] = cv2.blur(out, (3, 3))[unknown]
            return out
        kf = kn.astype(np.float32)
        ch, cw = (h + 1) // 2, (w + 1) // 2
        # 粗层已知色 = 只对已知像素加权下采样(避免未知的 0 把颜色拉黑)
        c_im = cv2.resize(im * kf[..., None], (cw, ch), interpolation=cv2.INTER_AREA)
        c_den = cv2.resize(kf, (cw, ch), interpolation=cv2.INTER_AREA)
        c_im /= np.maximum(c_den, 1e-6)[..., None]
        c_solved = solve(c_im, c_den > 1e-3)                 # 递归解粗层
        up = cv2.resize(c_solved, (w, h), interpolation=cv2.INTER_LINEAR)
        out = im.copy()
        out[unknown] = up[unknown]                           # 粗解当初值
        for _ in range(smooth_iters):                        # 细层 Jacobi:传播局部边界
            out[unknown] = cv2.blur(out, (3, 3))[unknown]
        return out

    kn = source_mask.astype(bool)
    hm = solve(img, kn)                                   # 平滑 harmonic 解
    if locality_px > 0:
        nn = bleed_from(src_img, kn).astype(np.float32)   # 最近边界色(最局部)
        dist = cv2.distanceTransform((~kn).astype(np.uint8) * 255, cv2.DIST_L2, 3)
        wl = np.exp(-dist / float(locality_px))[..., None]  # 边界处~1,越深越小
        hm = wl * nn + (1 - wl) * hm
        hm[kn] = src_img[kn]                              # 已知处保持原色
    return np.clip(hm, 0, 255).astype(np.uint8)


def _inpaint(img, mask, method, radius):
    """Fill the masked (white) region.

    'pushpull' = 近重远轻的平滑金字塔填充(默认,最顺),
    'bleed'    = nearest boundary colour(会条纹),
    'mode'     = 邻域众数(会面片),
    'lama'     = smart AI completion(会幻想结构),
    'telea'    = classical fast blur.
    """
    if method == "pushpull":
        return bleed_from_pushpull(img, mask == 0)   # 已知=非填充区
    if method == "mode":
        return bleed_from_mode(img, mask)
    if method == "bleed":
        return _bleed_fill(img, mask)
    if method == "lama":
        try:
            from simple_lama_inpainting import SimpleLama
            lama = SimpleLama()
            out = lama(Image.fromarray(img), Image.fromarray(mask).convert("L"))
            return np.asarray(out.convert("RGB"))[: img.shape[0], : img.shape[1]]
        except Exception as exc:
            print(f"LaMa unavailable ({exc}); falling back to bleed")
            return _bleed_fill(img, mask)
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)


def build_background(image_path, scene_path, labels_path, out_path,
                     fg_thresh=None, method="pushpull", dilate=9, radius=4):
    """Inpaint the moving layers out of the image and tag movers in scene.json.

    Default: everything moves except the single farthest region, which stays in the
    plate as the static backdrop. Pass fg_thresh to instead keep every far region
    below that depth static. Marking movers in the scene keeps the editor's sprite
    set identical to what was inpainted here, so the plate always matches.
    """
    img = np.asarray(Image.open(image_path).convert("RGB"))
    scene = json.load(open(scene_path))
    lab = np.asarray(Image.open(labels_path).convert("L"))

    regions = scene["regions"]
    if not regions:
        Image.fromarray(img).save(out_path)
        return 0
    if fg_thresh is None:                                     # everything but the backdrop
        far = min(regions, key=lambda r: r["depth_mean"])["id"]
        movers = [r for r in regions if r["id"] != far]
    else:
        movers = [r for r in regions if r["depth_mean"] >= fg_thresh]
        if not movers:
            far = min(regions, key=lambda r: r["depth_mean"])["id"]
            movers = [r for r in regions if r["id"] != far]

    mover_ids = {r["id"] for r in movers}
    for r in regions:
        r["mover"] = r["id"] in mover_ids
    json.dump(scene, open(scene_path, "w"), indent=2)

    # Fill each mover layer separately, nearest first, so every layer's vacated
    # area bleeds from its OWN immediate surroundings (the layer behind it) rather
    # than from one far backdrop across the whole frame. "每一层都渗透".
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    plate = img.copy()
    for r in sorted(movers, key=lambda r: -r["depth_mean"]):
        m = np.zeros(lab.shape, np.uint8)
        m[lab == r["id"]] = 255
        m = cv2.dilate(m, kernel)
        plate = _inpaint(plate, m, method, radius)
    Image.fromarray(plate).save(out_path)
    return len(movers)


def main():
    p = argparse.ArgumentParser(description="Inpaint the background plate for layered rendering")
    p.add_argument("--image", default=str(OUTPUT_DIR / "cropped_input.png"))
    p.add_argument("--scene", default=str(OUTPUT_DIR / "scene.json"))
    p.add_argument("--labels", default=str(OUTPUT_DIR / "region_labels.png"))
    p.add_argument("--out", default=str(OUTPUT_DIR / "background.png"))
    p.add_argument("--fg-thresh", type=float, default=None,
                   help="Keep regions below this depth static (default: only the farthest is static)")
    p.add_argument("--method", choices=["pushpull", "bleed", "mode", "lama", "telea"], default="pushpull",
                   help="pushpull = 平滑金字塔(默认), bleed/mode = 最近邻/众数, lama = AI, telea = 经典")
    args = p.parse_args()
    n = build_background(args.image, args.scene, args.labels, args.out,
                         fg_thresh=args.fg_thresh, method=args.method)
    print(f"Inpainted {n} mover regions -> {args.out}")


if __name__ == "__main__":
    main()
