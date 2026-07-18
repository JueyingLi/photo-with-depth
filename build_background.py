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


def _inpaint(img, mask, method, radius):
    """Fill the masked (white) region.

    'bleed' = nearest boundary colour propagated inward (default, faithful),
    'lama'  = smart AI completion (reconstructs structure, may hallucinate),
    'telea' = classical fast blur.
    """
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
                     fg_thresh=None, method="lama", dilate=9, radius=4):
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
    p.add_argument("--method", choices=["bleed", "lama", "telea"], default="bleed",
                   help="bleed = nearest boundary colour inward (default), lama = AI, telea = classical")
    args = p.parse_args()
    n = build_background(args.image, args.scene, args.labels, args.out,
                         fg_thresh=args.fg_thresh, method=args.method)
    print(f"Inpainted {n} mover regions -> {args.out}")


if __name__ == "__main__":
    main()
