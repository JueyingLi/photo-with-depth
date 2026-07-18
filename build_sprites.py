"""Build layered-depth-image (LDI) sprites: one extended RGBA layer per region.

Each layer is the background for the layers in front of it. So every layer's
sprite keeps its own pixels AND extends (bleeds) its colour into the holes left
by all nearer layers. When a near layer moves, the layer behind — already filled
in and moving at its own parallax — shows through, instead of an empty gap.

    python build_sprites.py

Writes outputs/sprites/sprite_00.png (farthest) .. sprite_NN.png (nearest) and
tags each region with layer_index / sprite_count in scene.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from build_background import bleed_from_mode

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def build_sprites(image_path, scene_path, labels_path, out_dir, dilate=7):
    img = np.asarray(Image.open(image_path).convert("RGB"))
    scene = json.load(open(scene_path))
    lab = np.asarray(Image.open(labels_path).convert("L"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regions = sorted(scene["regions"], key=lambda r: r["depth_mean"])   # far -> near
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))

    for j, r in enumerate(regions):
        region_mask = lab == r["id"]
        nearer = np.zeros(lab.shape, bool)
        for rr in regions[j + 1:]:
            nearer |= lab == rr["id"]

        rgb = img.copy()
        if nearer.any():
            filled = bleed_from_mode(img, region_mask)  # nearer holes <- mode of this layer's nearby colour
            rgb[nearer] = filled[nearer]

        cover = region_mask | nearer
        alpha = cv2.GaussianBlur((cover.astype(np.uint8) * 255), (0, 0), 1.0)
        # keep the farthest layer fully opaque so the composite has no gaps
        if j == 0:
            alpha = np.full(lab.shape, 255, np.uint8)
        rgba = np.dstack([rgb, alpha])
        Image.fromarray(rgba, "RGBA").save(out_dir / f"sprite_{j:02d}.png")
        r["layer_index"] = j

    # write layer_index back onto the original (unsorted) region dicts + count
    order = {r["id"]: r["layer_index"] for r in regions}
    for r in scene["regions"]:
        r["layer_index"] = order[r["id"]]
    scene["sprite_count"] = len(regions)
    json.dump(scene, open(scene_path, "w"), indent=2)
    return len(regions)


def main():
    p = argparse.ArgumentParser(description="Build LDI sprites for the layered editor")
    p.add_argument("--image", default=str(OUTPUT_DIR / "cropped_input.png"))
    p.add_argument("--scene", default=str(OUTPUT_DIR / "scene.json"))
    p.add_argument("--labels", default=str(OUTPUT_DIR / "region_labels.png"))
    p.add_argument("--out", default=str(OUTPUT_DIR / "sprites"))
    args = p.parse_args()
    n = build_sprites(args.image, args.scene, args.labels, args.out)
    print(f"Wrote {n} LDI sprites -> {args.out}/sprite_00..{n-1:02d}.png")


if __name__ == "__main__":
    main()
