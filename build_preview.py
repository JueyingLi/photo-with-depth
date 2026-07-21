"""Render a parallax preview GIF from the depth map (far plane pinned).

Displacement is proportional to depth, anchored at the far plane: the farthest
layer (depth 0) doesn't move, and everything nearer moves more the closer it is —
matching the editor's default. Optionally freeze a far background band for a hard
cut between a static backdrop and moving foreground.

    python build_preview.py
    python build_preview.py --amp 28 --frames 24 --size 480
    python build_preview.py --freeze-bg 0.2      # lock everything with depth < 0.2
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Render a depth parallax preview GIF")
    p.add_argument("--image", default=str(OUTPUT_DIR / "cropped_input.png"))
    p.add_argument("--depth", default=str(OUTPUT_DIR / "depth_map.png"))
    p.add_argument("--out", default=str(OUTPUT_DIR / "parallax_preview.gif"))
    p.add_argument("--amp", type=float, default=12.0, help="Max pixel sway of the nearest layer")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=360, help="Output GIF square size")
    p.add_argument("--colors", type=int, default=64, help="GIF palette size (smaller = lighter file)")
    p.add_argument("--freeze-bg", type=float, default=None,
                   help="Freeze all pixels with depth below this (hard static backdrop)")
    p.add_argument("--snap", action="store_true",
                   help="Move each region as one flat plane at its depth (uses scene.json layer groups)")
    p.add_argument("--layered", action="store_true",
                   help="Ghost-free: move each layer as an inpainted cutout sprite, composite far->near")
    p.add_argument("--fg-thresh", type=float, default=0.4,
                   help="layered mode: regions with depth_mean >= this are moving foreground cutouts")
    p.add_argument("--scene", default=str(OUTPUT_DIR / "scene.json"))
    p.add_argument("--labels", default=str(OUTPUT_DIR / "region_labels.png"))
    p.add_argument("--background", default=None,
                   help="layered mode: background plate (default: background.png beside --scene)")
    return p.parse_args()


def _translate(arr, dx, border=cv2.BORDER_REPLICATE):
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]), borderMode=border,
                          flags=cv2.INTER_LINEAR)


def layered_frames(img, scene, lab, amp, n_frames, plate_path):
    """Ghost-free parallax matching the editor: slide each tagged mover cutout
    (far->near) over the prebuilt background plate (bleed/LaMa-filled)."""
    background = np.asarray(Image.open(plate_path).convert("RGB"))
    movers = sorted([r for r in scene["regions"] if r.get("mover")], key=lambda r: r["depth_mean"])

    sprites = []
    for r in movers:
        alpha = cv2.GaussianBlur((lab == r["id"]).astype(np.uint8) * 255, (0, 0), 1.0)
        sprites.append((r["depth_mean"], alpha))

    frames = []
    for k in range(n_frames):
        par = float(np.sin(2 * np.pi * k / n_frames))
        canvas = background.astype(np.float32).copy()
        for d, alpha in sprites:                       # far -> near, nearer painted last
            dx = d * amp * par
            ws = _translate(img, dx).astype(np.float32)
            wa = (_translate(alpha, dx).astype(np.float32) / 255.0)[..., None]
            canvas = ws * wa + canvas * (1 - wa)
        frames.append(np.clip(canvas, 0, 255).astype(np.uint8))
    return frames


def snapped_depth(scene_path, labels_path, shape):
    """Per-pixel depth where every pixel takes its region's layer depth (flat planes)."""
    import json
    scene = json.load(open(scene_path))
    lab = np.asarray(Image.open(labels_path).convert("L"))
    out = np.zeros(shape, np.float32)
    for r in scene["regions"]:
        out[lab == r["id"]] = r["depth_mean"]
    return out


def main():
    args = parse_args()
    import json
    img = np.asarray(Image.open(args.image).convert("RGB"))

    if args.layered:
        # Ghost-free: cutouts move with their alpha over the prebuilt plate.
        # Depth isn't read here — the sprites already carry their layer order.
        scene = json.load(open(args.scene))
        lab = np.asarray(Image.open(args.labels).convert("L"))
        # Plate lives beside the scene it belongs to, so --scene from a case dir stays consistent.
        plate = args.background or str(Path(args.scene).parent / "background.png")
        print(f"layered mode: {sum(1 for r in scene['regions'] if r.get('mover'))}"
              f" mover cutouts over {Path(plate).name}")
        rgb_frames = layered_frames(img, scene, lab, args.amp, args.frames, plate)
    else:
        depth = np.asarray(Image.open(args.depth).convert("L")).astype(np.float32) / 255.0
        H, W = depth.shape
        if args.snap:
            depth = snapped_depth(args.scene, args.labels, (H, W))
            print("snapped mode: each region moves as one flat plane at its layer depth")
        motion = depth.copy()
        if args.freeze_bg is not None:
            motion[depth < args.freeze_bg] = 0.0
        motion = cv2.GaussianBlur(motion, (0, 0), 3)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        xx = xx.astype(np.float32); yy = yy.astype(np.float32)
        rgb_frames = []
        for k in range(args.frames):
            par = float(np.sin(2 * np.pi * k / args.frames))
            disp = args.amp * par * motion
            map_x = np.ascontiguousarray(np.clip(xx + disp, 0, W - 1), dtype=np.float32)
            map_y = np.ascontiguousarray(yy, dtype=np.float32)
            rgb_frames.append(cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE))

    frames = [Image.fromarray(cv2.resize(f, (args.size, args.size), interpolation=cv2.INTER_AREA))
              .convert("P", palette=Image.ADAPTIVE, colors=args.colors) for f in rgb_frames]
    out = Path(args.out)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
    kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({args.frames} frames, {args.size}px, {kb:.0f} KB)")


if __name__ == "__main__":
    main()
