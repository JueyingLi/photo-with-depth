"""Object-aware scene build: SAM 2 everything-mode masks fused with the depth map.

Reads outputs/cropped_input.png + outputs/depth_map.png, runs SAM 2 to segment
objects, fuses them with depth into whole-object regions (depth-cluster fill for
the rest), and writes region_labels.png + scene.json for the editor.

SAM masks are cached to outputs/sam_masks.npz so you can re-tune the fusion
(merge thresholds, min area) without re-running the model.

    python build_objects.py                 # run SAM (or use cache) then fuse
    python build_objects.py --refresh-sam   # force re-run SAM
    python build_objects.py --min-area 0.001 --contain 0.7
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from objects import (DEFAULT_MODEL, build_hybrid_regions, build_layer_groups,
                     build_sam_valley_regions, run_sam2_masks)
from regions import save_scene

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MASK_CACHE = OUTPUT_DIR / "sam_masks.npz"


def parse_args():
    p = argparse.ArgumentParser(description="Fuse SAM 2 object masks with the depth map")
    p.add_argument("--image", default=str(OUTPUT_DIR / "cropped_input.png"))
    p.add_argument("--depth", default=str(OUTPUT_DIR / "depth_map.png"))
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF SAM2 checkpoint id (tiny/small/base-plus/large)")
    p.add_argument("--points-per-side", type=int, default=32, help="Grid density; higher finds more objects, slower")
    p.add_argument("--refresh-sam", action="store_true", help="Re-run SAM even if a cache exists")
    p.add_argument("--min-area", type=float, default=0.0015, help="Min object area (fraction of image)")
    p.add_argument("--max-area", type=float, default=0.85, help="Max object area (drops full-frame masks)")
    p.add_argument("--contain", type=float, default=0.75, help="Overlap fraction to merge a part into an object")
    p.add_argument("--depth-merge", type=float, default=0.12, help="Max depth gap to merge parts")
    p.add_argument("--per-object", action="store_true",
                   help="每个物体一个区域(build_hybrid_regions)")
    p.add_argument("--layer-groups", action="store_true",
                   help="旧的层组模式(build_layer_groups)")
    p.add_argument("--groups", type=int, default=None,
                   help="layer-groups 模式:合并成几层(默认自动)")
    p.add_argument("--dominant-frac", type=float, default=0.9,
                   help="默认模式:物体 ≥ 这个比例在同一层就整块归层")
    p.add_argument("--extract-area", type=float, default=0.03,
                   help="默认模式:面积 ≥ 这个比例且深度均匀的整体独立成层")
    return p.parse_args()


def load_or_run_masks(image_path: Path, model: str, refresh: bool, points_per_side: int):
    if MASK_CACHE.exists() and not refresh:
        data = np.load(MASK_CACHE)
        masks = [data[k] for k in sorted(data.files, key=lambda s: int(s.split("_")[1]))]
        print(f"Loaded {len(masks)} cached SAM masks from {MASK_CACHE}")
        return masks
    image = Image.open(image_path).convert("RGB")
    print(f"Running SAM 2 ({model}, {points_per_side}x{points_per_side} pts) on CPU — this can take a few minutes…")
    masks, scores = run_sam2_masks(image, model_id=model, points_per_side=points_per_side)
    np.savez_compressed(MASK_CACHE, **{f"m_{i}": m for i, m in enumerate(masks)})
    print(f"SAM produced {len(masks)} raw masks (cached to {MASK_CACHE})")
    return masks


def main():
    args = parse_args()
    depth = np.asarray(Image.open(args.depth).convert("L"), dtype=np.float32) / 255.0
    masks = load_or_run_masks(Path(args.image), args.model, args.refresh_sam, args.points_per_side)

    if args.per_object:
        label_map, regions = build_hybrid_regions(
            depth, masks,
            min_area_frac=args.min_area, max_area_frac=args.max_area,
            contain_thresh=args.contain, depth_merge=args.depth_merge,
        )
        n_obj = sum(1 for r in regions if r["source"] == "object")
        print(f"Fused into {len(regions)} regions: {n_obj} objects + {len(regions)-n_obj} depth-background")
    elif args.layer_groups:
        label_map, regions, centers = build_layer_groups(
            depth, masks, n_groups=args.groups,
            min_area_frac=args.min_area, max_area_frac=args.max_area,
        )
        print(f"Merged into {len(regions)} depth-layer groups (objects snapped to their dominant layer)")
    else:  # 默认:notebook 的 valley + SAM snap(90%) + extract(平整整体)
        label_map, regions = build_sam_valley_regions(
            depth, masks, min_area_frac=args.min_area,
            dominant_frac_thresh=args.dominant_frac, extract_area_frac=args.extract_area,
        )
        print(f"valley+SAM: {len(regions)} 区域 (snap≥{args.dominant_frac}, extract大整体≥{args.extract_area})")
    save_scene(label_map, regions, OUTPUT_DIR)
    from build_background import build_background
    n_mov = build_background(args.image, OUTPUT_DIR / "scene.json",
                             OUTPUT_DIR / "region_labels.png", OUTPUT_DIR / "background.png")
    from build_sprites import build_sprites
    build_sprites(args.image, OUTPUT_DIR / "scene.json",
                  OUTPUT_DIR / "region_labels.png", OUTPUT_DIR / "sprites")

    for r in regions:
        print(f"  #{r['id']:2d} {r['label']:28s} depth {r['depth_min']:.2f}-{r['depth_max']:.2f} ({r['area_frac']*100:.1f}%)")
    print(f"Wrote scene.json, region_labels.png, and background.png ({n_mov} movers inpainted)")


if __name__ == "__main__":
    main()
