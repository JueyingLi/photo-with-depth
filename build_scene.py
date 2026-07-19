"""Fast region export from already-generated outputs.

Reads outputs/cropped_input.png + outputs/depth_map.png (produced by
generate_depth_photo.py) and writes region_labels.png + scene.json for the
browser editor. Does NOT import torch, so it runs in a second.

Usage:
    python build_scene.py
    python build_scene.py --bands 8 --min-area 0.003
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from step_3_build_regions import build_regions
from regions import save_scene

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description="Build region label map + scene.json from existing depth output")
    p.add_argument("--depth", default=str(OUTPUT_DIR / "depth_map.png"))
    p.add_argument("--clusters", type=int, default=None, help="Force this many depth clusters (default: auto-pick via elbow)")
    p.add_argument("--bands", type=int, default=None, help="Legacy: uniform depth bands instead of clustering")
    p.add_argument("--min-area", type=float, default=0.002, help="Min region area as fraction of image")
    return p.parse_args()


def main():
    args = parse_args()
    depth_path = Path(args.depth)
    if not depth_path.exists():
        raise SystemExit(f"Depth map not found: {depth_path}\nRun generate_depth_photo.py first.")

    depth = np.asarray(Image.open(depth_path).convert("L"), dtype=np.float32) / 255.0
    label_map, regions = build_regions(
        depth, n_clusters=args.clusters, n_bands=args.bands, min_area_frac=args.min_area
    )
    scene_path = save_scene(label_map, regions, OUTPUT_DIR)

    if args.bands:
        how = f"{args.bands} uniform depth bands"
    elif args.clusters:
        how = f"{args.clusters} forced depth clusters"
    else:
        how = "auto depth clusters (elbow)"
    print(f"Segmented {len(regions)} regions via {how}")
    for r in regions:
        print(f"  #{r['id']:2d} {r['label']:22s} depth {r['depth_min']:.2f}-{r['depth_max']:.2f}"
              f" ({r['depth_mode']}, {r['area_frac']*100:.1f}%)")
    print(f"Wrote {scene_path}")
    print(f"Wrote {OUTPUT_DIR / 'region_labels.png'}")


if __name__ == "__main__":
    main()
