"""流水线入口:读图 → 裁边 → 估深度 → 分层,产出 cropped_input.png / depth_map.png / scene.json。

更高级的「SAM 物体分层 + 背景补全 + LDI 贴图」在 build_objects.py / build_sprites.py。
"""
import argparse
from pathlib import Path

from PIL import Image  # 读写图片

from step_1_crop_frame import remove_white_frame              # ①裁边(见 step_1_crop_frame.py)
from step_2_build_depth_map import build_depth_map, save_depth_image  # ②估深度(见 step_2_build_depth_map.py)
from step_3_build_regions import build_regions                # ③分层(见 step_3_build_regions.py)
from regions import save_scene                                # 保存 scene.json / region_labels.png


# Path(__file__).resolve().parent:不管从哪个目录运行,路径都相对脚本自身,不会找不到文件
ROOT = Path(__file__).resolve().parent
INPUT_IMAGE = ROOT / "examples" / "intro-park-entrance.png"  # 默认输入图
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DEPTH = OUTPUT_DIR / "depth_map.png"       # 灰度深度图
OUTPUT_CROPPED = OUTPUT_DIR / "cropped_input.png" # 裁边后的原图(后续所有步骤都用它)


def ensure_output_dir():
    # parents=True:父目录不存在也一起建;exist_ok=True:已存在不报错
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a depth-based photo effect using Depth Anything V2")
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_IMAGE),
        help="Path to the input image. Defaults to the sample image in examples/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_image = Path(args.input).expanduser().resolve()  # 展开 ~ 并转绝对路径
    ensure_output_dir()

    image = Image.open(input_image).convert("RGB")
    image = remove_white_frame(image)      # ①裁边
    image.save(OUTPUT_CROPPED)             # 存 cropped_input.png(后续都用它)

    depth = build_depth_map(image)         # ②估深度
    save_depth_image(depth, OUTPUT_DEPTH)  # 存 depth_map.png

    label_map, regions = build_regions(depth)   # ③按深度聚类分层(自动决定层数)
    save_scene(label_map, regions, OUTPUT_DIR)  # 存 region_labels.png + scene.json

    print(f"Saved depth map to {OUTPUT_DEPTH}")
    print(f"Segmented {len(regions)} editable regions -> {OUTPUT_DIR / 'scene.json'}")
    print(f"Saved cropped input to {OUTPUT_CROPPED}")
    print(f"Source image: {input_image}")


if __name__ == "__main__":  # 只有直接运行本文件才执行 main(),被 import 时不执行
    main()
