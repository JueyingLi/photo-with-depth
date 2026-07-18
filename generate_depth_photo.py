"""流水线入口:读图 → 裁边 → 估深度 → 分层,产出 cropped_input.png / depth_map.png / scene.json。

更高级的「SAM 物体分层 + 背景补全 + LDI 贴图」在 build_objects.py / build_sprites.py。
本文件里的 build_layers 是早期的「烤死 4 层」老方案,现已被 LDI 取代(编辑器不再读它)。
"""
import argparse
from pathlib import Path

import cv2          # OpenCV:颜色转换、盒式滤波、remap 重映射
import numpy as np  # 数组运算,整套代码的骨架
import torch        # 跑深度模型
from PIL import Image  # 读写图片
from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # Depth Anything V2

from regions import build_regions, save_scene  # ③分层(深度聚类,见 regions.py)
from crop_frame import remove_white_frame       # ①裁边(独立模块,见 crop_frame.py)


# Path(__file__).resolve().parent:不管从哪个目录运行,路径都相对脚本自身,不会找不到文件
ROOT = Path(__file__).resolve().parent
INPUT_IMAGE = ROOT / "examples" / "intro-park-entrance.png"  # 默认输入图
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DEPTH = OUTPUT_DIR / "depth_map.png"       # 灰度深度图
OUTPUT_CROPPED = OUTPUT_DIR / "cropped_input.png" # 裁边后的原图(后续所有步骤都用它)
OUTPUT_GIF = OUTPUT_DIR / "depth_parallax.gif"
OUTPUT_LAYERS_DIR = OUTPUT_DIR / "layers"         # 老方案的烤层输出
HF_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"  # HuggingFace 上的模型名


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


def load_depth_map(image: Image.Image):
    """用 Depth Anything V2 估计单目深度,返回 (图, 0~1 深度数组)。

    只负责"估深度"这一件事:传进来的图应当已经裁好边(裁边在 main 里做)。
    """
    # 有 GPU 就用 GPU,否则 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using Hugging Face depth model: {HF_DEPTH_MODEL}")

    # HuggingFace 标准套路:processor 负责缩放/归一化成模型输入,model 出深度
    processor = AutoImageProcessor.from_pretrained(HF_DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(HF_DEPTH_MODEL).to(device)
    model.eval()  # 推理模式(关掉 dropout 等)

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():                      # 推理不需要梯度,省内存/加速
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth  # 模型输出的低分辨率深度

    # 把低分辨率深度双三次插值放大回原图尺寸
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),   # (B,H,W) -> (B,1,H,W),interpolate 要有通道维
        size=image.size[::-1],          # ⚠️ PIL 的 size 是 (宽,高),取反成 (高,宽)
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()           # 去掉多余维度 → 搬回 CPU → 转 numpy

    # 归一化到 0~1:0=最远,1=最近。后面分层/视差都依赖这个约定。
    depth = np.clip(prediction, 1e-4, None)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    print("Depth model loaded successfully")
    return image, depth


def build_layers(image: Image.Image, depth: np.ndarray, num_layers: int = 4):
    """⚠️ 老方案:把深度切成 num_layers 个"带",每带做一次水平视差 warp,烤成 RGBA。

    现在编辑器改用 build_sprites.py 的 LDI 分层贴图,这段仅作历史/教学保留。
    但它展示了视差最朴素的原理:位移量正比于深度。
    """
    img = np.array(image.convert("RGB"))
    h, w = depth.shape
    # 生成坐标网格:xx[i,j]=j(列号),yy[i,j]=i(行号),后面按深度移动像素要用
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    depth_norm = np.clip(depth, 1e-4, None)
    depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min())
    depth_norm = cv2.GaussianBlur(depth_norm.astype(np.float32), (0, 0), 1.2)  # 轻微模糊,减少带边缘锯齿

    layers = []
    for layer_idx in range(num_layers):
        # 按深度切"带":把 0~1 分成 num_layers 段,只保留落在本段的像素(其余透明)
        if layer_idx == num_layers - 1:
            band_mask = (depth_norm < 0.35).astype(np.uint8)   # 最后一层=最远的背景
        else:
            start = layer_idx / max(1, num_layers)
            end = (layer_idx + 1) / max(1, num_layers)
            band_mask = ((depth_norm >= start) & (depth_norm < end)).astype(np.uint8)

        if layer_idx == 0:
            band_mask = (depth_norm >= 0.65).astype(np.uint8)  # 第一层=最近的前景

        alpha = band_mask * 255                       # 掩码 0/1 -> 透明度 0/255
        rgba = np.dstack([img, alpha]).astype(np.uint8)  # 未使用,仅示意

        # 计算每个像素的水平位移:越近(depth大)、越靠前的层,移动越多 = 视差
        if layer_idx == num_layers - 1:
            offset = np.zeros_like(depth_norm, dtype=np.float32)  # 背景不动
        else:
            t = layer_idx / max(1, num_layers - 1)
            motion_scale = 0.6 + 0.8 * (1.0 - t)
            offset = ((depth_norm - 0.5) * motion_scale * 8).astype(np.float32)

        # cv2.remap:按 (map_x, map_y) 把源像素搬到新位置。这是"视差"的核心操作。
        map_x = xx.astype(np.float32) + offset  # 目标 x = 原 x + 位移
        map_y = yy.astype(np.float32)           # y 不动
        map_x = np.clip(map_x, 0, w - 1)        # 别采样到图像外
        map_y = np.clip(map_y, 0, h - 1)
        map_xy = np.stack([map_x, map_y], axis=-1)
        warped = cv2.remap(
            img,
            map_xy,
            None,
            interpolation=cv2.INTER_LINEAR,       # 双线性插值
            borderMode=cv2.BORDER_REPLICATE,      # 边缘外用最近像素填
        )

        # 组装成 RGBA:warp 后的颜色 + 本层掩码作为透明度
        warped_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        warped_rgba[:, :, :3] = warped
        warped_rgba[:, :, 3] = alpha
        layers.append(Image.fromarray(warped_rgba, mode="RGBA"))

    return layers


def save_layers(layers, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, layer in enumerate(layers):
        layer.save(output_dir / f"layer_{index + 1:02d}.png")  # :02d = 补零成两位,便于排序


def save_depth_image(depth: np.ndarray):
    depth_vis = (depth * 255).astype(np.uint8)  # 0~1 浮点 -> 0~255 灰度,才能存成 PNG
    Image.fromarray(depth_vis).save(OUTPUT_DEPTH)


def save_gif(frames, output_path: Path):
    # 注意:main 里没有调用它,是段未使用代码
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,  # 每帧毫秒
        loop=0,        # 0 = 无限循环
    )


def main():
    args = parse_args()
    input_image = Path(args.input).expanduser().resolve()  # 展开 ~ 并转绝对路径
    ensure_output_dir()
    if OUTPUT_CROPPED.exists() and input_image != OUTPUT_CROPPED:
        print(f"Using cropped input from {OUTPUT_CROPPED}")

    image = Image.open(input_image).convert("RGB")
    image = remove_white_frame(image)   # ①裁边(独立一步)
    image.save(OUTPUT_CROPPED)          # 存 cropped_input.png(后续都用它)

    image, depth = load_depth_map(image)  # ②估深度(只做深度)
    save_depth_image(depth)               # 存 depth_map.png

    layers = build_layers(image, depth, num_layers=4)  # (老)烤 4 层视差
    save_layers(layers, OUTPUT_LAYERS_DIR)

    label_map, regions = build_regions(depth)     # ③按深度聚类分层(自动决定层数)
    save_scene(label_map, regions, OUTPUT_DIR)    # 存 region_labels.png + scene.json

    print(f"Saved depth map to {OUTPUT_DEPTH}")
    print(f"Segmented {len(regions)} editable regions -> {OUTPUT_DIR / 'scene.json'}")
    print(f"Saved cropped input to {OUTPUT_CROPPED}")
    print(f"Saved {len(layers)} layered images to {OUTPUT_LAYERS_DIR}")
    print(f"Source image: {input_image}")


if __name__ == "__main__":  # 只有直接运行本文件才执行 main(),被 import 时不执行
    main()
