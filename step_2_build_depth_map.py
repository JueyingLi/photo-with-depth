"""第 2 步:从一张(已裁好的)图,用 Depth Anything V2 估计深度图。

输入:一张 RGB 图(通常是第 1 步裁好的 outputs/cropped_input.png)
输出:
  - 内存里:一个和图同尺寸的 numpy 数组,值域 0~1(0=最远, 1=最近)
  - 磁盘上:outputs/depth_map.png —— 把上面的数组存成灰度图(白=近, 黑=远)

单独测试:
    python step_2_build_depth_map.py                          # 用 outputs/cropped_input.png
    python step_2_build_depth_map.py --input path/to/img.png  # 换图
    python step_2_build_depth_map.py --save outputs/depth_map.png
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parent
HF_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def build_depth_map(image: Image.Image) -> np.ndarray:
    """用 Depth Anything V2 估计单目深度,返回和图同尺寸的 0~1 深度数组。

    只做"估深度"一件事:传进来的图应当已经裁好边。0=最远, 1=最近,
    这个 0~1 约定是后面分层/视差的基础。
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

    # 归一化到 0~1
    depth = np.clip(prediction, 1e-4, None)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    print("Depth model loaded successfully")
    return depth


def save_depth_image(depth: np.ndarray, path: Path):
    depth_vis = (depth * 255).astype(np.uint8)  # 0~1 浮点 -> 0~255 灰度,才能存成 PNG
    Image.fromarray(depth_vis).save(path)


def main():
    p = argparse.ArgumentParser(description="Step 2: estimate a depth map with Depth Anything V2")
    p.add_argument("--input", default=str(ROOT / "outputs" / "cropped_input.png"))
    p.add_argument("--save", default=str(ROOT / "outputs" / "depth_map.png"))
    args = p.parse_args()

    image = Image.open(args.input).convert("RGB")
    depth = build_depth_map(image)
    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_depth_image(depth, out)
    print(f"输入 {image.size} -> 深度数组 {depth.shape} (min={depth.min():.2f}, max={depth.max():.2f})")
    print(f"已存深度图到 {out}")


if __name__ == "__main__":
    main()
