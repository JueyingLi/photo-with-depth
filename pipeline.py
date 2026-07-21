"""一张照片 → 一个案例文件夹(可被网页编辑器直接加载)。

把现有流水线串成一个可调用函数,供桌面 app 后端(app.py)使用:
  裁边 → 深度 → SAM+valley 分层 → 背景补全 → LDI 贴图
产物写进 case_dir:cropped_input.png / depth_map.png / region_labels.png /
scene.json / background.png / sprites/sprite_00..NN.png
"""
from pathlib import Path

import numpy as np
from PIL import Image

from step_1_crop_frame import remove_white_frame
from step_2_build_depth_map import build_depth_map, save_depth_image
from objects import run_sam2_masks, build_sam_valley_regions, save_segments
from regions import save_scene
from build_background import build_background
from build_sprites import build_sprites


def process_image(input_path, case_dir, points_per_side: int = 32, progress=None):
    """跑完整流水线,产物写进 case_dir,返回 case_dir。

    progress: 可选回调 progress(step:int, total:int, msg:str),给前端显示进度。
    """
    def tick(i, msg):
        if progress:
            progress(i, 5, msg)

    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    tick(1, "Cropping the frame")
    img = remove_white_frame(Image.open(input_path).convert("RGB"))
    img.save(case_dir / "cropped_input.png")

    tick(2, "Estimating depth")
    depth = build_depth_map(img)
    save_depth_image(depth, case_dir / "depth_map.png")

    tick(3, "SAM segmentation + valley slicing")
    masks, _ = run_sam2_masks(img, points_per_side=points_per_side)
    label_map, regions = build_sam_valley_regions(np.asarray(depth, np.float32) if not isinstance(depth, np.ndarray) else depth, masks)
    save_scene(label_map, regions, case_dir)   # 写 scene.json + region_labels.png
    save_segments(masks, case_dir / "segments.png")   # SAM 分割图,供网页「SAM 选物体」

    tick(4, "Inpainting the backplate")
    build_background(case_dir / "cropped_input.png", case_dir / "scene.json",
                     case_dir / "region_labels.png", case_dir / "background.png")

    tick(5, "Baking layer sprites")
    build_sprites(case_dir / "cropped_input.png", case_dir / "scene.json",
                  case_dir / "region_labels.png", case_dir / "sprites")
    return case_dir
