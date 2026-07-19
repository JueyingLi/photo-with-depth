"""裁边(step_1_crop_frame.remove_white_frame)的单元测试。

用 examples/intro-park-entrance.png 作为样例:一幅有浅色衬纸边框的油画。
只依赖 cv2/numpy/PIL,不加载深度模型,跑得很快。
"""
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from step_1_crop_frame import remove_white_frame

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "intro-park-entrance.png"


def border_uniformity(arr: np.ndarray, k: int = 6) -> float:
    """四条边(各 k 像素宽)的灰度标准差均值:越小 = 边越均匀(越像空白边框)。"""
    g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edges = [g[:k, :], g[-k:, :], g[:, :k], g[:, -k:]]
    return float(np.mean([e.std() for e in edges]))


class RemoveWhiteFrameTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.open(SAMPLE).convert("RGB")

    def test_crop_is_smaller(self):
        # 裁掉边框后,宽高都应变小
        cropped = np.array(remove_white_frame(self.image))
        self.assertLess(cropped.shape[0], self.image.size[1])
        self.assertLess(cropped.shape[1], self.image.size[0])

    def test_keeps_most_of_the_picture(self):
        # 不能裁过头:内容应保留原图大半(每维 > 50%)
        cropped = np.array(remove_white_frame(self.image))
        self.assertGreater(cropped.shape[1], 0.5 * self.image.size[0])
        self.assertGreater(cropped.shape[0], 0.5 * self.image.size[1])

    def test_removes_uniform_border(self):
        # 核心:裁完后的边应该比原图的边"更有内容"(标准差更大 = 均匀边框被削掉了)
        orig = np.array(self.image)
        cropped = np.array(remove_white_frame(self.image))
        self.assertGreater(border_uniformity(cropped), border_uniformity(orig))

    def test_flat_frac_monotonic(self):
        # flat_frac 越小越容易判为空白边 → 削得越狠 → 裁出的图不会更大
        aggressive = np.array(remove_white_frame(self.image, flat_frac=0.6))
        conservative = np.array(remove_white_frame(self.image, flat_frac=0.95))
        self.assertLessEqual(aggressive.shape[0], conservative.shape[0])
        self.assertLessEqual(aggressive.shape[1], conservative.shape[1])


if __name__ == "__main__":
    unittest.main()
