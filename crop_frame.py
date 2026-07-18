"""裁掉画作四周的画框/白边 —— 独立模块,方便单独调参和测试。

快速测试:
    python crop_frame.py                          # 用默认样例
    python crop_frame.py --input path/to/img.png  # 换图
    python crop_frame.py --input img.png --window 21 --pad 6 --save out.png

会打印检测到的裁剪框,并把裁剪结果存到 --save(默认 outputs/debug_crop.png),
方便你改完参数立刻看效果。
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "examples" / "intro-park-entrance.png"


def remove_white_frame(image: Image.Image, window: int = 15, pad: int = 0,
                       flat_tol: float = 12.0, flat_frac: float = 0.85) -> Image.Image:
    """裁掉画作四周的画框/衬纸(白框、黑框、灰边都行)。

    核心思想:画面内容有纹理 → 局部方差大;画框/衬纸颜色均匀 → 方差小。
    判据是「均匀度」而非「亮暗」,所以对白框、黑框、灰衬纸一视同仁。

    边线是否算「空白边」用的是鲁棒判据:整条里 ≥flat_frac 的像素都接近中位色
    (差 <flat_tol)就算边框——这样即使有几行内容穿过(树枝伸到边),也不会漏削。

    参数:
        window    : 算局部方差的窗口大小(越大越平滑,对细纹理越不敏感)
        pad       : 检测出边界后往外多留的像素(默认 0;设 >0 会把边框重新包一点回来)
        flat_tol  : 判「接近中位色」的灰度容差(越大越容易判为均匀)
        flat_frac : 整条里需要多大比例接近中位色才算空白边(越小削得越狠)
    """
    img = np.array(image)          # PIL 图 -> numpy 数组 (H, W, 3)
    if img.ndim != 3:              # 不是彩色图(比如灰度/带透明)就不处理
        return image

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)  # 转灰度并转 float,防平方溢出

    # 用方差公式 Var = E[x²] - (E[x])² 算每个像素的"局部标准差"。
    # boxFilter = 在 window×window 窗口里求平均,很快。
    mean = cv2.boxFilter(gray, -1, (window, window), normalize=True)          # 邻域均值 E[x]
    sq = cv2.boxFilter(gray * gray, -1, (window, window), normalize=True)     # 邻域 E[x²]
    activity = np.sqrt(np.clip(sq - mean * mean, 0, None))  # 标准差;clip 防浮点误差出现负数

    # 把 2D 纹理图压成两条 1D 曲线:每列的纹理强度、每行的纹理强度
    col_act = activity.mean(axis=0)  # 沿行平均 → 长度=宽,描述横向哪里有内容
    row_act = activity.mean(axis=1)  # 沿列平均 → 长度=高,描述纵向哪里有内容

    def content_run(profile: np.ndarray, frac: float = 0.15):
        # 在一条曲线上找"纹理超过阈值"的第一个和最后一个位置 = 内容区间
        thr = profile.min() + (profile.max() - profile.min()) * frac  # 阈值取范围的 15%
        idx = np.where(profile > thr)[0]  # 满足条件的下标数组([0] 因为 where 返回元组)
        if len(idx) == 0:
            return 0, len(profile)        # 全平:退回整幅
        return int(idx.min()), int(idx.max()) + 1  # 首/尾下标 → [起, 止)

    x0, x1 = content_run(col_act)  # 左右边界
    y0, y1 = content_run(row_act)  # 上下边界

    # 精修:上一步是粗框,这里从每条边往里走,只要那条边线"大部分是同一个色"就继续削。
    # 用中位色 + 比例判据(鲁棒),几行内容穿过也不影响。与明暗无关(白/黑/灰都算)。
    def uniform_strip(line: np.ndarray) -> bool:
        med = np.median(line)
        return float(np.mean(np.abs(line - med) < flat_tol)) >= flat_frac

    while x0 < x1 - 10 and uniform_strip(gray[y0:y1, x0]):        # 削左:第 x0 列
        x0 += 1
    while x1 > x0 + 10 and uniform_strip(gray[y0:y1, x1 - 1]):    # 削右
        x1 -= 1
    while y0 < y1 - 10 and uniform_strip(gray[y0, x0:x1]):        # 削上:第 y0 行
        y0 += 1
    while y1 > y0 + 10 and uniform_strip(gray[y1 - 1, x0:x1]):    # 削下
        y1 -= 1

    # 往外留一点边距,避免把内容边缘也削掉;并夹在图像范围内
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)

    # 兜底:检测结果小得离谱(说明算法失败),退回"固定裁掉 6% 边",而不是返回垃圾
    if x1 - x0 < 0.3 * w or y1 - y0 < 0.3 * h:
        border = max(8, int(min(w, h) * 0.06))
        return image.crop((border, border, w - border, h - border))

    print(f"Content crop: ({x0}, {y0}, {x1}, {y1}) from ({w}, {h})")
    return image.crop((x0, y0, x1, y1))  # PIL 的 crop 参数是 (左, 上, 右, 下)


def main():
    p = argparse.ArgumentParser(description="Test the frame-cropping in isolation")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--window", type=int, default=15, help="局部方差窗口大小")
    p.add_argument("--pad", type=int, default=0, help="边界外留的像素(默认 0)")
    p.add_argument("--flat-tol", type=float, default=12.0, help="判『接近中位色』的灰度容差")
    p.add_argument("--flat-frac", type=float, default=0.85, help="整条需多大比例接近中位色才算空白边")
    p.add_argument("--save", default=str(ROOT / "outputs" / "debug_crop.png"))
    args = p.parse_args()

    image = Image.open(args.input).convert("RGB")
    cropped = remove_white_frame(image, window=args.window, pad=args.pad,
                                 flat_tol=args.flat_tol, flat_frac=args.flat_frac)
    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out)
    print(f"原图 {image.size} -> 裁剪后 {cropped.size},已存到 {out}")


if __name__ == "__main__":
    main()
