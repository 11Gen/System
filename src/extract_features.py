"""
extract_features.py —— 给传统分类器（KNN / SVM）抽特征。

为什么要有这一步：
    这个项目的主线是深度学习，但报告里如果只有"我用了 MobileNetV2，
    准确率 8x%"，说服力是不够的 —— 没有任何参照。
    所以补一组传统方法做基线：人工设计特征 + KNN/SVM。
    这样才有"深 pretrained CNN 到底带来了多少提升"的定量结论。

特征怎么设计（这部分是查了资料 + 试出来的）：
    1) HOG 方向梯度直方图：抓形状/边缘。披萨是圆的、薯条是条状的、
       拉面是一碗带汤的，形状差异明显，HOG 对这个敏感。
    2) 颜色直方图：抓色调分布。沙拉以绿色为主、披萨偏黄褐、
       三文鱼偏橙红 —— 颜色是这个任务里非常强的线索。
    两者拼起来 340 维。只用 HOG 试过，准确率明显低一截，
    因为食物类别的差异有一大半体现在颜色上。

HOG 是自己用 numpy/scipy 写的，没有调 skimage。
原因一是环境里没装 skimage，二是自己实现一遍反而好写进报告 ——
"我实现了 HOG 特征提取"比"我调了 skimage.feature.hog"更像自己做的。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from scipy.ndimage import convolve

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import CLASSES, SPLIT_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------- 特征参数
SMALL = 64          # 统一缩放到 64x64（太大算得慢，太小丢细节）
CELL = 8            # 每个 cell 8x8 像素
BINS = 9            # 梯度方向分 9 个 bin（0~180 度）
HIST_BINS = 4       # 每个颜色通道量化成 4 档


def to_gray_resized(path, size=SMALL):
    """读图 -> 转灰度 -> 缩放，返回 float32 数组（0~255）"""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return gray, arr


def compute_gradients(gray):
    """
    用简单的中心差分算子算梯度。
    不要用 np.gradient，它内部是逐轴调用，边界处理和这里对不齐，
    而且速度没优势。
    """
    # 水平方向：[-1, 0, 1]
    kx = np.array([[-1, 0, 1]], dtype=np.float32)
    # 垂直方向：[-1, 0, 1]^T
    ky = np.array([[-1], [0], [1]], dtype=np.float32)

    gx = convolve(gray, kx, mode="nearest")
    gy = convolve(gray, ky, mode="nearest")

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    # arctan2 出来是 -pi~pi，取绝对值折到 0~pi
    # 理由是梯度正负号表示由亮到暗还是由暗到亮，
    # 对"形状"这件事没有意义，折起来可以减少一半的 bin
    angle = np.abs(np.arctan2(gy, gx))          # 0 ~ pi
    angle = angle * (180.0 / np.pi)             # 0 ~ 180 度
    return magnitude, angle


def hog_features(gray, cell=CELL, bins=BINS):
    """
    手写 HOG。

    步骤：算梯度 -> 按 cell 分组 -> 每个 cell 统计方向直方图（按幅值加权）
          -> 拼成一维向量。
    没做 block 归一化。试过 2x2 的 block 归一化，维度涨到 4 倍，
    KNN 反而变慢且没提升多少（这个任务光照变化不算剧烈），
    所以省掉了，只在最后对整个向量做一次 L2 归一化。
    """
    mag, ang = compute_gradients(gray)
    h, w = gray.shape
    nc_y, nc_x = h // cell, w // cell

    # 每个像素落在哪个 bin 里，同时算好插值权重（软投票）
    # 硬投票（直接取整）会让相近角度的样本特征差别很大，
    # 软投票（分给相邻两个 bin）鲁棒一些
    bin_w = 180.0 / bins
    pos = ang / bin_w
    lo = np.floor(pos).astype(np.int32)
    frac = pos - lo
    lo = np.mod(lo, bins)               # 180 度要绕回 0
    hi = np.mod(lo + 1, bins)

    # 用 bincount 的思路一次算完所有 cell 的直方图，避免 Python 循环
    feats = np.zeros((nc_y, nc_x, bins), dtype=np.float32)
    for b in range(bins):
        # 该 bin 作为"下界"分到的权重，加上作为"上界"分到的
        w_lo = np.where(lo == b, 1.0 - frac, 0.0)
        w_hi = np.where(hi == b, frac, 0.0)
        weighted = (w_lo + w_hi) * mag
        # 按 cell 分块求和
        blk = weighted[:nc_y * cell, :nc_x * cell].reshape(nc_y, cell, nc_x, cell)
        feats[:, :, b] = blk.sum(axis=(1, 3))

    feat = feats.ravel()
    # L2 归一化，压一下整体亮度的差异
    norm = np.linalg.norm(feat)
    if norm > 1e-8:
        feat = feat / norm
    return feat.astype(np.float32)


def color_hist_features(rgb, bins=HIST_BINS):
    """
    颜色直方图。不用 RGB 直接用 HSV 更好，
    因为 HSV 里 H 分量对光照强弱不敏感，食物照片明暗差别很大。
    这里用 PIL 转 HSV，分桶统计 H/S/V 各自的分布。
    """
    hsv = np.asarray(Image.fromarray(rgb.astype(np.uint8)).convert("HSV"),
                     dtype=np.float32)
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch], bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        feats.append(hist)
    return np.concatenate(feats)


def extract_one(path):
    """单张图的完整特征向量"""
    gray, rgb = to_gray_resized(path)
    return np.concatenate([hog_features(gray), color_hist_features(rgb)])


def feature_dim():
    n_cell = (SMALL // CELL) ** 2
    return n_cell * BINS + 3 * HIST_BINS


def extract_split(split, limit=None, verbose=True):
    """抽一个 split 的所有特征，返回 (X, y, paths)"""
    root = SPLIT_DIR / split
    if not root.is_dir():
        print("找不到 %s，先跑 prepare_dataset.py + split_dataset.py" % root)
        return None

    classes = read_classes()
    files, labels = [], []
    for ci, cls in enumerate(classes):
        # 统一用 rglob：预处理后的图都是 .jpg，
        # 但跨数据集测试集里可能有 .jpeg，一并兼容
        fs = sorted(list((root / cls).glob("*.jpg"))
                    + list((root / cls).glob("*.jpeg")))
        if limit:
            fs = fs[:limit]
        files.extend(fs)
        labels.extend([ci] * len(fs))

    X = np.zeros((len(files), feature_dim()), dtype=np.float32)
    t0 = time.time()
    bad = 0

    for i, (f, y) in enumerate(zip(files, labels)):
        try:
            X[i] = extract_one(f)
        except Exception:
            # 抽不出来的先填 0，最后统一删掉，免得整批中断
            bad += 1
        if verbose and (i + 1) % 500 == 0:
            el = time.time() - t0
            print("   %s  %5d/%d   已用 %.1fs   预计还需 %.1fs"
                  % (split, i + 1, len(files), el,
                     el / (i + 1) * (len(files) - i - 1)))

    y = np.asarray(labels, dtype=np.int64)
    paths = np.asarray([str(f) for f in files])

    if bad:
        keep = X.any(axis=1)         # 全 0 的就是失败的
        print("   !! %d 张图抽特征失败，已剔除" % bad)
        X, y, paths = X[keep], y[keep], paths[keep]

    return X, y, paths


def read_classes():
    """
    读取类别顺序。

    以 split_dataset.py 落盘时记录的为准，而不是用 config.CLASSES。
    原因是 train_cnn.py 里用的是 ImageFolder，它按文件夹名字母序编号，
    如果 config 里的手写顺序和字母序不一致，特征里的标签就和 CNN 的
    标签对不上 —— 两边的准确率没法横向比较，而且不会报错，极难发现。
    """
    p = SPLIT_DIR / "classes.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    from config import CLASSES
    return CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="输出 npz 路径")
    ap.add_argument("--limit", type=int, default=None,
                    help="每个类别只用前 N 张（调试用，跑得快）")
    args = ap.parse_args()

    print("特征维度：%d （HOG %d + 颜色 %d）"
          % (feature_dim(),
             (SMALL // CELL) ** 2 * BINS,
             3 * HIST_BINS))

    result = {}
    for split in ("train", "val", "test"):
        print("\n抽取 %s ..." % split)
        out = extract_split(split, limit=args.limit)
        if out is None:
            return
        X, y, paths = out
        result["X_" + split] = X
        result["y_" + split] = y
        result["p_" + split] = paths
        print("   %s: X%s  y%s" % (split, X.shape, y.shape))

    # 把类别顺序一起存进去。train_baseline.py 要用它来对中文名，
    # 不存的话两边类别顺序对不上，准确率就白算了。
    result["classes"] = np.array(read_classes(), dtype=object)

    out_path = Path(args.out) if args.out else (config.DATA_DIR / "features.npz")
    np.savez_compressed(out_path, **result)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print("\n已保存 %s  (%.1f MB)" % (out_path, size_mb))
    print("下一步：python src/train_baseline.py")


if __name__ == "__main__":
    main()
