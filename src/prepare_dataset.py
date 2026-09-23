"""
prepare_dataset.py —— 把原始数据整理成训练用的 train/val/test 目录结构。

本项目实际用了两个数据集，这个脚本负责把它们统一成同一种格式：

  主数据集  data/raw/16-famous-chinese-dishes
            16 道中国名菜，YOLO 目标检测格式（images/*.jpg + labels/*.txt）
            图片文件名带类别前缀，标注文件给出菜在画面里的位置

  辅助数据集 data/raw/common-chinese-food
            5 道家常菜，已经是"一个目录一个类"的分类格式

这里做了两件正则数据集里没做的事：

1) 按标注框裁剪。
   主数据集原本是给目标检测用的，很多照片里除了主菜还有别的菜
   （比如一桌菜的照片）。直接当分类数据用的话，一张图里可能有好几道菜，
   标签到底是哪个就对不上了。所以这里读标注文件，只保留"画面里
   基本只有这一道菜"的图，并按框裁掉多余边缘。
   这个清洗步骤在报告里算数据预处理的一部分。

2) 去掉单色图。
   实际看了数据才发现，里面混了一批几乎纯色/纯黑的图（应该是下载失败
   留下的占位图）。这种图当训练样本纯属噪声，而且数量不少，
   不处理会明显拉低准确率。

用法：
    python src/prepare_dataset.py --source dishes16        # 主数据集
    python src/prepare_dataset.py --source cnfood5         # 辅助数据集
    python src/prepare_dataset.py --check                  # 查看整理结果
"""

import argparse
import hashlib
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import RAW_DIR, RANDOM_SEED, SPLIT_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------- 数据集定义
D16_ROOT = RAW_DIR / "16-famous-chinese-dishes"

# 16 道名菜的中文名 / 菜系 / 简要说明。报告和界面上都要用。
D16_INFO = {
    "Peking_Duck": ("北京烤鸭", "京菜", "果木炭火烤制，皮脆肉嫩"),
    "Sweet_and_Sour_Pork": ("咕咾肉", "粤菜", "酸甜口的炸猪肉块"),
    "Mapo_Tofu": ("麻婆豆腐", "川菜", "麻辣鲜香的烧豆腐"),
    "Yuxiang_Shredded_Pork": ("鱼香肉丝", "川菜", "酸甜微辣的肉丝"),
    "Husband_and_Wife_Lung_Slices": ("夫妻肺片", "川菜", "凉拌牛杂，红油调味"),
    "Twice_Cooked_Pork": ("回锅肉", "川菜", "五花肉先煮后炒，配蒜苗"),
    "Kung_Pao_Chicken": ("宫保鸡丁", "川菜", "鸡丁配花生米，糊辣荔枝味"),
    "Saliva_Chicken": ("口水鸡", "川菜", "凉拌鸡块，红油麻辣"),
    "Soup_Dumplings": ("小笼汤包", "苏菜", "薄皮含汤汁的小笼包"),
    "Braised_Pork_Meatballs_in_Brown_Sauce": ("红烧狮子头", "苏菜", "大肉丸红烧"),
    "Dongpo_Pork": ("东坡肉", "浙菜", "五花肉慢炖，肥而不腻"),
    "West_Lake_Vinegar_Fish": ("西湖醋鱼", "浙菜", "草鱼浇糖醋汁"),
    "Buddha_Jumps_Over_the_Wall": ("佛跳墙", "闽菜", "多种海味与肉类同炖"),
    "Steamed_Sea_Bass": ("清蒸鲈鱼", "粤菜", "清蒸保留原味"),
    "Fish_with_Pickled_Cabbage_and_Chili": ("酸菜鱼", "川菜", "鱼片配酸菜"),
    "Boiled_Fish_with_Sichuan_Peppercorns": ("水煮鱼", "川菜", "鱼片在麻辣红油中"),
}

CN5_ROOT = RAW_DIR / "common-chinese-food"
CN5_INFO = {
    "Braised_Pork": ("红烧肉", "家常菜", "五花肉红烧"),
    "Fish-Flavored_Shredded_Pork": ("鱼香肉丝", "川菜", "与名菜数据集重叠，用作跨集测试"),
    "Kung_Pao_Chicken": ("宫保鸡丁", "川菜", "与名菜数据集重叠，用作跨集测试"),
    "Stir-fried_Shredded_Potatoes": ("炒土豆丝", "家常菜", "醋溜土豆丝"),
    "Tomato_Scrambled_Eggs": ("番茄炒蛋", "家常菜", "国民家常菜"),
}


def is_blank_image(path, std_threshold=6.0, dark_threshold=12.0):
    """
    判断是不是近纯色/纯黑的占位图。

    做法：转灰度后算标准差。纯色图的标准差接近 0，正常照片里
    有纹理和明暗变化，标准差一般都在 30 以上。
    阈值 6.0 是试出来的：设太高会误杀"背景干净的白盘子特写"
    （这类图标准差能到 10 左右），设太低又筛不掉纯色图。
    """
    try:
        with Image.open(path) as im:
            g = im.convert("L")
            # 缩到小尺寸再算，快很多，对判断结果没影响
            g = g.resize((64, 64))
            arr = np.asarray(g, dtype=np.float32)
    except Exception:
        return True, "打不开"

    std = float(arr.std())
    mean = float(arr.mean())
    if std < std_threshold:
        return True, "近纯色(std=%.1f,mean=%.1f)" % (std, mean)
    if mean < dark_threshold:
        return True, "近全黑(mean=%.1f)" % mean
    return False, ""


def crop_by_label(img_path, label_path, pad_ratio=0.04, max_boxes=1):
    """
    读 YOLO 标注，判断这张图能不能当分类样本，并按框裁剪。

    返回 (是否可用, 裁剪后的图 或 None, 说明)

    判断规则：
      - 标注里必须恰好有 1 个框（多个框说明画面里有多道菜，标签有歧义）
      - 框的面积占比要在合理范围：太小说明菜只是画面一角，
        裁出来分辨率不够；太大沿用原图即可
    """
    try:
        lines = [l.split() for l in
                 label_path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if l.strip()]
        boxes = [(int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]))
                 for p in lines if len(p) >= 5]
    except Exception:
        return False, None, "标注读不了"

    if len(boxes) != 1:
        return False, None, "有%d个框" % len(boxes)

    _, cx, cy, bw, bh = boxes[0]
    if bw <= 0 or bh <= 0:
        return False, None, "框异常"

    area = bw * bh
    if area < 0.12:
        # 菜只占画面很小一块，裁出来太糊
        return False, None, "框太小(%.2f)" % area

    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            # 归一化坐标 -> 像素坐标，再按 pad_ratio 往外扩一点
            pad_w, pad_h = pad_ratio, pad_ratio
            x1 = max(0.0, (cx - bw / 2 - pad_w)) * W
            y1 = max(0.0, (cy - bh / 2 - pad_h)) * H
            x2 = min(1.0, (cx + bw / 2 + pad_w)) * W
            y2 = min(1.0, (cy + bh / 2 + pad_h)) * H
            if x2 - x1 < 64 or y2 - y1 < 64:
                return False, None, "裁剪后太小"
            # 裁到只留白边的情况就没必要裁了，直接用原图
            if (x2 - x1) > 0.97 * W and (y2 - y1) > 0.97 * H:
                return True, im.copy(), "整图"
            return True, im.crop((int(x1), int(y1), int(x2), int(y2))), "已裁剪"
    except Exception as e:
        return False, None, "打不开:%s" % str(e)[:30]


def file_md5(path, chunk=1 << 20):
    """算文件指纹，用来去重。同一张图在数据集里出现两次会污染评估。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_dishes16(out_root, val_ratio=0.1, test_ratio=0.1, seed=RANDOM_SEED):
    """整理 16 道名菜数据集"""
    if not D16_ROOT.exists():
        print("找不到 %s，先跑 fetch_dataset.py" % D16_ROOT)
        return None

    img_dir = D16_ROOT / "images"
    lab_dir = D16_ROOT / "labels"
    imgs = sorted(img_dir.glob("*.jpg"))
    print("原始图片 %d 张" % len(imgs))

    # 按文件名前缀归类
    def cls_of(stem):
        m = re.match(r"^(.*?)_new_\d+$", stem)
        if m:
            return m.group(1)
        m = re.match(r"^(.*?)_\d+$", stem)
        return m.group(1) if m else stem

    by_cls = defaultdict(list)
    for p in imgs:
        by_cls[cls_of(p.stem)].append(p)

    rng = random.Random(seed)
    stats = {}
    seen_md5 = {}

    for cls in sorted(by_cls):
        files = by_cls[cls]
        kept, reasons = [], Counter()

        for p in files:
            lab = lab_dir / (p.stem + ".txt")
            if not lab.exists():
                reasons["无标注"] += 1
                continue

            ok, cropped, why = crop_by_label(p, lab)
            if not ok:
                reasons[why] += 1
                continue

            # 单色图检查放在裁剪之后：原图可能有黑边，
            # 裁掉之后才发现主体就是纯色
            blank, why_blank = _check_array(cropped)
            if blank:
                reasons[why_blank] += 1
                continue

            kept.append((p, cropped))

        # 去重
        uniq = []
        for p, im in kept:
            h = file_md5(p)
            if h in seen_md5:
                reasons["重复"] += 1
                continue
            seen_md5[h] = p
            uniq.append((p, im))

        rng.shuffle(uniq)
        n = len(uniq)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)

        splits = {
            "val": uniq[:n_val],
            "test": uniq[n_val:n_val + n_test],
            "train": uniq[n_val + n_test:],
        }

        for split, items in splits.items():
            d = out_root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i, (src, im) in enumerate(items):
                im.save(d / ("%s_%04d.jpg" % (cls, i)), quality=92)

        stats[cls] = {
            "原始": len(files), "保留": n,
            "train": len(splits["train"]), "val": len(splits["val"]),
            "test": len(splits["test"]),
            "剔除": dict(reasons),
        }

    return stats


def _check_array(im):
    """对已经读进内存的 PIL 图做纯色检查，避免重复读盘"""
    g = im.convert("L").resize((64, 64))
    arr = np.asarray(g, dtype=np.float32)
    std, mean = float(arr.std()), float(arr.mean())
    if std < 6.0:
        return True, "近纯色(std=%.1f)" % std
    if mean < 12.0:
        return True, "近全黑"
    return False, ""


def build_cnfood5(out_root, val_ratio=0.1, test_ratio=0.1, seed=RANDOM_SEED):
    """整理辅助的 5 类家常菜数据集（作为跨数据集测试集）"""
    if not CN5_ROOT.exists():
        print("找不到 %s" % CN5_ROOT)
        return None

    rng = random.Random(seed)
    stats = {}
    for cls_dir in sorted(CN5_ROOT.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name.replace("-", "_")
        # 实际结构是 类别/类别/*.jpeg 两层同名目录
        inner = cls_dir / cls_dir.name
        src_dir = inner if inner.is_dir() else cls_dir
        files = sorted(list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.jpg")))

        kept = []
        reasons = Counter()
        for p in files:
            bad, why = is_blank_image(p)
            if bad:
                reasons[why] += 1
                continue
            kept.append(p)

        rng.shuffle(kept)
        n = len(kept)
        n_test = int(n * 0.15)
        # 辅助集只当测试集用，不参与训练（避免和主数据集类别定义冲突）
        splits = {"test": kept[:n_test], "extra": kept[n_test:]}

        for split, items in splits.items():
            d = out_root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(items):
                shutil.copy2(src, d / ("%s_%04d%s" % (cls, i, src.suffix)))

        stats[cls] = {"原始": len(files), "test": len(splits["test"]),
                      "extra": len(splits["extra"]), "剔除": dict(reasons)}
    return stats


def print_stats(stats, title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    print("%-42s %7s %7s %7s %7s %7s" %
          ("类别", "原始", "保留", "train", "val", "test"))
    print("-" * 88)
    tot = Counter()
    for cls, s in sorted(stats.items()):
        print("%-42s %7d %7d %7d %7d %7d"
              % (cls, s["原始"], s.get("保留", s["原始"]),
                 s.get("train", 0), s.get("val", 0), s.get("test", 0)))
        for k, v in s.items():
            if isinstance(v, int):
                tot[k] += v
    print("-" * 88)
    print("%-42s %7d %7d %7d %7d %7d"
          % ("合计", tot["原始"], tot.get("保留", 0),
             tot.get("train", 0), tot.get("val", 0), tot.get("test", 0)))

    print("\n被剔除的样本及原因：")
    allr = Counter()
    for s in stats.values():
        allr.update(s.get("剔除", {}))
    for why, c in allr.most_common():
        print("   %-34s %5d" % (why, c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dishes16", "cnfood5"], default="dishes16")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()

    if args.check:
        for sub in sorted(SPLIT_DIR.iterdir()) if SPLIT_DIR.exists() else []:
            if not sub.is_dir():
                continue
            n = sum(1 for _ in sub.rglob("*.jpg"))
            print("%-16s %6d 张   %d 类"
                  % (sub.name, n, len([d for d in sub.iterdir() if d.is_dir()])))
        return

    if args.source == "dishes16":
        out = Path(args.out) if args.out else (RAW_DIR / "clean16")
        if out.exists():
            shutil.rmtree(out)
        stats = build_dishes16(out)
        if stats:
            print_stats(stats, "16 道名菜数据集整理结果")
            print("\n输出目录：%s" % out)
            print("下一步：python src/split_dataset.py --src %s" % out)
    else:
        out = Path(args.out) if args.out else (RAW_DIR / "clean5")
        if out.exists():
            shutil.rmtree(out)
        stats = build_cnfood5(out)
        if stats:
            print_stats(stats, "5 类家常菜（辅助测试集）整理结果")
            print("\n输出目录：%s" % out)


if __name__ == "__main__":
    main()
