"""
split_dataset.py —— 把已清洗的数据组织成训练用的目录结构。

其实 prepare_dataset.py 已经按 8:1:1 分好 train/val/test 了，这个脚本
负责的是"最后一公里"：

1) 校验：再逐张确认图片能正常解码。
   这一步看起来多余，但吃过亏 —— Food-101 里有文件头正常、真正
   解码时才炸的图，训练跑到一半崩在 DataLoader 里，因为有多个
   worker 进程，报错信息根本看不出是哪张图。所以在训练前统一筛一遍。

2) 校验类别名和 config.CLASSES 是否一致。
   ImageFolder 是按文件夹名的字母序排类别的，和 config 里手写的顺序
   不一定相同，不一致的话标签会整体错位，而且不会报错 ——
   只是准确率莫名其妙很低，特别难查。所以这里直接比出来。

3) 把跨数据集测试集单独放好，用于"泛化能力"实验。

用法：
    python src/split_dataset.py                 # 校验并落位
    python src/split_dataset.py --check-only    # 只统计不复制
"""

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import CLASSES, RAW_DIR, SPLIT_DIR
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLEAN16 = RAW_DIR / "clean16"      # prepare_dataset.py 输出
CLEAN5 = RAW_DIR / "clean5"        # 5 类家常菜，作跨数据集测试


def is_readable(path):
    """能不能真的解码。verify() 只看文件头不够，必须再 load() 一次。"""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
        return True, None
    except Exception as exc:
        return False, str(exc)[:80]


def verify_and_copy(src_root, dst_root, splits, class_names, do_copy=True):
    """校验 + 复制。返回 (统计, 坏图清单)"""
    stat = defaultdict(Counter)
    broken = []

    for split in splits:
        for cls in class_names:
            src_dir = src_root / split / cls
            if not src_dir.is_dir():
                print("   !! 缺少目录 %s" % src_dir)
                continue
            dst_dir = dst_root / split / cls
            if do_copy:
                dst_dir.mkdir(parents=True, exist_ok=True)

            # 两种扩展名都要收。主数据集是 .jpg，
            # 但跨数据集那批原图是 .jpeg —— 一开始只 glob 了 *.jpg，
            # 结果跨数据集统计全空、一张图都没复制过去，
            # 而且因为"没找到文件"不算错误，连警告都不会打，很容易漏掉。
            for p in sorted(list(src_dir.glob("*.jpg"))
                            + list(src_dir.glob("*.jpeg"))):
                ok, err = is_readable(p)
                if not ok:
                    broken.append((str(p), err))
                    stat[split]["坏图"] += 1
                    continue
                if do_copy:
                    shutil.copy2(p, dst_dir / p.name)
                stat[split][cls] += 1
    return stat, broken


def print_stat(stat, title, classes=None):
    """
    打印统计表。

    classes 默认用主数据集的 16 类，但跨数据集测试要传它自己的类别名 ——
    不然会拿 16 个类名去 5 个目录里找，表格全空。
    """
    classes = classes if classes is not None else CLASSES
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    splits = list(stat.keys())
    header = "%-42s" % "类别" + "".join("%10s" % s for s in splits)
    print(header)
    print("-" * 92)
    total = Counter()
    for cls in classes:
        row = "%-42s" % cls
        for s in splits:
            n = stat[s].get(cls, 0)
            total[s] += n
            row += "%10d" % n
        print(row)
    print("-" * 92)
    print("%-42s" % "合计" + "".join("%10d" % total[s] for s in splits))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="只统计不复制")
    ap.add_argument("--with-extra", action="store_true",
                    help="同时准备跨数据集测试集(5类家常菜)")
    ap.add_argument("--force", action="store_true", help="目标目录非空也重建")
    args = ap.parse_args()

    config.ensure_dirs()

    if not CLEAN16.exists():
        print("找不到 %s" % CLEAN16)
        print("先跑：python src/prepare_dataset.py --source dishes16")
        sys.exit(1)

    # ---- 类别一致性检查。这一步很关键，见文件开头说明。----
    actual = sorted([d.name for d in (CLEAN16 / "train").iterdir() if d.is_dir()])
    expected = sorted(CLASSES)
    if actual != expected:
        print("!! 类别不一致，必须修正后再训练")
        print("   目录里 (%d)：%s" % (len(actual), actual))
        print("   配置里 (%d)：%s" % (len(expected), expected))
        print("   只在目录里：%s" % sorted(set(actual) - set(expected)))
        print("   只在配置里：%s" % sorted(set(expected) - set(actual)))
        sys.exit(1)
    print("类别一致性检查通过，共 %d 类" % len(actual))

    # ---- 把 ImageFolder 实际会看到的顺序落盘 ----
    # 这个文件是"唯一真源"：特征抽取、基线训练、评估、界面全都读它。
    # 不落盘而各自去 glob 目录的话，一旦有人改了目录结构，
    # 各处读到的顺序就可能不同，标签整体错位却不会报错。
    if not args.check_only:
        import json
        with open(SPLIT_DIR / "classes.json", "w", encoding="utf-8") as f:
            json.dump(actual, f, ensure_ascii=False, indent=2)
        print("\n类别顺序已写入 %s" % (SPLIT_DIR / "classes.json"))

    print("\nImageFolder 读取顺序（按文件夹名字母序）：")
    for i, c in enumerate(actual):
        print("   [%2d] %-42s %s" % (i, c, config.CN_NAME.get(c, "?")))

    if SPLIT_DIR.exists() and any(SPLIT_DIR.iterdir()) and not args.force \
            and not args.check_only:
        print("\n%s 里已有内容。要重建请加 --force。" % SPLIT_DIR)
        sys.exit(1)

    stat, broken = verify_and_copy(
        CLEAN16, SPLIT_DIR, ["train", "val", "test"], CLASSES,
        do_copy=not args.check_only)

    total = print_stat(stat, "主数据集 16 道中国名菜")

    if broken:
        print("\n发现 %d 张坏图，已剔除：" % len(broken))
        for p, e in broken[:15]:
            print("   %s -> %s" % (p, e))
        log = SPLIT_DIR / "broken_images.txt"
        if not args.check_only:
            with open(log, "w", encoding="utf-8") as f:
                for p, e in broken:
                    f.write("%s\t%s\n" % (p, e))
            print("   清单已写入 %s" % log)
    else:
        print("\n图片完整性检查：全部正常")

    n_classes = len(CLASSES)
    print("\n训练集随机猜测基线：%.2f%% （%d 类均匀分布）"
          % (100.0 / n_classes, n_classes))

    # ---- 跨数据集测试集 ----
    if args.with_extra and CLEAN5.exists():
        print("\n准备跨数据集测试集（5 类家常菜）...")
        extra_dst = SPLIT_DIR / "cross_dataset"
        if extra_dst.exists() and args.force:
            shutil.rmtree(extra_dst)

        # 注意这里要传 clean5 自己的类别名，不能传 16 道名菜的 CLASSES。
        # 一开始传错了，结果统计表全是空白 —— 因为拿 16 个类名去 5 个目录里找，
        # 一个都对不上。而且 print_stat 用的也是 CLASSES，
        # 所以连"合计"都是空的，问题不容易一眼看出来。
        cn5_classes = sorted([d.name for d in (CLEAN5 / "test").iterdir()
                              if d.is_dir()])
        st2, broken2 = verify_and_copy(CLEAN5, extra_dst, ["test"], cn5_classes)
        print_stat(st2, "跨数据集测试集（另一个来源的 5 类家常菜）",
                   classes=cn5_classes)

        print("\n跨数据集测试集结构：%s/<类别名>/" % extra_dst)
        print("用法：python src/evaluate.py --root %s --tag cross" % extra_dst)

        overlap = [c for c in cn5_classes if c in CLASSES]
        novel = [c for c in cn5_classes if c not in CLASSES]
        print("\n其中 %d 类与主数据集重叠（%s）" % (len(overlap), "、".join(overlap)))
        print("     %d 类主数据集没有（%s）" % (len(novel), "、".join(novel)))
        print("     → 重叠的类可以测「换了数据集还认不认得」，")
        print("       没见过的类可以测「会不会硬归类」（开集识别）。")

    print("\n完成。数据目录：%s" % SPLIT_DIR)
    print("下一步：python src/extract_features.py   （传统方法基线）")
    print("        python src/train_cnn.py           （深度学习主线）")


if __name__ == "__main__":
    main()
