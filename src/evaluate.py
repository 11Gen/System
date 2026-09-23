"""
evaluate.py —— 模型评估与误差分析。

训练脚本里只看了一个总的准确率，那是不够的。这个脚本负责出报告要用的东西：
  1) 分类报告：每类的 precision / recall / f1
  2) 混淆矩阵：看看到底谁被错认成谁
  3) 错分样本可视化：把置信度高但判错的图挑出来看
  4) 置信度分布：模型到底是"很确定地错"还是"犹豫地错"

第 3 点是我觉得最有意思的一步，也是报告里"误差分析"那一节的素材来源。
光看混淆矩阵只能知道"哪两类容易混"，把具体图片调出来看才知道
"为什么混" —— 比如沙拉类之间混，主要是因为颜色和形状都太像了。
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # 无界面环境必须指定，否则 matplotlib 会去找显示器
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             top_k_accuracy_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import (FIGURE_DIR, IMG_SIZE, MODEL_DIR, NUM_CLASSES, SPLIT_DIR,
                    ensure_dirs)
from train_cnn import build_model, pick_device

# 中文字体。Windows 上一般有微软雅黑，Linux 上退到文泉驿。
# 不设这个的话图里的中文全是方框，报告里没法看。
for _font in ["Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "DejaVu Sans"]:
    if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [_font]
        break
plt.rcParams["axes.unicode_minus"] = False      # 负号显示成方框的老问题


def load_model_classes():
    """
    模型训练时用的类别顺序。

    必须从 split_dataset.py 落盘的 classes.json 读，不能从被评估数据的
    文件夹名推 —— 跨数据集测试时，那批数据的类别名和训练时完全不同
    （5 类家常菜 vs 16 道名菜），按它建模型会导致权重 shape 不匹配。
    这个坑踩过：一开始按数据目录建模型，跨数据集评估直接报
    size mismatch for classifier.1.weight。
    """
    p = SPLIT_DIR / "classes.json"
    if p.exists():
        import json
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    from config import CLASSES
    return list(CLASSES)


def load_eval_loader(split="test", batch_size=32, workers=0, root=None):
    """
    构造评估用的 DataLoader。

    root 直接给一个"每类一个子目录"的目录时，就用它。
    这个参数是为跨数据集测试加的：主测试集在 dataset/test/<类>，
    而跨数据集测试集在 dataset/cross_dataset/<类>，
    层级不一样，写死 split 名字就读不到。
    """
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    d = Path(root) if root else (SPLIT_DIR / split)
    if not d.is_dir():
        print("找不到 %s" % d)
        return None, None
    ds = datasets.ImageFolder(str(d), transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers), ds


def remap_labels(ds, model_classes):
    """
    把数据集自带的标签映射到模型类别空间。

    返回 (每个样本对应模型类别索引的数组, 每类统计信息)
    数据集里模型没学过的类别标记为 -1。
    """
    ds_classes = ds.classes
    mapping = []
    info = []
    for i, c in enumerate(ds_classes):
        if c in model_classes:
            mapping.append(model_classes.index(c))
            info.append((c, model_classes.index(c), True))
        else:
            mapping.append(-1)
            info.append((c, -1, False))
    return np.array(mapping, dtype=np.int64), info, [c for c, _, ok in info if ok]


@torch.no_grad()
def predict(model, loader, device):
    """跑一遍，返回 (真实标签, 预测标签, 每类概率, 文件路径)"""
    model.eval()
    ys, ps, probs, paths = [], [], [], []

    for x, y in loader:
        out = model(x.to(device))
        prob = torch.softmax(out, dim=1).cpu().numpy()
        probs.append(prob)
        ps.append(prob.argmax(1))
        ys.append(y.numpy())

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    probs = np.concatenate(probs)
    # ImageFolder 的 samples 顺序和 DataLoader(shuffle=False) 一致
    paths = [p for p, _ in loader.dataset.samples]
    return ys, ps, probs, paths


def plot_confusion(ys, ps, class_names, out_path, normalize=True,
                   title="混淆矩阵"):
    cm = confusion_matrix(ys, ps, labels=list(range(len(class_names))))
    if normalize:
        # 按行归一化，因为各类样本数可能不相等，
        # 看原始计数的话大类会把小类压得看不见
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_disp = np.nan_to_num(cm_disp)
        fmt = ".2f"
    else:
        cm_disp = cm
        fmt = "d"

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.62), max(6, n * 0.55)))
    sns.heatmap(cm_disp, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=0.4, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7 if n > 12 else 9}, ax=ax)
    ax.set_xlabel("预测类别", fontsize=11)
    ax.set_ylabel("真实类别", fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  混淆矩阵已保存 %s" % out_path)
    return cm


def plot_per_class_f1(ys, ps, class_names, out_path):
    """每类 F1 的横向条形图，比表格直观"""
    rep = classification_report(ys, ps, labels=list(range(len(class_names))),
                                target_names=class_names, output_dict=True,
                                zero_division=0)
    f1s = [rep[c]["f1-score"] for c in class_names]
    order = np.argsort(f1s)

    fig, ax = plt.subplots(figsize=(7.5, max(4, len(class_names) * 0.34)))
    colors = plt.cm.RdYlGn([max(0.0, min(1.0, f)) for f in np.array(f1s)[order]])
    ax.barh([class_names[i] for i in order], np.array(f1s)[order],
            color=colors, edgecolor="gray", linewidth=0.4)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("F1 分数", fontsize=11)
    ax.set_title("各类别 F1 分数（按高低排序）", fontsize=12)
    ax.axvline(np.mean(f1s), color="steelblue", linestyle="--", linewidth=1.2,
               label="平均 %.3f" % np.mean(f1s))
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("   每类 F1 图已保存 %s" % out_path)
    return rep


def plot_top_confusions(cm, class_names, out_path, topn=12):
    """挑出最严重的若干组混淆，单独画一张"""
    pairs = []
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    pairs = pairs[:topn]
    if not pairs:
        print("   没有错分样本，跳过混淆对图")
        return []

    labels = ["%s\n-> %s" % (a, b) for _, a, b in pairs]
    vals = [v for v, _, _ in pairs]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(range(len(vals)), vals, color="#c0504d", edgecolor="gray",
           linewidth=0.4)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("错分次数", fontsize=11)
    ax.set_title("最易混淆的类别对（真实 -> 预测）", fontsize=12)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("   易混淆类别对图已保存 %s" % out_path)
    return pairs


def plot_misclassified(ys, ps, probs, paths, class_names, out_path,
                       n=16, min_conf=0.5):
    """
    把"判错了但模型很自信"的样本挑出来拼图。

    比随机挑错分样本有意义得多：随机挑出来的往往是本来就模糊的图，
    看不出问题。挑高置信度的错分，才能暴露模型真正的系统性缺陷。
    """
    wrong = np.where(ys != ps)[0]
    if len(wrong) == 0:
        print("   没有错分样本")
        return []
    conf = probs[wrong, ps[wrong]]
    # 先按置信度降序排，不够就放宽 min_conf 凑够 n 张
    order = np.argsort(-conf)
    pick = [wrong[i] for i in order][:n]

    from PIL import Image
    cols = 4
    rows = (len(pick) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.75))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, pick):
        try:
            img = Image.open(paths[idx]).convert("RGB")
        except Exception:
            ax.axis("off")
            continue
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("真实:%s\n预测:%s  %.0f%%"
                     % (class_names[ys[idx]], class_names[ps[idx]],
                        conf[np.where(wrong == idx)[0][0]] * 100),
                     fontsize=7.5, color="#a03030")

    for ax in axes[len(pick):]:
        ax.axis("off")

    fig.suptitle("高置信度错分样本（模型很确定但判错了）", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("   错分样本拼图已保存 %s" % out_path)
    return pick


def plot_confidence(ys, ps, probs, out_path):
    """正确/错误预测的置信度分布对比"""
    conf = probs.max(axis=1)
    correct = (ys == ps)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.linspace(0, 1, 26)
    ax.hist(conf[correct], bins=bins, alpha=0.72, label="预测正确 (%d)" % correct.sum(),
            color="#4f81bd", edgecolor="white", linewidth=0.4)
    if (~correct).sum() > 0:
        ax.hist(conf[~correct], bins=bins, alpha=0.78,
                label="预测错误 (%d)" % (~correct).sum(),
                color="#c0504d", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("最大预测概率（置信度）", fontsize=11)
    ax.set_ylabel("样本数", fontsize=11)
    ax.set_title("预测置信度分布", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("   置信度分布图已保存 %s" % out_path)

    # 一个实用的结论：如果错误的预测置信度普遍很低，
    # 那就可以设一个阈值，低于阈值就提示"不确定"，交给用户确认。
    stats = {
        "correct_mean_conf": float(conf[correct].mean()) if correct.any() else 0.0,
        "wrong_mean_conf": float(conf[~correct].mean()) if (~correct).any() else 0.0,
        "wrong_above_0.7": int((conf[~correct] > 0.7).sum()) if (~correct).any() else 0,
        "wrong_total": int((~correct).sum()),
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="权重路径，默认 models/best.pt")
    ap.add_argument("--split", default="test",
                    choices=["test", "official_test", "val"])
    ap.add_argument("--root", default=None,
                    help="直接指定数据根目录（每类一个子目录），"
                         "用于跨数据集测试")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-plots", action="store_true", help="只出数字不出图")
    args = ap.parse_args()

    ensure_dirs()
    device = pick_device(args.cpu)

    ckpt = Path(args.ckpt) if args.ckpt else (MODEL_DIR / "best.pt")
    if not ckpt.exists():
        ckpt = MODEL_DIR / "stage1_best.pt"
    if not ckpt.exists():
        print("找不到权重文件，先跑 train_cnn.py")
        sys.exit(1)

    split_label = args.root if args.root else args.split
    # 模型类别以训练时的 classes.json 为准，而不是被评估数据的目录名。
    # 见 load_model_classes() 的说明。
    model_classes = load_model_classes()
    print("=" * 72)
    print(" 评估   权重：%s   数据：%s   设备：%s"
          % (ckpt.name, split_label, device))
    print("  模型类别数：%d" % len(model_classes))
    print("=" * 72)

    loader, ds = load_eval_loader(args.split, args.batch_size, root=args.root)
    if loader is None:
        sys.exit(1)
    ds_classes = ds.classes
    print("  样本数 %d   数据目录里有 %d 类" % (len(ds), len(ds_classes)))

    model = build_model(num_classes=len(model_classes),
                        pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print("  权重加载完成，开始推理 ...")

    ys_ds, ps_model, probs, paths = predict(model, loader, device)

    # 把数据集标签映射到模型类别空间
    label_map, cls_info, known = remap_labels(ds, model_classes)
    ys = label_map[ys_ds]

    known_mask = ys >= 0
    novel_mask = ~known_mask
    print("\n  类别对照：")
    for c, idx, ok in cls_info:
        cn = config.CN_NAME.get(c, c)
        print("     %-42s %s" % (c, ("模型类别 #%d（%s）" % (idx, cn)) if ok
                                else "!! 模型未学过这一类"))

    if novel_mask.any():
        print("\n  数据目录里有 %d 类模型没学过，共 %d 张图。"
              % (sum(1 for _, _, ok in cls_info if not ok), int(novel_mask.sum())))
        print("  这几类无法计算准确率（模型本来就不认识），"
              "改为统计模型的置信度表现，见下面的开集分析。")

    if not known_mask.any():
        print("\n  没有任何一类能与模型类别对上，无法计算准确率。")
        sys.exit(1)

    # 只用模型学过的那些类算准确率
    ys_k, ps_k, probs_k, paths_k = (ys[known_mask], ps_model[known_mask],
                                    probs[known_mask],
                                    [p for p, m in zip(paths, known_mask) if m])
    acc = accuracy_score(ys_k, ps_k)
    top3 = top_k_accuracy_score(ys_k, probs_k, k=min(3, len(model_classes)),
                                labels=list(range(len(model_classes))))

    # 宏 F1 只在"本次评估里真的出现过的类别"上平均。
    # 跨数据集场景下这点很关键：模型有 16 类，但跨数据集只覆盖其中 1 类，
    # 如果在全部 16 类上做宏平均，另外 15 类的 F1 都是 0，
    # 宏 F1 会变成 0.06 —— 一个完全误导人的数字，
    # 而实际那 1 类的 F1 是 0.9855。
    present = sorted(set(ys_k.tolist()) | set(ps_k.tolist()))
    macro = f1_score(ys_k, ps_k, labels=present, average="macro",
                     zero_division=0)

    print("\n  可比样本数：%d" % len(ys_k))
    print("  总体准确率 (top-1) : %.4f" % acc)
    print("  总体准确率 (top-3) : %.4f" % top3)

    print("\n分类报告：")
    rep_txt = classification_report(ys_k, ps_k,
                                    labels=list(range(len(model_classes))),
                                    target_names=model_classes, digits=4,
                                    zero_division=0)
    print(rep_txt)

    tag = ("_" + args.tag) if args.tag else ""
    summary = {
        "checkpoint": ckpt.name,
        # 记实际用的数据源，而不是 --split 的默认值。
        # 跨数据集测试时 --root 才是有意义的那个，
        # 如果这里写 args.split，summary 里会显示成 "test"，
        # 回头看不出这份结果到底是哪个数据集跑的。
        "split": split_label,
        "data_root": str(Path(args.root)) if args.root else str(SPLIT_DIR / args.split),
        "n_samples": int(len(ys_k)),
        "n_total_samples": int(len(ys_ds)),
        "n_classes": len(model_classes),
        "classes": model_classes,
        "accuracy": round(float(acc), 4),
        "top3_accuracy": round(float(top3), 4),
        # 宏 F1 只在出现过的类别上算，理由见上面 present 那段注释
        "macro_f1": round(float(macro), 4),
        "macro_f1_over_n_classes": len(present),
        # 跨数据集时，把模型没学过的类别的置信度单独记下来，
        # 供开集识别分析用
        "novel_class_stats": None,
    }

    # 模型没学过的那些类：算不了准确率，但可以看置信度
    if novel_mask.any():
        novel_probs = probs[novel_mask]
        novel_conf = novel_probs.max(axis=1)
        known_conf = probs_k.max(axis=1) if len(probs_k) else np.array([0.0])
        summary["novel_class_stats"] = {
            "n_samples": int(novel_mask.sum()),
            "classes": [c for c, _, ok in cls_info if not ok],
            "novel_mean_conf": round(float(novel_conf.mean()), 4),
            "known_mean_conf": round(float(known_conf.mean()), 4),
            "novel_above_0.5": int((novel_conf > 0.5).sum()),
            "known_above_0.5": int((known_conf > 0.5).sum()),
        }
        print("\n开集表现（模型没学过的那些类）：")
        print("   未知类别平均置信度 %.3f，已知类别平均置信度 %.3f"
              % (novel_conf.mean(), known_conf.mean()))
        print("   未知类别中置信度 > 0.5 的有 %d / %d 张（%.1f%%）"
              % ((novel_conf > 0.5).sum(), len(novel_conf),
                 100.0 * (novel_conf > 0.5).mean()))
        print("   → 这个比例越低，说明模型越能意识到自己不认识，")
        print("     但闭集分类器没有拒识机制，仍会强行给一个答案。")

    if not args.no_plots:
        print("\n生成图表 ...")
        cm = plot_confusion(ys_k, ps_k, model_classes,
                            FIGURE_DIR / ("confusion_matrix%s.png" % tag))
        rep = plot_per_class_f1(ys_k, ps_k, model_classes,
                                FIGURE_DIR / ("per_class_f1%s.png" % tag))
        pairs = plot_top_confusions(cm, model_classes,
                                    FIGURE_DIR / ("top_confusions%s.png" % tag))
        picks = plot_misclassified(ys_k, ps_k, probs_k, paths_k, model_classes,
                                   FIGURE_DIR / ("misclassified%s.png" % tag))
        cstats = plot_confidence(ys_k, ps_k, probs_k,
                                 FIGURE_DIR / ("confidence%s.png" % tag))

        summary["confidence_stats"] = cstats
        summary["top_confusion_pairs"] = [
            {"true": a, "pred": b, "count": int(c)} for c, a, b in pairs[:10]]
        summary["per_class_f1"] = {
            c: round(float(rep[c]["f1-score"]), 4) for c in model_classes}

        # 顺手把最有意思的几条结论打印出来，写报告时直接用
        print("\n" + "-" * 72)
        print(" 可以直接写进报告的几点观察：")
        print("-" * 72)
        worst = sorted(summary["per_class_f1"].items(), key=lambda kv: kv[1])[:3]
        print(" 1) F1 最低的三类：%s"
              % "、".join("%s(%.3f)" % (k, v) for k, v in worst))
        if pairs:
            print(" 2) 最严重的混淆：%s 被判成 %s，共 %d 次"
                  % (pairs[0][1], pairs[0][2], pairs[0][0]))
        print(" 3) 错误预测的平均置信度 %.3f，正确预测 %.3f"
              % (cstats["wrong_mean_conf"], cstats["correct_mean_conf"]))
        print("    其中置信度 >0.7 的错误有 %d 个，占总错误 %.1f%%"
              % (cstats["wrong_above_0.7"],
                 100.0 * cstats["wrong_above_0.7"] / max(cstats["wrong_total"], 1)))

    out = MODEL_DIR / ("eval_summary%s.json" % tag)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(MODEL_DIR / ("classification_report%s.txt" % tag), "w",
              encoding="utf-8") as f:
        f.write("checkpoint: %s\ndata: %s\n\n" % (ckpt.name, split_label))
        f.write("accuracy: %.4f\ntop3_accuracy: %.4f\n\n" % (acc, top3))
        f.write(rep_txt)
    print("\n评估结果已写入 %s" % out)


if __name__ == "__main__":
    main()
