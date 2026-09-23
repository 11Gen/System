"""
plot_history.py —— 从训练记录重新生成训练曲线。

为什么需要单独的脚本：
    训练曲线的绘图代码是写在 train_cnn.py 里的，但那是训练**结束**时调用的。
    如果模型是在加这段代码之前训好的（本项目就是这种情况），
    就没有曲线图。与其重训一次，不如从 train_result_*.json 里
    把 history 读出来重画 —— 记录里存了每个 epoch 的全部指标，
    信息是完整的。

用法：
    python src/plot_history.py                       # 画 main
    python src/plot_history.py --tag full            # 画指定实验
    python src/plot_history.py --all                 # 把所有实验画在一起对比
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import FIGURE_DIR, MODEL_DIR


def setup_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fam in ["Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "DejaVu Sans"]:
        if fam in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [fam]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def load(tag):
    p = MODEL_DIR / ("train_result_%s.json" % tag)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def plot_one(tag, filename=None):
    plt = setup_font()
    res = load(tag)
    if not res or not res.get("history"):
        print("找不到 %s 的训练记录" % tag)
        return False

    hist = res["history"]
    eps = list(range(1, len(hist) + 1))

    switch = None
    for i, h in enumerate(hist):
        if "阶段二" in h.get("stage", ""):
            switch = i + 1
            break

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    data = [(  # (训练, 验证, 标题, y 轴)
        [h["train_acc"] for h in hist], [h["val_acc"] for h in hist],
        "准确率", "准确率"),
        ([h["train_loss"] for h in hist], [h["val_loss"] for h in hist],
         "损失", "损失"),
    ]
    for ax, (a, b, title, ylab) in zip(axes, data):
        ax.plot(eps, a, "o-", label="训练集", color="#4f81bd", markersize=3.5)
        ax.plot(eps, b, "s-", label="验证集", color="#c0504d", markersize=3.5)
        if switch:
            ax.axvline(switch - 0.5, color="gray", linestyle="--", linewidth=1.2)
            ax.text(switch - 0.4, ax.get_ylim()[1], " 阶段二开始",
                    fontsize=8, color="gray", va="top")
        ax.set_xlabel("epoch（两阶段连续编号）")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    acc = res.get("test_acc")
    if acc:
        axes[0].axhline(acc, color="green", linestyle=":", linewidth=1.4,
                        label="测试集 %.4f" % acc)
        axes[0].legend(fontsize=9)

    plt.tight_layout()
    out = FIGURE_DIR / (filename or "training_curve.png")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("已保存 %s" % out)
    return True


def plot_all(tags):
    """把多组实验的验证集准确率画在一张图上做对比"""
    plt = setup_font()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = ["#4f81bd", "#c0504d", "#9bbb59", "#8064a2", "#f79646"]
    plotted = 0
    for i, tag in enumerate(tags):
        res = load(tag)
        if not res or not res.get("history"):
            continue
        hist = res["history"]
        ax.plot(range(1, len(hist) + 1), [h["val_acc"] for h in hist],
                "o-", markersize=3.5, color=colors[i % len(colors)],
                label="%s（测试 %.4f）" % (res.get("name", tag), res.get("test_acc", 0)))
        plotted += 1
    if not plotted:
        print("没有可画的数据")
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("验证集准确率")
    ax.set_title("消融实验验证集准确率对比")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGURE_DIR / "ablation_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("已保存 %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        plot_all(["full", "noaug", "stage1only", "scratch"])
        return
    plot_one(args.tag)


if __name__ == "__main__":
    main()
