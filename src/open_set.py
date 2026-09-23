"""
open_set.py —— 开集识别（拒识）评测。

要解决的问题：
    前面所有评估都建立在一个前提下——测试集里的每一类模型都学过。
    但真实使用中，用户随手拍一张番茄炒蛋喂进来，模型也会给出一个
    「置信度很高」的答案。第 4.6 节已经量化过这个问题：
    未知类别中有 85.9% 的样本最大置信度超过 0.5。

    所以需要给模型加一个「拒识」能力：判断这个输入是否属于
    模型学过的类别，不属于就明确说「不认识」，而不是硬猜一个。

为什么不做端到端的开集训练：
    主流做法（如基于原型网络、生成式回放）需要改动训练流程并重训模型，
    而本项目已经训练好的模型还有完整的实验结果。更重要的是——
    **区分「认识」和「不认识」这件事，未必需要重新训练**：
    模型输出的 logits 本身就携带了「这个输入像不像训练分布」的信息。
    所以本文采用「后处理式拒识」：在不改动模型的前提下，
    用一个评分函数把已知/未知样本分开。这也是工程上更划算的方案。

三种评分方法（都是 2017~2020 年开集识别方向的代表性做法）：
    1. MSP（Maximum Softmax Probability）—— 直接用最大 softmax 概率
       Hendrycks & Gimpel, ICLR 2017。最简单，作为基线。
    2. 熵（Predictive Entropy）—— softmax 分布的归一化熵
       分布越平坦说明模型越犹豫。
    3. 能量（Energy）—— logsumexp(logits)
       Liu et al., NeurIPS 2020。只用 softmax 有个已知问题：
       经过指数归一化后，不同样本的 logits 差距会被压平，
       所以直接用 logits 的能量往往分得更开。

评测指标：
    AUROC    —— 把已知/未知分开的整体排序能力，不受阈值影响
    TNR@95TPR —— 在保证 95% 的已知样本被正常接受的前提下，
                 能拒掉多少未知样本。这是开集识别文献里的常用指标，
                 因为实际部署时不能为了拒识而牺牲太多正常识别。

用法：
    python src/open_set.py
    python src/open_set.py --save-threshold    # 把推荐阈值写进配置供界面使用
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import FIGURE_DIR, MODEL_DIR, SPLIT_DIR
from evaluate import load_eval_loader, load_model_classes
from train_cnn import build_model, pick_device


# ---------------------------------------------------------------- 评分函数
def score_energy(logits, temperature=1.0):
    """
    能量分：T · logsumexp(logits / T)，**不取负**。

    方向说明（这里踩过一次坑，值得记下来）：
        实测已知样本的能量分中位数是 8.29，未知样本是 3.03 ——
        **已知样本的能量更高**。原因是模型对训练分布内的样本会给出
        整体更大的 logits，logsumexp 保留了这个绝对大小。

        所以判据是「能量 **高于** 阈值 → 接受为已知」，
        而不是反过来。一开始写的时候习惯性地取了负号当"异常分"，
        结果方向判反，阈值标定出来的已知接受率只有 5%。

        下面所有评分函数统一返回「**越大越像未知**」的异常分，
        这样 ROC 那套可以直接用，方向不会记错。
    """
    z = np.asarray(logits, dtype=np.float64) / temperature
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return temperature * lse          # 越大越像**已知**，调用处会取负


def score_msp(probs):
    """
    最大 softmax 概率的负值，作为异常分。

    已知样本应该「很确定」（概率高 → 负值小），
    未知样本应该「不确定」（概率低 → 负值大）。
    统一返回「越大越像未知」。
    """
    return -probs.max(axis=1)


def score_entropy(probs):
    """归一化预测熵。0~1，越大越犹豫 —— 天然就是「越大越像未知」。"""
    eps = 1e-12
    h = -np.sum(probs * np.log(probs + eps), axis=1)
    return h / np.log(probs.shape[1])


def energy_anomaly(logits, temperature=1.0):
    """能量分取负，转成「越大越像未知」的异常分（供评测用）"""
    return -score_energy(logits, temperature)


# ---------------------------------------------------------------- 推理
@torch.no_grad()
def collect_outputs(model, loader, device):
    """跑一遍，把 logits 和标签都收下来。后续三种评分共用，避免重复推理。"""
    model.eval()
    logits_all, labels_all = [], []
    for x, y in loader:
        out = model(x.to(device))
        logits_all.append(out.cpu().numpy())
        labels_all.append(y.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def evaluate_scorer(name, scores_known, scores_novel, target_accept=0.95):
    """
    算一个评分函数的开集指标。

    scores_* 都是「越大越像未知」的异常分，标签：已知 = 0，未知 = 1。

    阈值的定法：直接取已知样本异常分的 (1-target_accept) 分位数，
    这样「已知接受率」严格等于 target_accept。

    这里原本是用 roc_curve 的返回值配合 searchsorted 找阈值，结果取错了。
    原因是 roc_curve 返回的 fpr 数组里有大量重复值（不是严格单调），
    searchsorted 撞上重复段会落到哪一端不好预测 ——
    实测取到的是「已知接受率 5%」那一端而不是 95%。
    改成直接按分位数算，语义明确，也不会受数组形状影响。
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.zeros(len(scores_known), dtype=int),
                        np.ones(len(scores_novel), dtype=int)])
    s = np.concatenate([scores_known, scores_novel])

    auroc = roc_auc_score(y, s)

    # 阈值：已知样本异常分的上分位数，超过它就判为未知
    threshold = float(np.percentile(scores_known, target_accept * 100))

    known_accept = float((scores_known <= threshold).mean())
    novel_reject = float((scores_novel > threshold).mean())

    return {
        "name": name,
        "auroc": round(float(auroc), 4),
        # 习惯上把这个数叫 TNR@95TPR，但严格说它是
        # 「固定已知接受率 95% 时的未知拒识率」，比 TNR@95TPR 更直观
        "novel_reject_at_95_accept": round(novel_reject, 4),
        "threshold": round(threshold, 4),
        "known_accept_rate": round(known_accept, 4),
    }


def energy_threshold_sweep(e_known, e_novel, thresholds):
    """
    能量分阈值对照表，**直接用原始能量分**（不取负）。

    判据：能量 >= 阈值 → 接受为已知；能量 < 阈值 → 判为未知。
    所以阈值越高越严格：已知接受率下降、未知拒识率上升。
    这张表是给界面和报告用的，必须用界面实际使用的那个分数域，
    否则读表的人会把方向搞反。
    """
    rows = []
    for t in thresholds:
        rows.append({
            "能量阈值": round(float(t), 2),
            "已知样本接受率": round(float((e_known >= t).mean()), 4),
            "未知样本拒识率": round(float((e_novel < t).mean()), 4),
        })
    return rows


def threshold_sweep(scores_known, scores_novel, thresholds):
    """
    通用异常分阈值对照表（越大越像未知）。

    保留这个函数是因为报告里做方法对比时要统一口径，
    但注意它的分数域是"异常分"，不是界面用的能量分。
    """
    rows = []
    for t in thresholds:
        rows.append({
            "阈值": round(float(t), 3),
            "已知样本接受率": round(float((scores_known <= t).mean()), 4),
            "未知样本拒识率": round(float((scores_novel > t).mean()), 4),
        })
    return rows


def plot_open_set(results, scores_by_name, out_path):
    """画两张图：ROC 曲线 + 各类分数分布"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for fam in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
        if fam in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [fam]
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # 左：ROC
    ax = axes[0]
    for name, (sk, sn) in scores_by_name.items():
        y = np.concatenate([np.zeros(len(sk), dtype=int),
                            np.ones(len(sn), dtype=int)])
        s = np.concatenate([sk, sn])
        fpr, tpr, _ = roc_curve(y, s)
        auroc = roc_auc_score(y, s)
        ax.plot(fpr, tpr, linewidth=1.8, label="%s (AUROC=%.3f)" % (name, auroc))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机猜测")
    ax.axvline(0.05, color="gray", linestyle=":", linewidth=1.2)
    ax.text(0.052, 0.06, "已知样本误拒率 5%", fontsize=8, color="gray")
    ax.set_xlabel("已知样本被误拒的比例（FPR）")
    ax.set_ylabel("未知样本被正确拒识的比例（TPR）")
    ax.set_title("开集识别 ROC 曲线")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # 右：用最优方法的分数分布对比
    ax = axes[1]
    best_name = max(results, key=lambda r: r["auroc"])["name"]
    sk, sn = scores_by_name[best_name]
    bins = np.linspace(min(sk.min(), sn.min()), max(sk.max(), sn.max()), 40)
    ax.hist(sk, bins=bins, alpha=0.7, label="已知类别（%d 张）" % len(sk),
            color="#4f81bd", edgecolor="white", linewidth=0.3)
    ax.hist(sn, bins=bins, alpha=0.75, label="未知类别（%d 张）" % len(sn),
            color="#c0504d", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("异常分 = −logsumexp(logits)（越靠左越像已知）")
    ax.set_ylabel("样本数")
    ax.set_title("能量分分布对比（取负后的异常分）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  图已保存 %s" % out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=1.0, help="能量分的温度参数")
    ap.add_argument("--save-threshold", action="store_true",
                    help="把推荐阈值写入 models/open_set.json，供界面读取")
    args = ap.parse_args()

    device = pick_device()
    model_classes = load_model_classes()

    ckpt = MODEL_DIR / "best.pt"
    if not ckpt.exists():
        print("找不到 %s，先跑 src/train_cnn.py" % ckpt)
        sys.exit(1)

    print("=" * 82)
    print("开集识别（拒识）评测")
    print("=" * 82)
    print("  模型权重 : %s" % ckpt.name)
    print("  已知样本 : test 集（模型学过的 16 类）")
    print("  未知样本 : cross_dataset 中模型没学过的 4 类")
    print("  设备     : %s\n" % device)

    model = build_model(num_classes=len(model_classes),
                        pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # ---- 已知样本 ----
    loader_k, ds_k = load_eval_loader("test", 32)
    if loader_k is None:
        sys.exit(1)
    logits_k, _ = collect_outputs(model, loader_k, device)
    probs_k = torch.softmax(torch.from_numpy(logits_k), dim=1).numpy()
    print("  已知样本 %d 张（%d 类）" % (len(logits_k), len(ds_k.classes)))

    # ---- 未知样本 ----
    cross_root = SPLIT_DIR / "cross_dataset" / "test"
    if not cross_root.is_dir():
        cross_root = SPLIT_DIR / "cross_dataset"
    loader_u, ds_u = load_eval_loader(root=str(cross_root), batch_size=32)
    if loader_u is None:
        print("找不到跨数据集测试集，先跑 split_dataset.py --with-extra")
        sys.exit(1)

    logits_u_all, labels_u = collect_outputs(model, loader_u, device)
    # 只保留模型没学过的类别
    known_idx = {i for i, c in enumerate(ds_u.classes) if c in model_classes}
    novel_mask = np.array([l not in known_idx for l in labels_u])
    logits_u = logits_u_all[novel_mask]
    probs_u = torch.softmax(torch.from_numpy(logits_u), dim=1).numpy()

    novel_names = [ds_u.classes[i] for i in range(len(ds_u.classes))
                   if i not in known_idx]
    print("  未知样本 %d 张（%d 类：%s）\n"
          % (len(logits_u), len(novel_names),
             "、".join(config.CN_NAME.get(c, c) for c in novel_names)))

    # ---- 三种评分（都统一成「越大越像未知」的异常分）----
    scorers = {
        "MSP（最大softmax）": (score_msp(probs_k), score_msp(probs_u)),
        "熵": (score_entropy(probs_k), score_entropy(probs_u)),
        "能量": (energy_anomaly(logits_k, args.temp),
                 energy_anomaly(logits_u, args.temp)),
    }

    print("%-18s %9s %16s %11s %11s"
          % ("方法", "AUROC", "未知拒识率", "已知接受率", "阈值"))
    print("-" * 82)
    results = []
    for name, (sk, sn) in scorers.items():
        r = evaluate_scorer(name, sk, sn)
        results.append(r)
        print("%-18s %9.4f %16.4f %11.4f %11.4f"
              % (name, r["auroc"], r["novel_reject_at_95_accept"],
                 r["known_accept_rate"], r["threshold"]))
    print("-" * 82)
    print("「未知拒识率」是在固定「已知接受率 95%%」的前提下测的 ——")
    print("含义是：只让 5%% 的正常菜品被要求重拍，能挡掉多少陌生输入。")

    # ---- 补充：如果只看最大置信度，已知/未知的分布差多少 ----
    conf_k = probs_k.max(axis=1)
    conf_u = probs_u.max(axis=1)
    print("\n最大置信度对比：")
    print("  已知样本 : 平均 %.3f   中位数 %.3f" % (conf_k.mean(), np.median(conf_k)))
    print("  未知样本 : 平均 %.3f   中位数 %.3f" % (conf_u.mean(), np.median(conf_u)))
    print("  含义：模型对未知样本确实更不确定，但两者的分布有明显重叠，"
          "所以单靠置信度阈值无法完全分开。")

    # ---- 阈值对照表（用界面实际使用的能量分域）----
    best = max(results, key=lambda r: r["auroc"])
    e_k = score_energy(logits_k, args.temp)
    e_u = score_energy(logits_u, args.temp)

    print("\n能量分阈值对照表：")
    lo, hi = np.percentile(e_k, [1, 99])
    sweep = energy_threshold_sweep(e_k, e_u, np.linspace(lo, hi, 9))
    print("  %10s %16s %16s" % ("能量阈值", "已知接受率", "未知拒识率"))
    for row in sweep:
        print("  %10.2f %16.4f %16.4f"
              % (row["能量阈值"], row["已知样本接受率"], row["未知样本拒识率"]))

    print("\n怎么读这张表：")
    print("  能量分越高越像「模型学过的菜」。阈值定得高 → 更保守，")
    print("  更多输入被判为不认识，好处是不给错答案，"
          "坏处是正常菜品会被要求重拍。")
    print("  阈值定得低 → 几乎不拒识，等于退回普通闭集分类器。")
    print("  工程上的折中是取「已知接受率 ≈ 95%」那一档："
          "只让 5% 的正常请求重拍，换取挡掉近一半的陌生输入。")

    plot_open_set(results, scorers, FIGURE_DIR / "open_set_roc.png")

    # 界面要用的阈值是**原始能量分**的下限（高于它才接受）。
    # 评测里用的是取负后的异常分，所以要转回来。
    reject_threshold = -best["threshold"]

    out = {
        "checkpoint": ckpt.name,
        "n_known": int(len(logits_k)),
        "n_novel": int(len(logits_u)),
        "novel_classes": novel_names,
        "methods": results,
        "best_method": best["name"],
        # 说明字段含义，避免以后自己看混：
        # 这是原始能量分 logsumexp(logits) 的下限，
        # 能量 < 该值 → 判为未知类别，提示用户重拍
        "reject_threshold": round(float(reject_threshold), 4),
        "threshold_meaning": "原始能量分 logsumexp(logits) 的下限，"
                             "低于该值判为未知类别",
        "confidence": {
            "known_mean": round(float(conf_k.mean()), 4),
            "novel_mean": round(float(conf_u.mean()), 4),
        },
        "threshold_sweep": sweep,
    }
    p = MODEL_DIR / "open_set.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结果已写入 %s" % p)

    if args.save_threshold:
        cfg = {"reject_threshold": round(float(reject_threshold), 4),
               "method": best["name"],
               "known_accept_rate": best["known_accept_rate"],
               "novel_reject_rate": best["novel_reject_at_95_accept"]}
        cp = MODEL_DIR / "open_set_threshold.json"
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print("推荐阈值 %.4f 已写入 %s（界面会读取它做拒识判断）"
              % (reject_threshold, cp))


if __name__ == "__main__":
    main()
