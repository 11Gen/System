"""
train_baseline.py —— 传统机器学习基线（KNN / SVM）。

为什么要做这一步：
    如果报告里只有"我用了 MobileNetV2 迁移学习，准确率 8x%"，读者没法判断
    这个 8x% 是模型厉害还是任务简单。所以需要一个不依赖深度学习的对照组：
    人工设计特征 + 经典分类器。两者一比，"迁移学习到底带来了多少提升"
    才有定量结论。

特征用的是 extract_features.py 里手写的 HOG + HSV 颜色直方图（588 维）。

几个工程上的选择：
  1) KNN 直接用 588 维原始特征会很慢（要算 4287×527 次距离），
     而且高维下距离度量会退化。所以先标准化再 PCA 降到 100 维。
     PCA 也顺带做了去噪。
  2) KNN 的 k 不能拍脑袋定，这里网格搜一下。
  3) SVM 用线性核先跑。RBF 核在这个规模上训练明显更慢，
     而 588 维特征下两者的差距通常不大 —— 实测结果里会给出对比。

用法：
    python src/train_baseline.py
    python src/train_baseline.py --quick     # 每个类只用 100 张，快速验证流程
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import CLASSES, CN_NAME, MODEL_DIR, SPLIT_DIR

FEATURES = config.DATA_DIR / "features.npz"


def load_features():
    if not FEATURES.exists():
        print("找不到 %s" % FEATURES)
        print("先跑：python src/extract_features.py")
        return None
    d = np.load(FEATURES, allow_pickle=True)
    return {k: d[k] for k in d.files}


def evaluate(model, X, y, name, class_names):
    t0 = time.time()
    pred = model.predict(X)
    dt = time.time() - t0
    acc = accuracy_score(y, pred)
    # 宏平均 F1：因为各类样本数不完全相等（188~310），
    # 只看总体准确率会被大类带偏，宏 F1 更能反映"每类都认得好不好"
    mf1 = f1_score(y, pred, average="macro", zero_division=0)
    print("   %-22s 准确率 %.4f   宏F1 %.4f   预测耗时 %.1fs"
          % (name, acc, mf1, dt))
    return {"name": name, "accuracy": round(float(acc), 4),
            "macro_f1": round(float(mf1), 4)}, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="每类只取 100 张，跑得快")
    ap.add_argument("--pca-dim", type=int, default=100)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    config.ensure_dirs()
    data = load_features()
    if data is None:
        sys.exit(1)

    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"], data["y_val"]
    Xte, yte = data["X_test"], data["y_test"]

    # 用实际的类别顺序，避免和 config 不一致
    classes = [str(c) for c in data["classes"]] if "classes" in data else CLASSES
    class_names = [CN_NAME.get(c, c) for c in classes]

    print("=" * 80)
    print("传统方法基线")
    print("=" * 80)
    print("  特征维度 : %d" % Xtr.shape[1])
    print("  训练集   : %s" % (Xtr.shape,))
    print("  验证集   : %s" % (Xva.shape,))
    print("  测试集   : %s" % (Xte.shape,))

    if args.quick:
        # 分层抽样，每类取前 100 个
        idx = []
        for c in np.unique(ytr):
            idx.extend(np.where(ytr == c)[0][:100])
        idx = np.array(idx)
        Xtr, ytr = Xtr[idx], ytr[idx]
        print("  [--quick] 训练集缩小到 %s" % (Xtr.shape,))

    # ---------------- 预处理 ----------------
    print("\n预处理 ...")
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    n_comp = min(args.pca_dim, Xtr_s.shape[0], Xtr_s.shape[1])
    t0 = time.time()
    pca = PCA(n_components=n_comp, random_state=config.RANDOM_SEED).fit(Xtr_s)
    Xtr_p = pca.transform(Xtr_s)
    Xva_p = pca.transform(Xva_s)
    Xte_p = pca.transform(Xte_s)
    print("  PCA: 588 -> %d 维，用时 %.1fs，累计解释方差 %.3f"
          % (n_comp, time.time() - t0, pca.explained_variance_ratio_.sum()))

    results = []

    # ---------------- KNN：先在验证集上选 k ----------------
    print("\n[1/3] KNN —— 在验证集上选 k")
    best_k, best_acc, best_model = None, -1, None
    knn_scan = []
    for k in (3, 5, 9, 15, 25, 40):
        m = KNeighborsClassifier(n_neighbors=k, metric="euclidean",
                                 n_jobs=-1)
        m.fit(Xtr_p, ytr)
        a = accuracy_score(yva, m.predict(Xva_p))
        knn_scan.append({"k": k, "val_acc": round(float(a), 4)})
        print("   k=%-3d 验证集准确率 %.4f" % (k, a))
        if a > best_acc:
            best_k, best_acc, best_model = k, a, m
    print("   -> 选 k=%d（验证集 %.4f）" % (best_k, best_acc))

    r, pred_knn = evaluate(best_model, Xte_p, yte, "KNN (k=%d)" % best_k,
                           class_names)
    r["k"] = best_k
    r["k_scan"] = knn_scan
    results.append(r)

    # ---------------- 线性 SVM ----------------
    print("\n[2/3] SVM")
    svm_res = {}
    for C in (0.1, 1.0, 10.0):
        m = LinearSVC(C=C, max_iter=5000, random_state=config.RANDOM_SEED)
        m.fit(Xtr_p, ytr)
        a = accuracy_score(yva, m.predict(Xva_p))
        svm_res[C] = (a, m)
        print("   LinearSVC C=%-5g 验证集准确率 %.4f" % (C, a))
    best_C = max(svm_res, key=lambda c: svm_res[c][0])
    best_svm = svm_res[best_C][1]
    print("   -> 选 C=%g" % best_C)

    r, pred_svm = evaluate(best_svm, Xte_p, yte, "线性SVM (C=%g)" % best_C,
                           class_names)
    r["C"] = best_C
    results.append(r)

    # ---------------- RBF SVM（对比核函数的影响）----------------
    print("\n[3/3] RBF 核 SVM（对比用）")
    t0 = time.time()
    m_rbf = SVC(kernel="rbf", C=10.0, gamma="scale",
                random_state=config.RANDOM_SEED)
    m_rbf.fit(Xtr_p, ytr)
    print("   训练耗时 %.1fs" % (time.time() - t0))
    r, pred_rbf = evaluate(m_rbf, Xte_p, yte, "RBF-SVM (C=10)", class_names)
    results.append(r)

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 80)
    print("基线结果汇总（测试集）")
    print("=" * 80)
    print("%-24s %10s %10s" % ("方法", "准确率", "宏F1"))
    print("-" * 80)
    for r in results:
        print("%-24s %10.4f %10.4f" % (r["name"], r["accuracy"], r["macro_f1"]))

    best = max(results, key=lambda r: r["accuracy"])
    print("\n最佳传统方法：%s  准确率 %.4f" % (best["name"], best["accuracy"]))
    print("（这个数字就是后面深度学习要超越的基线）")

    # 保存结果
    tag = ("_" + args.tag) if args.tag else ""
    out = {
        "feature_dim": int(Xtr.shape[1]),
        "pca_dim": n_comp,
        "pca_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
        "n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0]),
        "n_test": int(Xte.shape[0]),
        "classes": classes,
        "results": results,
        "best": best["name"],
    }
    with open(MODEL_DIR / ("baseline_result%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 出图（混淆矩阵 + 方法对比）
    try:
        make_plots(best_svm, Xte_p, yte, class_names, results, tag)
    except Exception as e:
        print("出图失败（不影响结果）：%s" % e)

    # 分类报告，写进文件供报告引用
    with open(MODEL_DIR / ("baseline_report%s.txt" % tag), "w",
              encoding="utf-8") as f:
        f.write("传统方法基线（HOG + HSV 颜色直方图，%d 维，PCA 降维到 %d）\n\n"
                % (Xtr.shape[1], n_comp))
        for r in results:
            f.write("%s: accuracy=%.4f macro_f1=%.4f\n"
                    % (r["name"], r["accuracy"], r["macro_f1"]))
        f.write("\n最佳线性SVM 分类报告：\n")
        f.write(classification_report(yte, pred_svm, target_names=class_names,
                                      digits=4, zero_division=0))
    print("\n结果已写入 %s" % (MODEL_DIR / ("baseline_result%s.json" % tag)))


def make_plots(model, Xte, yte, class_names, results, tag):
    """画混淆矩阵和方法对比图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    for fam in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
        if fam in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [fam]
            break
    plt.rcParams["axes.unicode_minus"] = False

    pred = model.predict(Xte)
    cm = confusion_matrix(yte, pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(7, n * 0.55)))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=0.4, annot_kws={"size": 6}, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("传统方法（线性SVM）混淆矩阵")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    fig.savefig(config.FIGURE_DIR / ("baseline_confusion%s.png" % tag),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]
    bars = ax.bar(range(len(names)), accs, color="#4f81bd",
                  edgecolor="gray", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("测试集准确率")
    ax.set_ylim(0, 1.0)
    ax.axhline(1.0 / len(class_names), color="red", linestyle=":",
               linewidth=1.2, label="随机猜测 %.2f%%" % (100.0 / len(class_names)))
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a, "%.3f" % a,
                ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("传统方法基线对比")
    plt.tight_layout()
    fig.savefig(config.FIGURE_DIR / ("baseline_compare%s.png" % tag),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("图已保存到 %s" % config.FIGURE_DIR)


if __name__ == "__main__":
    main()
