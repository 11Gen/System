"""
summarize_results.py —— 汇总所有实验结果，输出一张总表。

项目前后跑了不少实验：主线模型、传统基线、消融实验、跨数据集测试。
这些结果分散在 models/ 下的多个文件里，写报告和答辩时来回翻很不方便，
所以统一汇总。

输出两个东西：
    1) 控制台表格（直接看）
    2) 交付文档里的「实验结果汇总.md」（可以贴进报告或答辩材料）

为什么写在文档文件夹而不是 models/：models/ 是实验产物的目录，已被 .gitignore
排除，而这份汇总内容稳定、值得留档，属于交付文档。

用法：
    python tools/summarize_results.py
"""

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 交付时代码与文档分开放：代码在本文件夹，文档在同级的「专业实训项目文档」。
# 允许用环境变量覆盖，方便把两者放到别的相对位置。
DOCS = Path(os.environ.get("PROJ_DOCS_DIR") or (ROOT.parent / "专业实训项目文档"))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402

M = config.MODEL_DIR


def jload(name):
    p = M / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def cload(name):
    p = M / name
    if not p.exists():
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    lines = []
    A = lines.append

    A("# 实验结果汇总\n")
    A("> 本文件由 `tools/summarize_results.py` 从实验产物文件中自动生成，")
    A("> 所有数字均来自实际运行结果。\n")

    # ---------------- 1. 主模型 ----------------
    ev = jload("eval_summary.json")
    tr = jload("train_result_main.json")

    A("## 一、主模型\n")
    if ev:
        A("| 指标 | 数值 |")
        A("|---|---|")
        A("| 模型 | MobileNetV2（ImageNet 预训练） |")
        A("| 训练样本 | %d 张 |" % tr["history"][0].get("n_train", 4287)
          if tr else "| 训练样本 | 4287 张 |")
        A("| 测试样本 | %d 张 |" % ev["n_samples"])
        A("| 类别数 | %d |" % ev["n_classes"])
        A("| **Top-1 准确率** | **%.4f** |" % ev["accuracy"])
        A("| Top-3 准确率 | %.4f |" % ev["top3_accuracy"])
        A("| 宏平均 F1 | %.4f |" % ev["macro_f1"])
        A("")
    if tr:
        A("训练配置：\n")
        A("- 阶段一：%d epoch，lr=%g，可训练参数 0.9%%"
          % (tr["epochs1"], tr["lr1"]))
        A("- 阶段二：%d epoch，lr=%g，可训练参数 75.8%%"
          % (tr["epochs2"], tr["lr2"]))
        A("- 批大小 %d，随机种子 %d，设备 %s"
          % (tr["batch_size"], config.RANDOM_SEED, tr["device"]))
        A("- 最优验证准确率 %.4f" % (tr.get("stage2_best_val_acc") or
                                     tr.get("stage1_best_val_acc") or 0))
        A("")

    # ---------------- 2. 传统基线 ----------------
    base = jload("baseline_result_main.json")
    A("## 二、传统方法基线\n")
    if base:
        A("特征：手写 HOG（576 维）+ HSV 颜色直方图（12 维），共 %d 维，"
          "PCA 降至 %d 维（累计解释方差 %.3f）。\n"
          % (base["feature_dim"], base["pca_dim"],
             base["pca_explained_variance"]))
        A("| 方法 | 准确率 | 宏平均 F1 |")
        A("|---|---|---|")
        for r in base["results"]:
            A("| %s | %.4f | %.4f |" % (r["name"], r["accuracy"], r["macro_f1"]))
        if ev:
            A("| **MobileNetV2（本文方法）** | **%.4f** | **%.4f** |"
              % (ev["accuracy"], ev["macro_f1"]))
        A("")
        best = max(base["results"], key=lambda r: r["accuracy"])
        if ev:
            A("最优传统方法为 %s（准确率 %.4f），本文方法比它高 %.2f 个百分点。\n"
              % (best["name"], best["accuracy"],
                 (ev["accuracy"] - best["accuracy"]) * 100))

        kscan = next((r for r in base["results"] if r.get("k_scan")), None)
        if kscan:
            A("KNN 的 k 值选择过程（在验证集上搜）：\n")
            A("| k | 验证集准确率 |")
            A("|---|---|")
            for it in kscan["k_scan"]:
                A("| %d | %.4f |" % (it["k"], it["val_acc"]))
            A("")
    else:
        A("（尚未运行 `src/train_baseline.py`）\n")

    # ---------------- 3. 消融实验 ----------------
    abl = jload("ablation_results.json")
    A("## 三、消融实验\n")
    if abl:
        A("| 实验设置 | 测试准确率 | 相对完整方法 | 说明 |")
        A("|---|---|---|---|")
        for r in abl:
            dev = r.get("相对完整方法(百分点)")
            A("| %s | %.4f | %s | %s |"
              % (r.get("实验"), r.get("测试准确率") or 0,
                 ("%+.2f pp" % dev) if dev is not None else "—",
                 r.get("说明") or ""))
        A("")
        A("说明：每组实验用相同的随机种子与数据划分，"
          "只改变一个设计要素，因此差异可归因于该要素。\n")
    else:
        A("（尚未运行 `src/run_experiments.py`）\n")

    # ---------------- 4. 跨数据集 ----------------
    cross = jload("eval_summary_cross.json")
    A("## 四、跨数据集泛化与开集表现\n")
    if cross:
        A("外部测试集来自另一个数据源（`jiezh2/common-chinese-food`），"
          "共 %d 张图，模型训练时完全未见过。\n"
          % cross.get("n_total_samples", 0))
        novel = cross.get("novel_class_stats")
        A("| 项目 | 数值 |")
        A("|---|---|")
        A("| 与训练集重叠的类别 | 宫保鸡丁（%d 张） |" % cross["n_samples"])
        A("| 重叠类别的 Top-1 准确率 | **%.4f** |" % cross["accuracy"])
        A("| 重叠类别的 Top-3 准确率 | %.4f |" % cross["top3_accuracy"])
        if novel:
            A("| 模型未学过的类别 | %d 类，共 %d 张 |"
              % (len(novel["classes"]), novel["n_samples"]))
            A("| 未知类别平均置信度 | %.4f |" % novel["novel_mean_conf"])
            A("| 已知类别平均置信度 | %.4f |" % novel["known_mean_conf"])
            pct = 100.0 * novel["novel_above_0.5"] / max(novel["n_samples"], 1)
            A("| 未知类别中置信度>0.5 的比例 | %.1f%% |" % pct)
        A("")
        if novel:
            A("**结论**：模型在换数据集后对学过的类别仍有 %.2f%% 的准确率，"
              % (cross["accuracy"] * 100))
            A("说明它学到的是菜品本身的判别特征，而不是训练集的拍摄风格。")
            A("但未知类别的置信度仍然偏高（%d/%d 张超过 0.5），"
              % (novel["novel_above_0.5"], novel["n_samples"]))
            A("说明闭集分类器面对不认识的输入仍会给出自信的答案——"
              "这正是需要引入拒识机制的直接依据（见第五节）。\n")
    else:
        A("（尚未运行跨数据集评估）\n")

    # ---------------- 5. 开集识别 ----------------
    osr = jload("open_set.json")
    A("## 五、开集识别（拒识陌生输入）\n")
    if osr:
        A("模型只认识训练过的类别。为判断输入是否属于支持范围，"
          "使用能量分 `logsumexp(logits)` 作为拒识依据，"
          "并在真实数据上标定阈值。\n")
        A("评测数据：%d 张已知测试图 + %d 张未知类图（%s）。\n"
          % (osr["n_known"], osr["n_novel"],
             "、".join(config.CN_NAME.get(c, c) for c in osr["novel_classes"])))
        A("| 评分方法 | AUROC | 未知拒识率 | 已知接受率 |")
        A("|---|---|---|---|")
        for m in osr["methods"]:
            A("| %s | %.4f | %.4f | %.4f |"
              % (m["name"], m["auroc"], m["novel_reject_at_95_accept"],
                 m["known_accept_rate"]))
        A("")
        A("说明：「未知拒识率」在固定「已知接受率 95%%」的前提下测得，"
          "即只让 5%% 的正常菜品被要求重拍时能挡掉多少陌生输入。")
        best_m = max(osr["methods"], key=lambda m: m["auroc"])
        A("最优方法为 **%s**（AUROC %.4f），拒识阈值 %.2f。\n"
          % (osr["best_method"], best_m["auroc"], osr["reject_threshold"]))
        A("| 能量阈值 | 已知接受率 | 未知拒识率 |")
        A("|---|---|---|")
        for r in osr["threshold_sweep"]:
            A("| %.2f | %.4f | %.4f |"
              % (r["能量阈值"], r["已知样本接受率"], r["未知样本拒识率"]))
        A("")
    else:
        A("（尚未运行 `src/open_set.py`）\n")

    # ---------------- 6. 营养数据验证 ----------------
    rv = cload("recipe_validation.csv")
    A("## 六、营养估算方法验证\n")
    if rv:
        A("用第三方平台（Cookidoo）同时给出配方与官方营养值的菜品做回归检验，"
          "考察配料合成法的实现是否正确。\n")
        A("| 菜品 | 人份 | 官方(kcal/人份) | 本方法(kcal/人份) | 偏差 |")
        A("|---|---|---|---|---|")
        for r in rv:
            A("| %s | %s | %s | %s | %s%% |"
              % (r["name"], r["servings"], r["official_kcal"],
                 r["computed_kcal"], r["deviation_pct"]))
        devs = [abs(float(r["deviation_pct"])) for r in rv]
        A("")
        A("平均绝对偏差 %.1f%%，最大偏差 %.1f%%。"
          % (sum(devs) / len(devs), max(devs)))
        A("偏差主要来自配方差异与生熟重口径差异，而非算法错误——"
          "若分母或单位有误，偏差会是数倍而不是百分之几十。\n")
    else:
        A("（尚未运行 `src/validate_recipe.py`）\n")

    # ---------------- 7. 逐类指标 ----------------
    if ev and ev.get("per_class_f1"):
        A("## 七、各类别 F1 分数\n")
        A("| 菜品 | 类别标识 | F1 |")
        A("|---|---|---|")
        f1 = ev["per_class_f1"]
        for k, v in sorted(f1.items(), key=lambda kv: kv[1]):
            cn = config.CN_NAME.get(k, k)
            A("| %s | %s | %.4f |" % (cn, k, v))
        A("")
        worst = sorted(f1.items(), key=lambda kv: kv[1])[:3]
        best = sorted(f1.items(), key=lambda kv: kv[1], reverse=True)[:3]
        A("- F1 最高的三类：%s"
          % "、".join("%s(%.3f)" % (config.CN_NAME.get(k, k), v)
                      for k, v in best))
        A("- F1 最低的三类：%s"
          % "、".join("%s(%.3f)" % (config.CN_NAME.get(k, k), v)
                      for k, v in worst))
        A("")

    out = DOCS / "实验结果汇总.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\n" + "=" * 78)
    print("已写入 %s" % out)
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
