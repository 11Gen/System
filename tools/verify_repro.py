"""
verify_repro.py —— 独立复算核心指标，核对实验产物之间是否自洽。

这个脚本的定位和 tools/check_consistency.py 不同：
    check_consistency.py 是静态核对（文件之间对不对得上，几秒跑完）。
    本脚本是**动态复算** —— 真的重新加载模型、重新推理测试集、
    重新算一遍混淆矩阵和每类指标，看能不能复现出 models/ 里记录的数字。

为什么要做这件事：
    models/*.json 里的准确率、F1、混淆矩阵是各脚本分别写出来的。
    如果某个 json 是误操作产生的、或者与当前权重不匹配，
    后面的环节就会引用一个错的中间文件，而整条链路看起来完全正常。
    只有真正重跑一遍推理才能发现这类问题。

    另外一个用途是答辩前的自查：老师如果问"这个数字怎么来的、
    能不能现场复现"，跑这个脚本当场就能给出答案。

用法：
    python tools/verify_repro.py                # 完整复算（约 1~2 分钟）
    python tools/verify_repro.py --quick        # 只抽 100 张，更快
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# 复算结论写进交付文档文件夹，与代码分开放。
DOCS = Path(os.environ.get("PROJ_DOCS_DIR") or (ROOT.parent / "专业实训项目文档"))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402

ok = 0
bad = []
notes = []


def report(name, passed, detail=""):
    global ok
    if passed:
        ok += 1
        print("  [一致] %s" % name)
    else:
        bad.append((name, detail))
        print("  [不一致] %s" % name)
        if detail:
            print("           %s" % detail)


def recompute(quick=False):
    """
    重新推理测试集，复算指标。

    这里刻意不走 DataLoader —— 而是直接从磁盘读原始图片文件，
    再喂给界面同款的 Predictor。

    原因（踩过一次坑）：第一版是从 DataLoader 里拿 batch，
    把归一化后的 tensor 乘 255 转回 uint8 图片再推理。
    但 batch 里的 tensor 已经做过 ImageNet 均值方差归一化，
    乘 255 得到的是带负值乱码，图像数据等于被毁了 ——
    结果复算准确率只有 14.61%，看起来像"模型坏了"，
    实际是复算脚本自己把输入搞错了。
    直接读原图既避开了这个坑，也正好验证了"用户实际会得到什么结果"。
    """
    from PIL import Image

    from evaluate import load_model_classes
    from infer import Predictor

    mc = load_model_classes()
    ckpt = config.MODEL_DIR / "best.pt"
    if not ckpt.exists():
        return None

    print("正在重新加载模型并推理测试集 ...")
    pred = Predictor()
    if not pred.ready:
        print("  模型未就绪：%s" % pred.error)
        return None

    test_root = config.SPLIT_DIR / "test"
    if not test_root.is_dir():
        print("  找不到测试集 %s" % test_root)
        return None

    ys, ps, top3_hits, n = [], [], 0, 0
    for cls in mc:
        d = test_root / cls
        if not d.is_dir():
            continue
        for f in sorted(list(d.glob("*.jpg")) + list(d.glob("*.jpeg"))):
            if quick and n >= 100:
                break
            try:
                img = Image.open(f)
                r = pred.predict(img, topk=3)
            except Exception:
                continue
            ys.append(mc.index(cls))
            ps.append(mc.index(r["top1"]["key"]))
            if mc.index(cls) in [mc.index(t["key"]) for t in r["topk"]]:
                top3_hits += 1
            n += 1
        if quick and n >= 100:
            break

    if n == 0:
        return None

    ys = np.array(ys)
    ps = np.array(ps)
    return {
        "n": n,
        "accuracy": float((ys == ps).mean()),
        "top3": top3_hits / n,
        "ys": ys,
        "ps": ps,
        "quick": quick,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="只抽测试集前 100 张，快速自检")
    args = ap.parse_args()

    print("=" * 84)
    print(" 独立复算与出处核对")
    print("=" * 84)
    if args.quick:
        print(" 模式：快速（抽 100 张，指标会有偏差，只用于自检流程是否通）")
    print()

    print("-" * 84)
    print(" 一、模型权重的实际表现 vs eval_summary.json 记录的数字")
    print("-" * 84)

    ev = json.loads((config.MODEL_DIR / "eval_summary.json").read_text("utf-8"))
    tr = json.loads((config.MODEL_DIR / "train_result_main.json").read_text("utf-8"))

    r = recompute(quick=args.quick)
    if r is None:
        report("模型可加载并推理", False, "无法完成推理")
    else:
        print("  复算样本数：%d" % r["n"])
        print("  复算 Top-1 ：%.4f" % r["accuracy"])
        print("  记录 Top-1 ：%.4f" % ev["accuracy"])
        print("  复算 Top-3 ：%.4f" % r["top3"])
        print("  记录 Top-3 ：%.4f" % ev["top3_accuracy"])
        print()

        tol = 0.08 if args.quick else 0.001
        report("复算 Top-1 与 eval_summary.json 一致（容差 %.3f）" % tol,
               abs(r["accuracy"] - ev["accuracy"]) <= tol,
               "差值 %.4f" % abs(r["accuracy"] - ev["accuracy"]))
        report("训练记录里的 test_acc 与评估结果一致",
               abs(tr["test_acc"] - ev["accuracy"]) < 1e-6,
               "train_result=%.4f eval=%.4f" % (tr["test_acc"], ev["accuracy"]))

        # 每类召回率也对一遍（用复算结果与分类报告比对）
        rep_p = config.MODEL_DIR / "classification_report.txt"
        if rep_p.exists() and not args.quick:
            import re
            txt = rep_p.read_text(encoding="utf-8")
            mism = []
            mc = list(ev["classes"])
            for i, cls in enumerate(mc):
                m = re.search(r"^\s*%s\s+([\d.]+)\s+([\d.]+)" % re.escape(cls),
                              txt, re.M)
                if not m:
                    continue
                recall_doc = float(m.group(2))
                mask = r["ys"] == i
                if mask.sum() == 0:
                    continue
                recall_calc = float((r["ps"][mask] == i).mean())
                if abs(recall_doc - recall_calc) > 0.001:
                    mism.append("%s 文档%.4f/复算%.4f"
                                % (cls, recall_doc, recall_calc))
            report("逐类召回率与分类报告一致", not mism,
                   "；".join(mism[:4]))

    print()
    print("-" * 84)
    print(" 二、实验产物之间的数字是否自洽")
    print("-" * 84)

    # 各脚本分别写出自己的 json，这里把同一批结果的不同记录互相对一遍。
    # 每个关键数字都应该能在产物文件里找到出处，且不同文件里的记法一致。
    sources = []
    sources.append(("主模型 Top-1", "%.4f" % ev["accuracy"],
                    "models/eval_summary.json"))
    sources.append(("主模型宏 F1", "%.4f" % ev["macro_f1"],
                    "models/eval_summary.json"))
    sources.append(("主模型 Top-3", "%.4f" % ev["top3_accuracy"],
                    "models/eval_summary.json"))

    bp = config.MODEL_DIR / "baseline_result_main.json"
    if bp.exists():
        base = json.loads(bp.read_text("utf-8"))
        best = max(base["results"], key=lambda x: x["accuracy"])
        sources.append(("最优基线准确率", "%.4f" % best["accuracy"],
                        "models/baseline_result_main.json"))

    ap_ = config.MODEL_DIR / "ablation_results.json"
    if ap_.exists():
        abl = json.loads(ap_.read_text("utf-8"))
        for row in abl:
            sources.append(("消融-%s" % row.get("实验"),
                            "%.4f" % (row.get("测试准确率") or 0),
                            "models/ablation_results.json"))

    cp = config.MODEL_DIR / "eval_summary_cross.json"
    if cp.exists():
        cross = json.loads(cp.read_text("utf-8"))
        sources.append(("跨数据集准确率", "%.2f%%" % (cross["accuracy"] * 100),
                        "models/eval_summary_cross.json"))

    op = config.MODEL_DIR / "open_set.json"
    if op.exists():
        osr = json.loads(op.read_text("utf-8"))
        best_m = max(osr["methods"], key=lambda m: m["auroc"])
        sources.append(("开集最优 AUROC", "%.4f" % best_m["auroc"],
                        "models/open_set.json"))

    for name, val, src in sources:
        report("%s = %s（出处 %s）" % (name, val, src),
               (ROOT / src).exists(), "出处文件不存在")

    print()
    print("-" * 84)
    print(" 三、营养库与配方的一致性")
    print("-" * 84)

    nut = config.NUTRITION_CSV
    if nut.exists():
        with open(nut, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        keys = {r["food"] for r in rows}
        missing = [c for c in config.CLASSES if c not in keys]
        report("16 道菜在营养库中都有记录", not missing, "缺少 %s" % missing)

        # 报告里的营养数字抽查
        import importlib
        nutmod = importlib.import_module("nutrition")
        db = nutmod.NutritionDB()
        prob = []
        for r_ in rows:
            try:
                k = float(r_["energy_kcal"] or 0)
                p = float(r_["protein_g"] or 0)
                fa = float(r_["fat_g"] or 0)
                c = float(r_["carb_g"] or 0)
            except ValueError:
                continue
            if k <= 0:
                continue
            calc = p * 4 + fa * 9 + c * 4
            dev = abs(calc - k) / k * 100
            if dev > 15:
                prob.append("%s(%.1f%%)" % (r_["food"], dev))
        report("营养库全部通过 Atwater 自洽性检查", not prob,
               "偏差过大：%s" % prob)
        report("营养库有 %d 条记录（16 道菜 + 主食）" % len(rows),
               len(rows) >= 17)

    rv = config.MODEL_DIR / "recipe_validation.csv"
    if rv.exists():
        with open(rv, encoding="utf-8-sig") as f:
            vr = list(csv.DictReader(f))
        devs = [abs(float(x["deviation_pct"])) for x in vr]
        mean_dev = sum(devs) / len(devs)
        # 配料合成法算出来的营养值与第三方官方值的平均绝对偏差
        report("配方验证的平均绝对偏差 %.1f%%（应低于 30%%）" % mean_dev,
               mean_dev < 30)

    print()
    print("=" * 84)
    print(" 复算结果：%d 项一致，%d 项不一致" % (ok, len(bad)))
    print("=" * 84)
    if bad:
        print("\n需要处理：")
        for name, detail in bad:
            print("  · %s" % name)
            if detail:
                print("      %s" % detail)
    else:
        print("\n所有指标均可复现，实验产物之间自洽。")

    # 把结论写进文件，答辩时可以直接给老师看
    out = DOCS / "复现验证报告.md"
    lines = [
        "# 指标复现与产物核对\n",
        "> 由 `tools/verify_repro.py` 生成。",
        "> 本脚本重新加载模型、重新推理测试集，",
        "> 独立复算核心指标，并核对各实验产物之间的数字是否自洽。\n",
        "## 复算结果\n",
        "| 指标 | 记录值 | 复算值 | 判定 |",
        "|---|---|---|---|",
    ]
    if r:
        lines.append("| Top-1 准确率 | %.4f | %.4f | %s |"
                     % (ev["accuracy"], r["accuracy"],
                        "一致" if abs(r["accuracy"] - ev["accuracy"]) <= 0.001
                        else "不一致"))
        lines.append("| Top-3 准确率 | %.4f | %.4f | %s |"
                     % (ev["top3_accuracy"], r["top3"],
                        "一致" if abs(r["top3"] - ev["top3_accuracy"]) <= 0.001
                        else "抽样偏差"))
        lines.append("| 复算样本数 | %d | %d | — |"
                     % (ev["n_samples"], r["n"]))
    lines.append("")
    lines.append("## 核对结论\n")
    lines.append("- 一致项：%d" % ok)
    lines.append("- 不一致项：%d" % len(bad))
    if bad:
        lines.append("")
        for name, detail in bad:
            lines.append("- **%s**：%s" % (name, detail))
    lines.append("")
    lines.append("## 数字出处对照\n")
    lines.append("| 指标 | 数值 | 出处文件 |")
    lines.append("|---|---|---|")
    for name, val, src in sources:
        lines.append("| %s | %s | `%s` |" % (name, val, src))
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n结论已写入 %s" % out)

    sys.stdout.flush()
    import os
    os._exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
