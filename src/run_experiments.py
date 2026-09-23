"""
run_experiments.py —— 消融实验编排脚本。

项目主线的准确率只说明"这套方法能到多少分"，说明不了
"每个设计选择各自贡献了多少"。所以这里把几个关键设计点逐个关掉，
看准确率掉多少 —— 掉得多的说明那个设计是必要的，掉得少的说明可以省掉。

设计的四组实验：
    1. 完整方法（预训练 + 两阶段 + 数据增强）—— 基线
    2. 去掉阶段二微调（只训练分类头）
    3. 去掉数据增强
    4. 去掉 ImageNet 预训练（从随机初始化开始）
       ↑ 这一组最关键：它直接量化了"迁移学习"本身的价值

注意：每组的权重用不同的文件名前缀保存。
一开始没做区分，四组实验都往 models/best.pt 写，
结果跑完只剩最后一组的权重，前面几组没法复现评估。

用法：
    python src/run_experiments.py --list         # 看看有哪些实验
    python src/run_experiments.py --only nocnn   # 只跑某一组
    python src/run_experiments.py                # 全部跑（耗时较长）
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import MODEL_DIR

# 实验定义。epochs 给得比主线少一些，因为消融实验看的是相对差异，
# 不需要每组都训到完全收敛 —— 而且 CPU 上每组都要半小时以上。
EXPERIMENTS = [
    {
        "tag": "full",
        "name": "完整方法",
        "desc": "ImageNet 预训练 + 两阶段训练 + 数据增强",
        "args": [],
    },
    {
        "tag": "noaug",
        "name": "去掉数据增强",
        "desc": "关掉 RandomResizedCrop / Flip / ColorJitter，只做缩放",
        "args": ["--aug", "none"],
    },
    {
        "tag": "stage1only",
        "name": "只训练分类头",
        "desc": "冻结主干，不做阶段二微调",
        "args": ["--stage1-only"],
    },
    {
        "tag": "scratch",
        "name": "不用预训练",
        "desc": "主干从随机初始化开始训练（量化迁移学习的价值）",
        "args": ["--no-pretrained"],
    },
]


def run_one(exp, epochs1, epochs2, extra):
    cmd = [sys.executable, "-u", str(Path(__file__).parent / "train_cnn.py"),
           "--epochs1", str(epochs1), "--epochs2", str(epochs2),
           "--tag", exp["tag"]] + exp["args"] + extra
    print("\n" + "=" * 84)
    print(" 实验：%s（%s）" % (exp["name"], exp["tag"]))
    print(" 说明：%s" % exp["desc"])
    print(" 命令：%s" % " ".join(cmd[1:]))
    print("=" * 84)

    t0 = time.time()
    # 用 subprocess 而不是直接调函数：每组实验要独立的进程，
    # 否则 torch 的显存/线程池状态和全局随机数种子会互相影响
    proc = subprocess.run(cmd, cwd=str(config.ROOT))
    dt = time.time() - t0

    if proc.returncode != 0:
        print("  [失败] 退出码 %d，耗时 %.1f 分钟"
              % (proc.returncode, dt / 60))
        return None

    rf = MODEL_DIR / ("train_result_%s.json" % exp["tag"])
    if not rf.exists():
        print("  [警告] 没有找到结果文件 %s" % rf.name)
        return None

    with open(rf, encoding="utf-8") as f:
        res = json.load(f)
    res["_name"] = exp["name"]
    res["_desc"] = exp["desc"]
    res["_minutes"] = round(dt / 60, 1)
    print("  [完成] 测试集准确率 %.4f，耗时 %.1f 分钟"
          % (res.get("test_acc", 0), dt / 60))
    return res


def make_table(results, baseline_tag="full"):
    """把各实验的结果汇总成一张对比表，并算出相对完整方法的差距"""
    base = next((r for r in results
                 if r and (r.get("tag") or r.get("TAG")) == baseline_tag), None)
    base_acc = base["test_acc"] if base else None

    print("\n" + "=" * 92)
    print(" 消融实验结果汇总")
    print("=" * 92)
    print("%-18s %10s %10s %12s %10s"
          % ("实验", "验证准确率", "测试准确率", "相对完整方法", "耗时(分)"))
    print("-" * 92)

    rows = []
    for r in results:
        if not r:
            continue
        name = r["_name"]
        va = r.get("stage2_best_val_acc") or r.get("stage1_best_val_acc") or 0
        ta = r.get("test_acc", 0)
        if base_acc:
            delta = ta - base_acc
            delta_s = "%+.2f 个百分点" % (delta * 100)
        else:
            delta_s = "-"
        print("%-18s %10.4f %10.4f %12s %10.1f"
              % (name, va, ta, delta_s, r.get("_minutes", 0)))
        rows.append({
            "实验": name, "标签": r.get("tag"),
            "验证准确率": va, "测试准确率": ta,
            "相对完整方法(百分点)": round((ta - base_acc) * 100, 2) if base_acc else None,
            "耗时(分钟)": r.get("_minutes"),
            "说明": r.get("_desc"),
        })

    print("-" * 92)
    if base_acc:
        print("\n完整方法（%s）测试集准确率 %.4f 作为基准线。" % (baseline_tag, base_acc))

    # 自动生成几条写报告能直接用的结论
    print("\n可以直接写进报告的结论：")
    for r in results:
        if not r or r.get("tag") == baseline_tag or not base_acc:
            continue
        delta = (r["test_acc"] - base_acc) * 100
        direction = "下降" if delta < 0 else "上升"
        print("  · 去掉「%s」后准确率%s %.2f 个百分点（%.4f -> %.4f）"
              % (r["_name"], direction, abs(delta), base_acc, r["test_acc"]))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只列出实验，不执行")
    ap.add_argument("--only", default=None, help="只跑指定 tag 的实验")
    ap.add_argument("--epochs1", type=int, default=8)
    ap.add_argument("--epochs2", type=int, default=6)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="透传给 train_cnn.py 的额外参数")
    args = ap.parse_args()

    if args.list:
        print("已定义的消融实验：")
        for e in EXPERIMENTS:
            print("  %-12s %-16s %s" % (e["tag"], e["name"], e["desc"]))
        return

    todo = EXPERIMENTS
    if args.only:
        todo = [e for e in EXPERIMENTS if e["tag"] == args.only]
        if not todo:
            print("没有找到 tag 为 %s 的实验" % args.only)
            print("可选：%s" % [e["tag"] for e in EXPERIMENTS])
            sys.exit(1)

    print("将执行 %d 组实验，每组 epochs1=%d epochs2=%d"
          % (len(todo), args.epochs1, args.epochs2))
    print("CPU 环境下预计每组 15~40 分钟，请耐心等待。")

    config.ensure_dirs()
    results = []
    t_all = time.time()
    for e in todo:
        results.append(run_one(e, args.epochs1, args.epochs2, args.extra))

    # 汇总时把已有的历史结果也带上（支持分批跑）
    for e in EXPERIMENTS:
        if e in todo:
            continue
        rf = MODEL_DIR / ("train_result_%s.json" % e["tag"])
        if rf.exists():
            with open(rf, encoding="utf-8") as f:
                r = json.load(f)
            r["_name"] = e["name"]
            r["_desc"] = e["desc"]
            r["_minutes"] = None
            results.insert(EXPERIMENTS.index(e), r)

    rows = make_table(results)

    out = MODEL_DIR / "ablation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n汇总已写入 %s" % out)
    print("总耗时 %.1f 分钟" % ((time.time() - t_all) / 60))


if __name__ == "__main__":
    main()
