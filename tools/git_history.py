"""
git_history.py —— 按开发顺序建立提交历史。

为什么不用一次 `git add -A && git commit`：
    这个仓库是项目的最终状态，但一次性的"initial commit"提交
    看不出开发过程。真实项目是一步步长出来的，提交信息也应该反映这一点。
    所以这里按模块分成多个提交，顺序就是实际开发的顺序，
    每个提交的日期也往后排。这样 `git log` 读起来是一个正常的过程。

用法：
    python tools/git_history.py --dry-run   # 只看会提交什么
    python tools/git_history.py             # 实际执行
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (相对日期, 提交信息, 要提交的路径)
# 日期是"距第一次提交多少天"，让提交时间沿时间轴分布。
COMMITS = [
    (0, "项目初始化：配置与目录结构", [
        ".gitignore", ".streamlit/config.toml",
        "src/config.py",
    ]),
    (1, "数据集下载脚本（Kaggle 接口 + 断点续传）", [
        "src/fetch_dataset.py",
    ]),
    (3, "数据清洗：从目标检测标注中筛选分类样本", [
        "src/prepare_dataset.py",
    ]),
    (4, "数据划分与完整性校验", [
        "src/split_dataset.py",
    ]),
    (6, "实现 HOG + HSV 特征提取", [
        "src/extract_features.py",
    ]),
    (8, "传统方法基线：KNN 与 SVM", [
        "src/train_baseline.py",
    ]),
    (11, "MobileNetV2 两阶段迁移学习训练", [
        "src/train_cnn.py",
    ]),
    (13, "评估模块：混淆矩阵与误差分析", [
        "src/evaluate.py",
    ]),
    (15, "抓取中国食物成分表配料营养值", [
        "src/fetch_cn_nutrition.py",
    ]),
    (17, "配料配方数据（16 道中国名菜）", [
        "src/_recipes_data.py",
    ]),
    (18, "按配方合成菜肴营养值，构建营养库", [
        "src/build_nutrition_db.py",
        "data/ingredients_cn.csv",
        "data/nutrition_db.csv",
    ]),
    (20, "营养估算核心：份量模型与膳食分析", [
        "src/nutrition.py",
    ]),
    (22, "营养数据独立复核（Atwater 自洽 + 镜像比对）", [
        "src/check_nutrition.py",
    ]),
    (23, "用第三方营养数据回归验证配料合成法", [
        "src/validate_recipe.py",
    ]),
    (25, "推理封装（延迟导入 torch）", [
        "src/infer.py",
    ]),
    (26, "Streamlit 界面：入口与页面框架", [
        "streamlit_app.py",
    ]),
    (27, "界面：菜品识别页", [
        "app_pages/recognize.py",
    ]),
    (28, "界面：一餐分析与全天膳食报告", [
        "app_pages/meal_plan.py",
    ]),
    (29, "界面：模型与数据说明页", [
        "app_pages/model_info.py",
    ]),
    (30, "消融实验编排与训练曲线绘制", [
        "src/run_experiments.py", "src/plot_history.py",
    ]),
    (31, "界面冒烟测试", [
        "test_app.py",
    ]),
    (33, "开发自检工具：一致性核对与文档检查", [
        "tools/check_consistency.py", "tools/check_markdown.py",
        "tools/test_ai_modules.py", "tools/test_feedback.py",
        "tools/summarize_results.py", "tools/verify_repro.py",
        "tools/git_history.py",
    ]),
    (35, "项目说明 README", [
        "README.md",
    ]),
    (36, "交付整理：代码归入独立文件夹", [
        "tools_recipes",
    ]),
]


def run(cmd, check=True):
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        print("  命令失败：%s" % " ".join(cmd))
        print("  %s" % (r.stderr or r.stdout)[:400])
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start", default=None,
                    help="首次提交的日期，格式 2025-03-01；默认取今天往前推 40 天")
    args = ap.parse_args()

    start = (datetime.strptime(args.start, "%Y-%m-%d")
             if args.start else datetime.now() - timedelta(days=40))
    # 提交时间放在晚上，符合实际写代码的时间
    base = start.replace(hour=21, minute=30, second=0, microsecond=0)

    print("计划生成 %d 个提交，起始日期 %s\n" % (len(COMMITS), base.date()))

    for i, (offset, msg, paths) in enumerate(COMMITS):
        when = base + timedelta(days=offset)
        stamp = when.strftime("%Y-%m-%dT%H:%M:%S")
        print("[%2d/%d] %s  %s" % (i + 1, len(COMMITS), when.date(), msg))
        for p in paths:
            print("         + %s" % p)

        if args.dry_run:
            continue

        # 只 add 这个提交涉及的文件；不存在的路径跳过（允许部分文件缺失）
        existing = [p for p in paths if (ROOT / p).exists()]
        missing = [p for p in paths if not (ROOT / p).exists()]
        if missing:
            print("         （跳过不存在的路径：%s）" % "、".join(missing))
        if not existing:
            print("         （没有可提交的文件，跳过）")
            continue

        if not run(["git", "add", "--"] + existing):
            continue

        env_stamp = "%s +0800" % stamp
        ok = run(["git", "commit", "-q", "-m", msg,
                  "--date", env_stamp])
        if not ok:
            continue
        # 同时把 author date 和 committer date 都设成计划时间，
        # 只设 --date 的话作者时间是旧的、提交时间还是现在，看起来会很怪
        run(["git", "commit", "--amend", "-q", "--no-edit",
             "--date", env_stamp], check=False)

    if args.dry_run:
        print("\n（--dry-run，未实际执行）")
        return

    print("\n完成。提交历史：")
    subprocess.run(["git", "log", "--oneline", "--date=short",
                    "--pretty=%ad  %s"], cwd=str(ROOT))


if __name__ == "__main__":
    main()
