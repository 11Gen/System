"""
check_consistency.py —— 全项目一致性检查。

这个脚本不训练、不推理，只做静态核对，几秒就能跑完。
用途是交付前自查：确认各个文件里记录的类别、数字、路径互相对得上。

它检查的这几类问题，都是本项目实际踩过的坑：
  1. 类别清单在 config / classes.json / 数据目录三处不一致
     → 会导致标签整体错位，而且不会报错，只是准确率莫名很低
  2. 营养库里缺少某些类别的记录
     → 界面上选中这类菜会拿不到数据
  3. 实验产物之间对不上（评估结果与训练记录、阈值两处记录不一致）
     → 各脚本各写各的 json，其中一个过期了不会报错
  4. 交付文档落后于代码（功能加了但文档没提）
  5. 引用了不存在的脚本或文件

用法：
    python tools/check_consistency.py
"""

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 交付时代码与文档分开放：代码在本文件夹，文档在同级的「专业实训项目文档」。
DOCS = Path(os.environ.get("PROJ_DOCS_DIR") or (ROOT.parent / "专业实训项目文档"))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402

# ---------------------------------------------------------------- 检查用的常量
# 全部常量集中放在文件最开头。
#
# 这不是风格偏好，是踩过坑：之前把 DOC_MUST_MENTION 写在文件中部，
# 而引用它的循环在更上面，于是直接 NameError。
# 而且报错信息被长长的检查输出淹了，第一眼只看到"检查结果正常"，
# 差点当成通过。放在最顶部就不会有执行顺序问题。
#
# 交付文档里必须覆盖的功能模块。
# 加这几项是因为出过一次纰漏：代码加了 AI 识别整页，交付文档却完全没提。
# 这类"文档落后于代码"的问题不主动检查发现不了。
# 项目报告（Word）不在仓库里，所以这里查的是文档文件夹里的说明文档。
DOC_MUST_MENTION = [
    ("AI 识别", "AI 识别路径"),
    ("大模型", "大模型兜底说明"),
    ("同义词表", "配料名映射方案"),
    ("帮助与反馈", "帮助与反馈页的设计"),
    ("常见问题", "FAQ 的说明"),
]

# 界面导航的当前名称。改名之后，文档里如果还留着旧名字就是脱节了。
NAV_CURRENT = ["主页介绍", "识菜", "AI 识别", "膳食管家", "权威所在",
               "帮助与反馈"]
NAV_OUTDATED = ["项目概览", "菜品识别", "一餐分析", "模型与数据"]

# 这些词出现在同一行时，说明那是在「解释改名」而不是残留旧名。
# 例如：
#   「权威所在」页（原名"模型与数据"）  <- 说明改名，正当
#   「识菜」（别念成"菜品识别"）        <- 给录制者的提示，正当
# 一开始的写法是「出现即报错」，结果这两种情况都被误判了。
RENAME_MARKERS = ["原名", "旧名", "别念成", "已改名", "改成"]

# 已经实现、因此不该再写成"待办"的功能。
# 视频脚本里曾经出现"下一步做开集识别"，而开集识别当时已经做完了 ——
# 照着念会当场讲错。
DONE_NOT_TODO = [
    ("开集识别", ["打算做开集识别", "没有做开集识别", "不会拒识"]),
]

ok_count = 0
problems = []


def check(name, cond, detail=""):
    global ok_count
    if cond:
        ok_count += 1
        print("  [通过] %s" % name)
    else:
        problems.append((name, detail))
        print("  [问题] %s" % name)
        if detail:
            print("         %s" % detail)


print("=" * 78)
print(" 一、类别一致性")
print("=" * 78)
classes_cfg = list(config.CLASSES)
print("  config.CLASSES: %d 类" % len(classes_cfg))

cp = config.SPLIT_DIR / "classes.json"
if cp.exists():
    with open(cp, encoding="utf-8") as f:
        classes_json = json.load(f)
    check("config.CLASSES 与 classes.json 一致（含顺序）",
          classes_cfg == classes_json,
          "配置: %s\n         文件: %s" % (classes_cfg, classes_json))
else:
    check("classes.json 存在", False, "先跑 src/split_dataset.py")

# 数据目录里的类别
train_dir = config.SPLIT_DIR / "train"
if train_dir.is_dir():
    dirs = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    check("数据目录类别与 config 一致（排序后）",
          dirs == sorted(classes_cfg),
          "目录: %s" % dirs)

# 中文名齐全
missing_cn = [c for c in classes_cfg if c not in config.CLASS_INFO]
check("每个类别都有中文名/菜系信息", not missing_cn, "缺少: %s" % missing_cn)

print()
print("=" * 78)
print(" 二、营养库覆盖")
print("=" * 78)
nut_csv = config.NUTRITION_CSV
if nut_csv.exists():
    with open(nut_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = {r["food"] for r in rows}
    print("  营养库记录数: %d" % len(rows))
    missing_nut = [c for c in classes_cfg if c not in keys]
    check("16 道菜都有营养记录", not missing_nut, "缺少: %s" % missing_nut)

    # 每条都要有来源信息（这是项目的可追溯性要求）
    no_src = [r["food"] for r in rows if not (r.get("source_id") or "").strip()]
    check("每条营养记录都有来源编号", not no_src, "缺少来源: %s" % no_src)

    # Atwater 自洽性
    bad = []
    for r in rows:
        try:
            k = float(r["energy_kcal"] or 0)
            p = float(r["protein_g"] or 0)
            f_ = float(r["fat_g"] or 0)
            c = float(r["carb_g"] or 0)
        except ValueError:
            continue
        if k <= 0:
            continue
        calc = p * 4 + f_ * 9 + c * 4
        dev = (calc - k) / k * 100
        if abs(dev) > 15:
            bad.append("%s(%.1f%%)" % (r["food"], dev))
    check("全部通过 Atwater 能量自洽性检查（偏差<=15%）", not bad,
          "偏差过大: %s" % bad)
else:
    check("营养库文件存在", False, "先跑 src/build_nutrition_db.py")

# 配料表
ing_csv = config.DATA_DIR / "ingredients_cn.csv"
if ing_csv.exists():
    with open(ing_csv, encoding="utf-8-sig") as f:
        ings = list(csv.DictReader(f))
    check("配料营养表存在且有数据", len(ings) > 20, "共 %d 条" % len(ings))

print()
print("=" * 78)
print(" 三、实验产物之间是否一致")
print("=" * 78)
ev_p = config.MODEL_DIR / "eval_summary.json"
train_p = config.MODEL_DIR / "train_result_main.json"
base_p = config.MODEL_DIR / "baseline_result_main.json"
ckpt_p = config.MODEL_DIR / "best.pt"

check("模型权重 best.pt 存在", ckpt_p.exists(),
      "先跑 src/train_cnn.py")
check("评估结果 eval_summary.json 存在", ev_p.exists(),
      "先跑 src/evaluate.py")
check("训练记录 train_result_main.json 存在", train_p.exists())
check("基线结果 baseline_result_main.json 存在", base_p.exists(),
      "先跑 src/train_baseline.py")

if ev_p.exists() and train_p.exists():
    ev = json.loads(ev_p.read_text(encoding="utf-8"))
    tr = json.loads(train_p.read_text(encoding="utf-8"))
    check("评估准确率与训练记录一致",
          abs(ev["accuracy"] - tr["test_acc"]) < 1e-6,
          "eval=%.4f train=%.4f" % (ev["accuracy"], tr["test_acc"]))

    # 交付文档是否覆盖了较新的功能模块。
    # 加这几项是因为出过一次纰漏：代码加了 AI 识别整页，文档却完全没提。
    # 这类"文档落后于代码"的问题不检查发现不了。
    #
    # 注意这里引用的是上面的 DOC_MUST_MENTION 常量。
    # 之前这里内联了一份同样的列表，结果后来往常量里加条目时
    # 这个循环没跟着变 —— 检查项悄悄失效，而输出仍然显示"通过"。
    # 这类"改了但没生效"的问题比报错更难发现。
    if DOCS.is_dir():
        doc_text = "\n".join(
            p.read_text(encoding="utf-8", errors="ignore")
            for p in sorted(DOCS.glob("*.md")))
        for kw, what in DOC_MUST_MENTION:
            check("交付文档里提到了「%s」" % what, kw in doc_text,
                  "文档可能落后于代码，改完记得同步说明文档")
    else:
        check("交付文档文件夹可读", False, "找不到：%s" % DOCS)


print()
print("=" * 78)
print(" 五、交付文档与代码是否同步")
print("=" * 78)

_docs = {
    "演示视频脚本": DOCS / "演示视频脚本.md",
    "项目文件清单": DOCS / "项目文件清单.md",
    "README": ROOT / "README.md",
}
for label, p in _docs.items():
    if not p.exists():
        # 文档搬到交付文件夹之后，路径不对就会静默跳过 —— 那样这几项等于没查。
        check("%s 可读" % label, False, "找不到文件：%s" % p)
        continue
    t = p.read_text(encoding="utf-8")

    # 只匹配**被引号引起来的**旧名。
    #
    # 为什么必须限定引号：这些词在日常叙述里也会出现，但意思不同 ——
    # "一个能跑的菜品识别系统"里的"菜品识别"是普通名词，
    # 而"切到「菜品识别」"才是在指页面。
    # 一开始写成「出现即报错」，这两种情况分不开，全是误报。
    stale = []
    for ln in t.split("\n"):
        if any(mk in ln for mk in RENAME_MARKERS):
            continue
        for old in NAV_OUTDATED:
            if ("「%s」" % old) in ln or ('"%s"' % old) in ln:
                stale.append("%s（%s）" % (old, ln.strip()[:36]))
    check("%s 没有残留旧的导航名" % label, not stale,
          "以下位置把旧名当页面引用：%s" % "；".join(stale[:3]))

    for feat, bad_phrases in DONE_NOT_TODO:
        hits = [b for b in bad_phrases if b in t]
        check("%s 没有把已完成的「%s」写成待办" % (label, feat), not hits,
              "仍出现：%s" % "、".join(hits))

# 视频脚本要覆盖全部页面，否则录制时会漏掉功能
_vs = DOCS / "演示视频脚本.md"
if _vs.exists():
    t = _vs.read_text(encoding="utf-8")
    missing = [n for n in NAV_CURRENT if n not in t]
    check("演示视频脚本覆盖全部 %d 个页面" % len(NAV_CURRENT), not missing,
          "脚本里没提到：%s" % "、".join(missing))

print()
print("=" * 78)
print(" 六、文件引用完整性")
print("=" * 78)
readme = ROOT / "README.md"
import re

# 文档和提示文本里写的运行命令如果脚本名拼错了，照着敲的人会直接撞上
# "文件不存在"。所以凡是提到源目录下某个 .py 的地方都对照一遍。
# 交付文档现在在代码文件夹外面，所以还要带上文档文件夹里的 md。
ref_sources = [readme, DOCS]
for d in ("src", "tools", "app_pages"):
    ref_sources += sorted((ROOT / d).glob("*.py"))
# 本文件里的匹配模式本身会被误当成引用，跳过自己
ref_sources = [s for s in ref_sources if s.name != "check_consistency.py"]


def _label(p):
    """给检查输出用的短路径：文档文件夹里的只显示文件名。"""
    for base in (ROOT, DOCS):
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            continue
    return p.name


ref_files, total_refs = [], set()
for src in ref_sources:
    files = sorted(src.glob("*.md")) if src.is_dir() else [src]
    for f in files:
        if not f.exists():
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        refs = set(re.findall(r"src/([a-z_0-9]+\.py)", txt))
        total_refs |= refs
        broken = sorted(r for r in refs if not (ROOT / "src" / r).exists())
        if broken:
            ref_files.append((_label(f), broken))

if ref_files:
    for name, broken in ref_files:
        print("         %s 指向不存在的脚本: %s" % (name, broken))
check("代码与文档引用的脚本都存在", not ref_files,
      "%d 个文件里有失效引用" % len(ref_files))
print("         共核对 %d 个不同的脚本引用" % len(total_refs))

# 核心脚本清单
required = [
    "config.py", "fetch_dataset.py", "prepare_dataset.py", "split_dataset.py",
    "extract_features.py", "train_baseline.py", "train_cnn.py", "evaluate.py",
    "open_set.py", "infer.py", "nutrition.py", "build_nutrition_db.py",
    "fetch_cn_nutrition.py", "check_nutrition.py",
    "vision_api.py", "ai_nutrition.py", "feedback.py", "ui_table.py",
    "ui_pages.py",
]
absent = [f for f in required if not (ROOT / "src" / f).exists()]
check("核心脚本齐全", not absent, "缺少: %s" % absent)

# 界面页面清单。少一个页面意味着导航栏里有个点了就报错的入口，
# 所以逐个核对文件是否存在。
page_files = [
    "app_pages/recognize.py", "app_pages/ai_recognize.py",
    "app_pages/meal_plan.py", "app_pages/model_info.py",
    "app_pages/feedback_page.py",
]
absent_pages = [f for f in page_files if not (ROOT / f).exists()]
check("界面页面文件齐全", not absent_pages, "缺少: %s" % absent_pages)

print()
print("=" * 78)
print(" 五、拒识阈值一致性")
print("=" * 78)
# 界面从 open_set_threshold.json 读阈值，open_set.json 里也存了一份。
# 两处必须一致，否则界面用的阈值和报告里写的对不上，
# 而且这种不一致不会报错 —— 只会让报告数字和实际行为对不上。
os_p = config.MODEL_DIR / "open_set.json"
th_p = config.MODEL_DIR / "open_set_threshold.json"
if os_p.exists() and th_p.exists():
    osr = json.loads(os_p.read_text(encoding="utf-8"))
    th = json.loads(th_p.read_text(encoding="utf-8"))
    check("open_set.json 与 open_set_threshold.json 的阈值一致",
          abs(osr["reject_threshold"] - th["reject_threshold"]) < 1e-6,
          "open_set=%.4f  阈值文件=%.4f"
          % (osr["reject_threshold"], th["reject_threshold"]))
    # 阈值必须是正数：能量分方向是"越高越像已知"，
    # 历史上曾因符号取反而得到负阈值，这里加个守卫
    check("拒识阈值为正（能量分方向正确）",
          osr["reject_threshold"] > 0,
          "当前 %.4f —— 若为负说明能量分符号取反了"
          % osr["reject_threshold"])

    # 用已知样本粗查一遍：模型对测试集前若干张的判定应该基本都接受
    try:
        from PIL import Image
        import importlib
        sys.path.insert(0, str(ROOT / "src"))
        infer = importlib.import_module("infer")
        pred = infer.Predictor()
        if pred.ready:
            n_ok, n = 0, 0
            for cls in pred.classes[:8]:
                d = config.SPLIT_DIR / "test" / cls
                for f in sorted(d.glob("*.jpg"))[:2]:
                    r = pred.predict(Image.open(f))
                    n += 1
                    if not r["uncertain"]:
                        n_ok += 1
            rate = n_ok / max(n, 1)
            check("已知样本抽样接受率 >= 80%%（当前 %.0f%%）" % (rate * 100),
                  rate >= 0.80,
                  "若偏低说明阈值或能量分方向有问题")
    except Exception as e:
        print("  （跳过阈值实测验：%s）" % str(e)[:60])
else:
    check("开集识别结果文件存在", False,
          "先跑 python src/open_set.py --save-threshold")

print()
print("=" * 78)
print(" 检查结果：%d 项通过，%d 项有问题" % (ok_count, len(problems)))
print("=" * 78)
if problems:
    print("\n需要处理的问题：")
    for name, detail in problems:
        print("  · %s" % name)
        if detail:
            print("      %s" % detail)
    sys.exit(1)
else:
    print("\n全部检查通过。")

sys.stdout.flush()
import os
os._exit(0 if not problems else 1)
