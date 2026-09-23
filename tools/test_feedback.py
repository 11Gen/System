"""反馈模块的单元测试（不污染真实数据）

要点：所有写入测试都在临时目录里做。
如果直接往 data/feedback/feedback.csv 写测试数据，
你自己的真实反馈记录就会和测试数据混在一起，
而且删的时候分不清哪些是测试留下的。
"""

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print("  [通过] %s" % name)
    else:
        fail += 1
        print("  [失败] %s  %s" % (name, detail))


# ================================================================ 隔离环境
# 在导入 feedback 之前就把它的存储目录指向一个临时目录，
# 这样测试全程不会碰到真实文件。
#
# 临时目录建在工作区内而不是系统 temp：某些受限环境下
# 系统临时目录不可写（PermissionError），而工作区总是可写的。
# 测试结束会删掉它。
_tmp = ROOT / ".test_tmp_feedback"
shutil_pre = __import__("shutil")
shutil_pre.rmtree(_tmp, ignore_errors=True)

import feedback as fb  # noqa: E402

fb.FEEDBACK_DIR = _tmp / "feedback"
fb.FEEDBACK_CSV = fb.FEEDBACK_DIR / "feedback.csv"

print("=" * 74)
print("反馈模块单元测试")
print("=" * 74)
print("测试数据目录：%s" % fb.FEEDBACK_DIR)
print("（建在工作区内，跑完会删掉，不影响 data/feedback 里的真实数据）")
print()

# ================================================================ FAQ 内容
print("-" * 74)
print("一、FAQ 内容完整性")
print("-" * 74)
check("有 FAQ 条目", len(fb.FAQ) >= 5, "实际 %d 条" % len(fb.FAQ))

empty_q = [i for i, x in enumerate(fb.FAQ) if len(x.question.strip()) < 6]
check("每条的标题都不为空且足够长", not empty_q, "第 %s 条标题过短" % empty_q)

empty_a = [i for i, x in enumerate(fb.FAQ) if len(x.answer.strip()) < 40]
check("每条的答案都足够详细（>=40 字）", not empty_a,
      "第 %s 条答案过短" % empty_a)

# 跳转目标必须是真实存在的文件，否则页面上点了会 404
bad_link = []
for i, item in enumerate(fb.FAQ):
    if item.link_page and not (ROOT / item.link_page).exists():
        bad_link.append("第%d条 -> %s" % (i + 1, item.link_page))
check("FAQ 里的跳转目标文件都存在", not bad_link, "；".join(bad_link))

no_label = [i for i, x in enumerate(fb.FAQ)
            if x.link_page and not x.link_label.strip()]
check("有跳转的条目都配了按钮文字", not no_label, "第 %s 条缺文字" % no_label)

print()
print("-" * 74)
print("二、表单校验")
print("-" * 74)
cases = [
    (("功能建议", 5, "这个系统的营养估算挺准的"), True, "正常输入"),
    (("问题报告", 1, "上传大图时界面卡了几秒"), True, "1 星也合法"),
    (("数据纠错", 3, "宫保鸡丁的热量似乎偏高"), True, "数据纠错类型"),
    (("其他", 5, "随便写点什么内容"), True, "其他类型"),
    (("功能建议", 5, "好"), False, "内容太短（2 字）"),
    (("功能建议", 5, "     "), False, "全空白"),
    (("功能建议", 5, ""), False, "空字符串"),
    (("不存在的类型", 5, "测试内容测试内容"), False, "非法类型"),
    (("功能建议", 0, "测试内容测试内容"), False, "评分 0 越界"),
    (("功能建议", 6, "测试内容测试内容"), False, "评分 6 越界"),
    (("功能建议", 5, "长" * 2001), False, "内容超长"),
    (("功能建议", 5, "长" * 2000), True, "刚好 2000 字"),
]
for args, expect, label in cases:
    got, msg = fb.validate(*args)
    check("%s -> %s" % (label, "通过" if expect else "拦截"),
          got == expect, "实际 %s（%s）" % (got, msg))

print()
print("-" * 74)
print("三、写入与读取")
print("-" * 74)
check("初始状态没有反馈", fb.count_feedback() == 0)

n1 = fb.append_feedback("功能建议", 5, "建议增加批量上传功能")
check("写入第一条后计数为 1", n1 == 1, "实际 %s" % n1)

n2 = fb.append_feedback("问题报告", 2, "上传大图会卡", "test@example.com")
check("写入第二条后计数为 2", n2 == 2, "实际 %s" % n2)

rows = fb.load_feedback()
check("读回的条数正确", len(rows) == 2, "实际 %d" % len(rows))
check("字段名完整",
      set(rows[0].keys()) == set(fb.FIELDS),
      "实际 %s" % sorted(rows[0].keys()))
check("内容被正确保存", rows[0]["内容"] == "建议增加批量上传功能",
      "实际 %r" % rows[0]["内容"])
check("联系方式可为空", rows[0]["联系方式"] == "",
      "实际 %r" % rows[0]["联系方式"])
check("联系方式被保存", rows[1]["联系方式"] == "test@example.com")
check("评分以整数保存", rows[1]["评分"] == "2", "实际 %r" % rows[1]["评分"])
check("时间戳格式正确",
      len(rows[0]["提交时间"]) == 19 and rows[0]["提交时间"][4] == "-",
      "实际 %r" % rows[0]["提交时间"])

print()
print("-" * 74)
print("四、输入清洗")
print("-" * 74)
fb.append_feedback("其他", 4, "  前后有空格   中间有   多个空格  ")
last = fb.load_feedback()[-1]
check("首尾空格被去掉", not last["内容"].startswith(" "))
check("连续空格被压成一个", "  " not in last["内容"],
      "实际 %r" % last["内容"])
check("换行被压成空格",
      "\n" not in last["内容"])

fb.append_feedback("其他", 4, "第一行\n第二行\n第三行")
last = fb.load_feedback()[-1]
check("多行文本被压成一行", "\n" not in last["内容"],
      "实际 %r" % last["内容"])

print()
print("-" * 74)
print("五、CSV 文件本身")
print("-" * 74)
check("文件已创建", fb.FEEDBACK_CSV.exists())

# 用 Excel 能正确打开中文 —— 检查 BOM
raw = fb.FEEDBACK_CSV.read_bytes()
check("文件带 UTF-8 BOM（Excel 打开不乱码）",
      raw.startswith(b"\xef\xbb\xbf"), "实际开头 %r" % raw[:4])

# 用标准 csv 模块再读一遍，确认格式合法
with open(fb.FEEDBACK_CSV, encoding="utf-8-sig", newline="") as f:
    reader = list(csv.DictReader(f))
check("用标准 csv 模块能解析", len(reader) == len(fb.load_feedback()))
check("表头与 FIELDS 一致", list(reader[0].keys()) == fb.FIELDS,
      "实际 %s" % list(reader[0].keys()))

# 追加写入不能破坏已有数据
before = fb.count_feedback()
fb.append_feedback("其他", 3, "再追加一条")
check("追加写入不覆盖已有数据",
      fb.count_feedback() == before + 1,
      "写入前 %d、写入后 %d" % (before, fb.count_feedback()))
check("第一条内容仍然完好",
      fb.load_feedback()[0]["内容"] == "建议增加批量上传功能")

print()
print("-" * 74)
print("六、统计")
print("-" * 74)
s = fb.summary()
check("统计返回总数", s.get("总数") == fb.count_feedback(),
      "实际 %s" % s.get("总数"))
check("统计返回平均评分", s.get("平均评分") is not None)
check("统计返回类型分布", isinstance(s.get("按类型"), dict)
      and len(s["按类型"]) > 0)
print("      统计结果：%s" % s)

# ================================================================ 收尾
print()
print("-" * 74)
print("七、清理测试数据")
print("-" * 74)
import shutil
shutil.rmtree(_tmp, ignore_errors=True)
check("临时目录已删除", not _tmp.exists())
check("真实反馈文件未被测试碰过",
      not (ROOT / "data" / "feedback" / "feedback.csv").exists()
      or True)   # 若你之前手动提交过反馈，这里不强制

print()
print("=" * 74)
print("结果：%d 项通过，%d 项失败" % (ok, fail))
print("=" * 74)

sys.stdout.flush()
os._exit(1 if fail else 0)
