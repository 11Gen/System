"""
feedback.py —— 「帮助与反馈」页的内容与数据逻辑。

两部分：
    1. 常见问题（FAQ）的内容，写在这里而不是界面文件里，
       因为界面上还要配"跳转按钮"，数据和跳转目标放一起更好维护。
    2. 反馈的存储：写入 data/feedback/feedback.csv。

存储方式的实际情况（界面上不再体现，但代码事实如此）：
    这是个本地运行的应用，没有服务器后端。用户提交的反馈只能写到
    **运行这个应用的这台电脑的磁盘上**。如果别人在自己电脑上打开，
    他们提交的反馈写在他们自己的磁盘上，应用作者是收不到的。

    原先界面上有一段说明向用户交代这一点，后来按要求删掉了。
    之所以在这里记一笔：将来若要做成真正的在线反馈，
    需要补的是后端服务与域名，而不是改这个文件。
"""

import csv
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import ui_pages

FEEDBACK_DIR = config.DATA_DIR / "feedback"
FEEDBACK_CSV = FEEDBACK_DIR / "feedback.csv"

# CSV 的列。顺序即写入顺序，改这里要同时考虑已存在的文件 ——
# 读的时候按列名取值，所以加列是安全的，改列名会导致旧数据读不到。
FIELDS = ["提交时间", "反馈类型", "评分", "内容", "联系方式"]

FEEDBACK_TYPES = ["功能建议", "问题报告", "数据纠错", "其他"]


@dataclass
class FAQItem:
    """一条常见问题"""
    question: str
    answer: str
    # 可选的跳转目标：界面会在答案下面配一个按钮
    link_page: Optional[str] = None
    link_label: str = ""


# ---------------------------------------------------------------- FAQ 内容
# 挑选标准：优先收"用户会疑惑但界面上一句话说不清"的问题。
# 尤其是那些"看起来像 bug、其实是有意设计"的点 ——
# 比如拒识提示、份量要手动填、AI 数据标了仅供参考。
FAQ: List[FAQItem] = [
    FAQItem(
        question="为什么它说「这看起来不是系统支持的菜品」？",
        answer=(
            "说明这张图不属于系统内置的 16 道菜。系统的本地模型只学过这 16 道，"
            "遇到范围外的输入时，它会明确告诉你「不认识」，而不是硬猜一个答案给你。\n\n"
            "**这是有意设计的。** 一个只会强行归类的系统，给出的错误答案"
            "看起来和有把握的正确答案一模一样，用户没法分辨。"
            "所以宁可拒识，也不要给一个自信的错答案。\n\n"
            "阈值是标定过的：在保证约 95% 的正常菜品能被接受的前提下，"
            "能挡掉约一半的陌生输入。代价是约 5% 的正常菜品会被误拒 —— "
            "如果确定拍的是内置菜品，忽略提示即可。"
        ),
        link_page=ui_pages.MODEL_INFO,
        link_label="看拒识机制的评测数据",
    ),
    FAQItem(
        question="为什么换一道菜就认不出来了？",
        answer=(
            "因为本地模型是**闭集分类器**，只认识训练过的 16 道中国名菜。"
            "这是这类模型的固有特性，不是程序出错。\n\n"
            "系统为此准备了第二条路径：**「AI 识别」**。"
            "它调用大模型的视觉能力，理论上任何菜品都能认出来，"
            "并且同样会给营养估算。代价是需要联网、有几秒延迟，"
            "还要填一个 AI 服务的 API Key。"
        ),
        link_page=ui_pages.AI_RECOGNIZE,
        link_label="去 AI 识别",
    ),
    FAQItem(
        question="「识菜」和「AI 识别」该用哪个？",
        answer=(
            "**优先用「识菜」。** 它是本地模型，毫秒级响应、不需要联网、不花钱。\n\n"
            "只有在它提示「不认识」的时候，才切到「AI 识别」。\n\n"
            "| | 识菜 | AI 识别 |\n"
            "|---|---|---|\n"
            "| 能认的范围 | 内置 16 道菜 | 理论上不限 |\n"
            "| 速度 | 毫秒级 | 几秒 |\n"
            "| 联网 | 不需要 | 需要 |\n"
            "| 费用 | 0 | 每次约 0.0005 元 |\n\n"
            "另外，如果 AI 认出的菜恰好属于内置的 16 道，"
            "系统会自动改用营养库里的权威数据，不用 AI 估算。"
        ),
        link_page=ui_pages.RECOGNIZE,
        link_label="去识菜",
    ),
    FAQItem(
        question="算出来的热量准吗？",
        answer=(
            "**每 100 g 的营养值是可靠的**：全部来自公开的食物成分数据库"
            "（中国食物成分表、USDA FoodData Central），"
            "逐条标注了来源编号，可以点开「这个数字是怎么来的」查看，"
            "也可以用项目里的复核脚本重新抓取核对。\n\n"
            "**但份量是估算的**，这是整个链路里最大的误差来源 —— "
            "同样一盘菜，小碟装和大盘装能差一倍。"
            "所以界面上把份量做成了可调参数，而不是写死一个数字。"
            "准确程度取决于你填的份量有多接近实际。\n\n"
            "另外，不同餐厅同一道菜的做法差异很大（放多少油、用什么部位），"
            "所以任何营养估算都只能作为参考，不能当作医学或营养学建议。"
        ),
        link_page=ui_pages.MODEL_INFO,
        link_label="看营养数据的具体来源",
    ),
    FAQItem(
        question="份量为什么要我自己填？",
        answer=(
            "**因为仅凭一张照片无法可靠地推算重量。**\n\n"
            "要自动估计份量，至少需要两个条件之一："
            "画面里有已知尺寸的参照物（比如标准大小的餐具），"
            "或者有深度信息（比如双目摄像头、深度相机）。"
            "普通的单张手机照片两个都没有。\n\n"
            "有些方案会让模型「猜」一个份量出来，但那个数字没有依据，"
            "误差可能比用户自己估还大，而且用户看不出来它不准。"
            "所以这里选择让份量显式可控：**误差是可见、可调节的**，"
            "而不是被隐藏在一个看似精确的数字里。"
        ),
    ),
    FAQItem(
        question="AI 识别出来的营养数据，为什么标了「仅供参考」？",
        answer=(
            "因为它的可信度和内置菜品**不是一个等级**，必须区分开。\n\n"
            "内置的 16 道菜：配料配方来自地方标准和行业协会标准"
            "（如四川省地方标准 DB51/T 1728），每条都可溯源。\n\n"
            "AI 识别的新菜：**配料和克重是大模型估算的**，没有权威依据。"
            "不过营养值仍然是查中国食物成分表算出来的 —— "
            "也就是说，不确定性被限制在「克重估得准不准」这一个环节上。\n\n"
            "系统本可以让 AI 直接报一个热量数字，那样界面看起来更简洁。"
            "但那个数字无从核查，而且会和内置那批可溯源的数据混在一起，"
            "拉低整体的可信度。所以没有那样做。"
        ),
    ),
    FAQItem(
        question="需要联网吗？我的 API Key 安全吗？",
        answer=(
            "**「识菜」不需要联网**，模型和数据都在本地，断网也能完整演示。\n\n"
            "「AI 识别」需要联网，并且需要你填一个 AI 服务的 API Key。"
            "安全性方面：\n\n"
            "- Key **只保存在浏览器的会话内存里**，不写入任何配置文件\n"
            "- 刷新页面或重启服务后需要重新填写 —— 这是有意的取舍，"
            "安全性优先于便利性\n"
            "- Key 只会发送给你选择的那家 AI 服务，不经过任何第三方\n"
            "- 项目代码里没有任何记录或上传 Key 的逻辑，可以自行查阅"
            " `src/vision_api.py` 确认\n\n"
            "如果你在公用电脑上演示，演示完建议关闭浏览器页面。"
        ),
        link_page=ui_pages.AI_RECOGNIZE,
        link_label="去 AI 识别（需要填 Key）",
    ),
    FAQItem(
        question="「权威所在」那一页是干什么的？",
        answer=(
            "那一页把系统里**每个数字的来路都摊开**："
            "数据集怎么来的、模型怎么训的、结果怎么测的、营养数据出自哪里。\n\n"
            "如果你怀疑某个数字的真实性，去那一页看。"
            "页面顶部的几个指标都可以用仓库里的脚本重新跑一遍验证 —— "
            "运行 `python tools/verify_repro.py` 会重新加载模型、"
            "重新推理测试集、复算全部指标，并核对报告里每个数字的出处文件。\n\n"
            "简单说：**这一页是用来接受检验的。**"
        ),
        link_page=ui_pages.MODEL_INFO,
        link_label="去权威所在",
    ),
]


# ---------------------------------------------------------------- 存储
def ensure_dir() -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def _normalize(text: str) -> str:
    """
    清理用户输入。

    主要是两点：把连续空白压成一个（用户可能粘贴一大段带换行的文本），
    以及去掉首尾空格。不做更激进的过滤 —— 反馈是给人看的，
    保留原样更真实。
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    return text


def validate(feedback_type: str, rating: int, content: str):
    """
    校验表单。返回 (是否通过, 错误提示)。

    这些校验在界面层也会做（比如必填项），但这里再做一遍 ——
    界面校验可以被绕过（比如以后加了命令行入口），
    数据层自己把关更稳妥。
    """
    if feedback_type not in FEEDBACK_TYPES:
        return False, "请选择反馈类型。"
    if not isinstance(rating, int) or not (1 <= rating <= 5):
        return False, "评分需要在 1 到 5 之间。"
    c = _normalize(content)
    if len(c) < 4:
        return False, "请把问题或建议写得再具体一些（至少 4 个字）。"
    if len(c) > 2000:
        return False, "内容太长了（超过 2000 字），请精简一下。"
    return True, ""


def append_feedback(feedback_type: str, rating: int, content: str,
                    contact: str = "") -> int:
    """
    追加一条反馈，返回它现在是第几条（从 1 开始）。

    为什么用追加而不是重写整个文件：
        追加不会碰到已有数据。如果哪天写入过程中出错，
        最多是这一条没写进去，不会把之前的反馈全毁掉。
        而且用户能立刻看到"我是第 N 条"。
    """
    ensure_dir()
    ok, msg = validate(feedback_type, rating, content)
    if not ok:
        raise ValueError(msg)

    row = {
        "提交时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "反馈类型": feedback_type,
        "评分": int(rating),
        "内容": _normalize(content),
        "联系方式": _normalize(contact),
    }

    need_header = not FEEDBACK_CSV.exists() or FEEDBACK_CSV.stat().st_size == 0
    # 用 utf-8-sig：这样 Excel 直接打开不会中文乱码。
    # 代价是文件开头多了 BOM，Python 读的时候要用 utf-8-sig 解码，
    # 下面的 load_feedback 已经这么做了。
    with open(FEEDBACK_CSV, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if need_header:
            w.writeheader()
        w.writerow(row)

    return count_feedback()


def load_feedback() -> List[dict]:
    """读回全部反馈（主要用于自测和统计）"""
    if not FEEDBACK_CSV.exists():
        return []
    with open(FEEDBACK_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def count_feedback() -> int:
    return len(load_feedback())


def summary() -> dict:
    """反馈的简单统计（自测和自评时用）"""
    rows = load_feedback()
    if not rows:
        return {"总数": 0}
    by_type = {}
    ratings = []
    for r in rows:
        by_type[r.get("反馈类型", "未知")] = \
            by_type.get(r.get("反馈类型", "未知"), 0) + 1
        try:
            ratings.append(int(r.get("评分", 0)))
        except (TypeError, ValueError):
            pass
    return {
        "总数": len(rows),
        "按类型": by_type,
        "平均评分": round(sum(ratings) / len(ratings), 2) if ratings else None,
    }


if __name__ == "__main__":
    # 直接运行本文件时做一次自检
    print("=" * 70)
    print("反馈模块自检")
    print("=" * 70)
    print("FAQ 条目数：%d" % len(FAQ))
    for i, item in enumerate(FAQ, 1):
        link = item.link_page or "（无跳转）"
        print("  %d. %s  -> %s" % (i, item.question[:32], link))

    print()
    print("存储位置：%s" % FEEDBACK_CSV)
    print("当前已有反馈：%d 条" % count_feedback())
    s = summary()
    if s.get("总数"):
        print("统计：%s" % s)

    print()
    print("校验逻辑抽查：")
    for args, expect in [
        (("功能建议", 5, "这个系统很好用"), True),
        (("功能建议", 5, "好"), False),           # 太短
        (("不存在的类型", 5, "测试内容啊"), False),
        (("功能建议", 9, "测试内容啊"), False),    # 评分越界
        (("问题报告", 3, "   "), False),          # 空白
    ]:
        ok, msg = validate(*args)
        flag = "通过" if ok else "拦截"
        mark = "OK" if ok == expect else "!! 与预期不符"
        print("  %-40s -> %s  %s" % (str(args[0]) + "," + str(args[1]) + ","
                                     + repr(args[2])[:12], flag, mark))
