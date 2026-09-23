"""
check_markdown.py —— 查找界面里"会把星号原样显示出来"的粗体写法。

背景：
    界面上出现了 **「膳食管家」** 这样带星号的文字。
    Markdown 的强调标记有个规则：**闭标记 `**` 的前一个字符不能是标点**，
    而中文标点（。！？」）（）、等）同样算标点。
    所以 `**xxx。**` 这类写法有把星号显示出来的风险。

为什么不用 Markdown 解析器来判定：
    试过用 markdown-it-py 渲染来判断，但同一个字符串
    （`**「膳食管家」**`）在不同测试脚本里给出了**相反**结果 ——
    说明测试装置有没控制住的变量，很可能是 Streamlit 内部的
    markdown 配置与我手工构造的不一致。
    依赖一个自己都不稳定的判据去改代码，风险比问题本身还大。

    所以改用保守策略：凡是 `**` 包裹的内容以中文标点结尾的，一律改写法。
    这个写法在任何 Markdown 实现下都不是最稳妥的，
    而改成「标点放到 ** 外面」在 CommonMark 里是无条件安全的。

用法：
    python tools/check_markdown.py
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 出现在 ** 内容末尾有风险的标点
TAIL_PUNCT = "。，、；：！？）」』》〉】”’…—·"
HEAD_PUNCT = "（「『《〈【“‘"

TARGETS = [
    "streamlit_app.py",
    "app_pages/recognize.py",
    "app_pages/meal_plan.py",
    "app_pages/model_info.py",
]


def scan_string(s):
    """找出所有 **...** 片段，返回 (片段, 是否结尾有标点, 位置)"""
    out = []
    i = 0
    while True:
        a = s.find("**", i)
        if a < 0:
            break
        b = s.find("**", a + 2)
        if b < 0:
            break
        inner = s[a + 2:b]
        if inner:
            out.append((inner, inner[-1] in TAIL_PUNCT,
                        inner[0] in HEAD_PUNCT))
        i = b + 2
    return out


def main():
    risky = []
    total = 0
    for f in TARGETS:
        p = ROOT / f
        if not p.exists():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            for inner, tail_bad, head_bad in scan_string(node.value):
                total += 1
                if tail_bad:
                    risky.append((f, node.lineno, inner, head_bad))

    print("=" * 78)
    print("粗体写法风险扫描")
    print("=" * 78)
    print("扫描文件：%s" % "、".join(TARGETS))
    print("共检查 %d 处 **粗体** 写法" % total)
    print()
    if not risky:
        print("没有发现以中文标点结尾的粗体写法。")
        return 0

    print("发现 %d 处「** 内容以中文标点结尾」，有把星号显示出来的风险："
          % len(risky))
    print()
    for f, ln, inner, head_bad in risky:
        print("  %s:%d" % (f, ln))
        print("     现在：**%s**" % inner)
        print("     建议：把结尾标点移到 ** 外面")
        print()
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
