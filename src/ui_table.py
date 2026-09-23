"""
ui_table.py —— 带清晰边框的静态表格。

为什么不用 st.dataframe：
    st.dataframe 底层是 glide-data-grid，用 **canvas 渲染** ——
    单元格不是 DOM 节点，所以"给 td 加边框"这类 CSS 完全无效。
    而它的网格线颜色写死在前端主题里（约 #e6e6e6），
    配本项目这套米白背景几乎看不见。Streamlit 也没提供任何
    控制表格网格线的参数（theme 配置和 column_config 里都没有）。

    试过的两个方案都不可行：
      - 覆盖 --gdg-* CSS 变量：这些变量是前端从 theme 对象生成后
        以内联样式写上去的，内联优先于外部样式表，覆盖不掉。
      - 换 st.table：边框是清楚了，但它底层还是同一套渲染，且失去排序。

所以对**行数不多的汇总表**，直接自己用 HTML 画。
好处是边框完全可控、不依赖 Streamlit 内部实现（升级不会失效）；
代价是没有排序和滚动 —— 但这些表最多十几行，用不上那些功能。

**大表格仍然用 st.dataframe**，因为排序和滚动对它们有实际价值。
详见各页面的调用处。
"""

from typing import Iterable, List, Optional, Sequence

import streamlit as st

# 边框颜色。比 Streamlit 默认（约 #e6e6e6）深两级，
# 在米白背景上看得清，又不会像 Excel 那样死板。
BORDER = "#b9b2a6"
HEADER_BG = "#ece6dc"
ZEBRA_BG = "#faf7f2"


def html_table(headers: Sequence[str], rows: Iterable[Sequence],
               caption: Optional[str] = None,
               align_right: Optional[Sequence[bool]] = None,
               zebra: bool = True,
               font_size: float = 0.88) -> None:
    """
    画一个带清晰边框的静态表格。

    headers    : 表头文字
    rows       : 数据行，每行长度应与 headers 一致
    caption    : 表格上方的小字说明
    align_right: 每列是否右对齐（数字列右对齐更好读），默认全部左对齐
    zebra      : 是否隔行浅底色。行多时更容易跟行，行少时反而花，可关掉
    font_size  : 相对字号，默认比正文略小一点
    """
    headers = list(headers)
    rows = [list(r) for r in rows]
    n = len(headers)
    if align_right is None:
        align_right = [False] * n

    if caption:
        st.caption(caption)

    if not rows:
        st.caption("（没有数据）")
        return

    # 拼 CSS。**不用 % 格式化** —— CSS 里到处是 `100%;` 这种写法，
    # % 会被当成格式化占位符而报错（本项目在别处已经踩过一次）。
    # 用 f-string 或直接拼接就没有这个隐患。
    zebra_css = ("table.dish-tbl tbody tr:nth-child(even) td "
                 "{ background: " + ZEBRA_BG + "; }") if zebra else ""

    css = (
        "<style>"
        ".dish-tbl-wrap { overflow-x: auto; margin: 2px 0 10px 0; }"
        "table.dish-tbl { border-collapse: collapse; width: 100%;"
        " font-size: " + ("%.2f" % font_size) + "em; line-height: 1.5; }"
        "table.dish-tbl th, table.dish-tbl td {"
        " border: 1px solid " + BORDER + ";"
        " padding: 5px 9px; text-align: left; vertical-align: top;"
        " white-space: nowrap; }"
        "table.dish-tbl th { background: " + HEADER_BG + ";"
        " font-weight: 600; }"
        + zebra_css +
        "</style>"
    )

    def esc(v) -> str:
        s = "" if v is None else str(v)
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;"))

    parts = [css, '<div class="dish-tbl-wrap"><table class="dish-tbl">',
             "<thead><tr>"]
    for i, h in enumerate(headers):
        style = ' style="text-align:right"' if align_right[i] else ""
        parts.append("<th%s>%s</th>" % (style, esc(h)))
    parts.append("</tr></thead><tbody>")

    for row in rows:
        parts.append("<tr>")
        for i in range(n):
            v = row[i] if i < len(row) else ""
            style = ' style="text-align:right"' if align_right[i] else ""
            parts.append("<td%s>%s</td>" % (style, esc(v)))
        parts.append("</tr>")
    parts.append("</tbody></table></div>")

    st.html("".join(parts))


def guess_right_align(headers: Sequence[str]) -> List[bool]:
    """
    根据表头猜哪些列该右对齐。

    数字列右对齐更好读 —— 位数对齐了才方便比较大小。
    判断依据是表头里有没有单位或数值性词汇。
    """
    NUM_HINTS = ("(g)", "（g）", "(mg)", "（mg）", "(kcal)", "（kcal）",
                 "g)", "mg)", "kcal)", "(%)", "％", "率", "分", "数",
                 "重量", "克数", "热量", "能量")
    out = []
    for h in headers:
        h = str(h)
        out.append(any(k in h for k in NUM_HINTS))
    return out


def ratio_bar_table(labels: Sequence[str], values: Sequence[float],
                    max_value: float = 2.0,
                    number_format: str = "{:.0%}",
                    caption: Optional[str] = None,
                    warn_over: float = 1.0) -> None:
    """
    带条形图的比例表。

    为什么不用 st.dataframe 的 ProgressColumn：
    它是 canvas 渲染的，没有 DOM 单元格，边框改不了（见模块开头说明）。
    而单纯换成静态表格又会丢掉"一眼看出超标"的可视化效果，
    所以这里自己画 —— 用 div 的宽度当条形，超标的标红。

    labels    : 每行的名称
    values    : 对应的比例（1.0 表示正好达到参考值）
    max_value : 条形满格对应的比例
    warn_over : 超过这个比例就标红
    """
    if caption:
        st.caption(caption)

    css = (
        "<style>"
        ".ratio-tbl-wrap { overflow-x: auto; margin: 2px 0 10px 0; }"
        "table.ratio-tbl { border-collapse: collapse; width: 100%;"
        " font-size: 0.9em; }"
        "table.ratio-tbl th, table.ratio-tbl td {"
        " border: 1px solid " + BORDER + "; padding: 5px 9px;"
        " text-align: left; }"
        "table.ratio-tbl th { background: " + HEADER_BG + ";"
        " font-weight: 600; }"
        "table.ratio-tbl td.num { text-align: right; }"
        ".ratio-bar-bg { background: #eee8dd; border-radius: 3px;"
        " height: 13px; width: 130px; display: inline-block;"
        " vertical-align: middle; overflow: hidden; }"
        ".ratio-bar { height: 13px; display: block;"
        " background: #7fa650; }"
        ".ratio-bar.over { background: #c0392b; }"
        "</style>"
    )

    def esc(v) -> str:
        s = "" if v is None else str(v)
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;"))

    parts = [css, '<div class="ratio-tbl-wrap"><table class="ratio-tbl">',
             "<thead><tr><th>项目</th><th>摄入量</th>"
             "<th>占参考值</th><th style='width:150px'>水平</th></tr></thead>",
             "<tbody>"]
    for lab, v in zip(labels, values):
        pct_w = max(0.0, min(1.0, (v / max_value) if max_value else 0.0)) * 100
        over = " over" if v > warn_over else ""
        parts.append(
            "<tr><td>%s</td><td class='num'>%s</td><td class='num'>%s</td>"
            "<td><span class='ratio-bar-bg'>"
            "<span class='ratio-bar%s' style='width:%.1f%%'></span>"
            "</span></td></tr>"
            % (esc(lab), number_format.format(v),
               number_format.format(v), over, pct_w))
    parts.append("</tbody></table></div>")
    st.html("".join(parts))
