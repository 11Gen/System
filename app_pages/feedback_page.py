"""
帮助与反馈页。

放在导航最底部，是用户遇到问题时会来找的地方。
两部分：
    上半部  常见问题 —— 帮用户自己解决，减少无效反馈
    下半部  反馈表单 —— 写到本地 CSV

关于反馈存储：这是本地应用，反馈写到运行这个应用的电脑上。
页面上原本有一段说明向用户交代这一点，按要求删掉了 ——
所以现在界面上不体现"反馈收不回作者手里"这件事，
但功能没变，提交仍然是写本地文件。
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import feedback as fb
import ui_pages
from ui_table import html_table

st.title("帮助与反馈", icon=":material/help:")

tab_faq, tab_feedback = st.tabs(
    [":material/quiz: 常见问题", ":material/rate_review: 提交反馈"])


def safe_page_link(page: str, label: str) -> None:
    """
    加一个跳转链接，失败时降级成文字提示，不让页面崩掉。

    为什么需要包一层：st.page_link 要求目标页面**已经注册在
    st.navigation 里**。在正常运行（从 streamlit_app.py 进入）时没问题，
    但如果这个页面被单独运行 —— 例如用 Streamlit 的测试框架
    逐页跑冒烟测试 —— 注册表不存在，page_link 会抛
    StreamlitPageNotFoundError，整页渲染失败。

    这类"只有测试环境才触发"的崩溃不该存在：测试跑不过，
    就失去了发现真实问题的能力。所以这里容错处理。
    """
    try:
        st.page_link(page, label=label, icon=":material/arrow_forward:")
    except Exception:
        # 退化成一个不带链接的提示。用户可以自己从左侧导航过去。
        st.caption(":material/arrow_forward: %s（从左侧导航栏进入）" % label)


# ================================================================ 常见问题
with tab_faq:
    st.caption("先看看这里有没有答案。这些问题都是使用中实际会遇到的困惑，"
               "尤其是那些「看起来像 bug、其实是有意设计」的地方。")

    for i, item in enumerate(fb.FAQ, 1):
        with st.expander("%d. %s" % (i, item.question),
                         icon=":material/help_outline:"):
            st.markdown(item.answer)
            if item.link_page:
                safe_page_link(item.link_page, item.link_label)

    st.space("small")
    with st.container(border=True):
        st.markdown("**没找到答案？**")
        st.caption("切到右边的「提交反馈」标签页写下来。"
                   "如果是报告问题，把当时的操作和看到的提示一起写上，"
                   "会更容易定位。")


# ================================================================ 反馈表单
with tab_feedback:
    st.caption("这一页收集使用中的问题、建议和数据纠错。")

    # 这里原本有一段 st.info 说明，讲"这是本地应用、反馈只写到本机、
    # 作者收不到"。按要求删掉了。
    # 保留这条注释是为了说明：删掉的只是一句说明文字，功能没变 ——
    # 反馈仍然写进 data/feedback/feedback.csv。

    with st.form("feedback_form", clear_on_submit=False):
        c1, c2 = st.columns([1.4, 1])
        with c1:
            ftype = st.selectbox("反馈类型", fb.FEEDBACK_TYPES,
                                 help="选一个最接近的即可")
        with c2:
            rating = st.slider("整体评分", 1, 5, 4,
                               help="给这个系统打个分，1 分最差、5 分最好")

        content = st.text_area(
            "具体内容", height=140,
            placeholder="例如：用「识菜」页上传麻婆豆腐，"
                        "提示「不是系统支持的菜品」，但麻婆豆腐在支持列表里。",
            help="如果是报告问题，写上当时的操作步骤和看到的提示，"
                 "能大幅提高定位效率")

        contact = st.text_input(
            "联系方式（可选）",
            placeholder="邮箱或其他联系方式，不填也能提交",
            help="只在需要进一步沟通时才用得上，不填完全没问题")

        submitted = st.form_submit_button(
            ":material/send: 提交反馈", type="primary", width="stretch")

    if submitted:
        ok, msg = fb.validate(ftype, rating, content)
        if not ok:
            st.error(msg, icon=":material/error:")
        else:
            try:
                n = fb.append_feedback(ftype, rating, content, contact)
                st.success("已保存，这是第 **%d** 条反馈。" % n,
                           icon=":material/check_circle:")
                st.caption("文件位置：`%s`（可以用 Excel 直接打开查看）"
                           % fb.FEEDBACK_CSV)
            except Exception as e:
                st.error("保存失败：%s" % str(e)[:200],
                         icon=":material/error:")

    # 已有的反馈记录
    st.space("small")
    rows = fb.load_feedback()
    if rows:
        st.markdown("**已提交的反馈**")
        cols = ["提交时间", "反馈类型", "评分", "内容", "联系方式"]
        html_table(cols, [[r.get(c, "") for c in cols] for r in rows],
                   align_right=[False, False, True, False, False],
                   caption="共 %d 条。这些记录也可以直接用 Excel 打开 "
                           "`data/feedback/feedback.csv` 查看。"
                           % len(rows))
        with st.expander("统计", icon=":material/analytics:"):
            s = fb.summary()
            cc = st.columns(3)
            cc[0].metric("反馈总数", "%d 条" % s.get("总数", 0))
            cc[1].metric("平均评分",
                         "%.1f / 5" % s["平均评分"]
                         if s.get("平均评分") else "—")
            by_type = s.get("按类型") or {}
            cc[2].metric("类型数", "%d 类" % len(by_type))
            if by_type:
                st.caption("按类型分布：%s"
                           % "、".join("%s %d 条" % (k, v)
                                       for k, v in by_type.items()))
    else:
        st.caption("还没有提交过反馈。")

    # 清理入口。放在最后、用折叠收起来，
    # 避免用户误点把记录删掉。
    if rows:
        with st.expander("清空全部反馈记录", icon=":material/delete_forever:"):
            st.caption("删除后无法恢复。主要用于演示前清掉测试数据。")
            if st.button("确认清空 %d 条记录" % len(rows), type="secondary"):
                try:
                    if fb.FEEDBACK_CSV.exists():
                        fb.FEEDBACK_CSV.unlink()
                    st.success("已清空。", icon=":material/check_circle:")
                    st.rerun()
                except Exception as e:
                    st.error("删除失败：%s" % str(e)[:200])
