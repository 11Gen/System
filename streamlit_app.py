"""
中国名菜识别与营养摄入估算系统 —— 界面入口。

运行：
    streamlit run streamlit_app.py

这个文件只做三件事：设置页面、注册页面、画首页。
具体的页面内容在 app_pages/ 下面。
"""

import sys
from pathlib import Path

import streamlit as st

# 让 app_pages 和 src 下的模块都能 import。
# 不这么写的话，streamlit 从别的目录启动就会 ModuleNotFoundError。
ROOT = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import ui_pages
from infer import Predictor

st.set_page_config(
    page_title="中国名菜识别与营养估算",
    page_icon=":material/restaurant:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="正在加载识别模型…")
def get_predictor():
    """
    模型只加载一次，之后所有会话共用。

    用 cache_resource 而不是 cache_data：模型是带状态的对象，
    cache_data 会尝试序列化它，对 torch 模型既慢又容易出错。
    """
    return Predictor()


def home_page():
    st.title("中国名菜识别与营养摄入估算", icon=":material/restaurant:")
    st.caption("基于迁移学习的菜品图像识别系统 · 专业综合实训项目")

    predictor = get_predictor()

    # 四个指标卡只放用户看得懂、且关心的东西。
    # 原来后两个是"识别主干 MobileNetV2""营养来源 中国食物成分表" ——
    # 那是给评委看的，用户既看不懂也不关心：
    # 用户想知道的是"认得准不准""数据靠不靠得住"，
    # 而不是用什么网络、从哪个库取的。
    # 技术细节没有丢，都搬到「权威所在」页了。
    import json as _json

    _ev = None
    _evp = config.MODEL_DIR / "eval_summary.json"
    if _evp.exists():
        try:
            with open(_evp, encoding="utf-8") as _f:
                _ev = _json.load(_f)
        except Exception:
            _ev = None

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("支持菜品", "%d 道" % len(predictor.classes))
    col_b.metric("菜系覆盖", "%d 大菜系"
                 % len({v[1] for v in config.CLASS_INFO.values()}))
    if _ev:
        col_c.metric("识别准确率", "%.1f%%" % (_ev["accuracy"] * 100),
                     help="在 %d 张模型没见过的测试图上的 Top-1 准确率"
                          % _ev["n_samples"])
    else:
        col_c.metric("识别准确率", "—")
    col_d.metric("营养数据", "%d 道菜可查" % len(predictor.classes),
                 help="每道菜的热量与三大营养素，逐条标注了数据来源")

    st.space("medium")

    left, right = st.columns([1.15, 1])

    with left:
        with st.container(border=True):
            st.subheader("这个系统做什么", icon=":material/help:")
            st.markdown(
                "上传一张菜品照片，系统会完成三件事：\n\n"
                "1. **识别菜名** —— 判断这是哪道中国名菜，并给出置信度\n"
                "2. **估算营养** —— 按份量换算出热量与三大营养素\n"
                "3. **膳食分析** —— 把几道菜汇成一餐，对照《中国居民膳食指南》"
                "给出供能比和用盐提示"
            )

        # 这里原来放的是"技术上有什么不一样"（两阶段迁移学习、HOG 基线、
        # 数据可追溯三条）。那些是给评委看的，用户打开应用时关心的是
        # "我该干嘛"，所以换成了操作步骤。
        # 技术内容没有丢：两阶段训练在「权威所在」页的训练配置表里，
        # 基线对比和消融实验在那一页的实验部分，
        # 数据可追溯性在那一页顶部那句"可独立复算"里。
        with st.container(border=True):
            st.subheader("怎么用", icon=":material/lightbulb:")
            st.markdown(
                "1. 拍一张**菜品照片**（也可以先从示例图片里挑一张试试）\n"
                "2. 看识别结果，以及这盘菜的**热量和营养**\n"
                # 注意这里的写法：「**膳食管家**」而不是 **「膳食管家」**。
                # Markdown 的强调标记有个规则：闭标记 ** 的前一个字符
                # 不能是标点。中文标点（「」（）《》、。等）同样算标点，
                # 所以 **「膳食管家」** 会失效、把星号原样显示出来。
                # 把标点移到 ** 外面就正常了。
                "3. 把几道菜加进「**膳食管家**」，看看这一餐吃得合不合理"
            )
            # width="stretch" 让链接铺满卡片宽度 —— 默认的 "content"
            # 只按文字宽度画按钮，放在有边框的卡片里会显得缩着。
            # 注意别用 use_container_width，那个参数已废弃。
            st.page_link("app_pages/recognize.py", label="现在去识菜",
                         icon=":material/arrow_forward:", width="stretch")

        # AI 识别入口。单独一张卡而不是塞进上面那张，
        # 因为它和「识菜」是并列的两条路，而且代价不同
        # （要联网、要 API Key、有延迟），需要用户明确选择。
        with st.container(border=True):
            st.subheader("认不出别的菜？", icon=":material/auto_awesome:")
            st.markdown(
                "上面的「识菜」只认识内置的 %d 道中国名菜，"
                "遇到别的菜它会说「不认识」。\n\n"
                "「**AI 识别**」调用大模型的视觉能力，"
                "理论上任何菜品都能认出名字，还能估算它的营养。"
                "需要你自己填一个 AI 服务的 API Key。"
                % len(predictor.classes)
            )
            st.page_link("app_pages/ai_recognize.py", label="去 AI 识别",
                         icon=":material/arrow_forward:", width="stretch")

    with right:
        with st.container(border=True):
            st.subheader("支持的菜品", icon=":material/menu_book:")
            # 按菜系分组展示，比平铺 16 行好看也更好找
            by_cuisine = {}
            for key in predictor.classes:
                cn, cuisine, desc = config.CLASS_INFO.get(key, (key, "", ""))
                by_cuisine.setdefault(cuisine, []).append((cn, desc))
            for cuisine in sorted(by_cuisine):
                names = "、".join(n for n, _ in by_cuisine[cuisine])
                st.markdown("**%s**：%s" % (cuisine, names))

    st.space("medium")

    # 这里原来显示的是 "模型已就绪：best.pt"。
    # 问题在于 best.pt 是权重文件名，对用户是天书 ——
    # 用户看到会以为要自己去操作什么。
    # 而且上面第 73 行的指标卡已经显示了"支持菜品 16 道"，
    # 模型可用的信息已经传达到了，这里再说一遍没必要。
    #
    # 现在改成一句用户能看懂的话，并且说清"可以开始做什么"，
    # 而不只是"某件事完成了"。异常情况仍然走上面的红色报错 ——
    # 权重路径这类技术细节只在出错时才有必要露出来。
    if not predictor.ready:
        st.error(
            "模型还没准备好，识别功能暂时不可用。\n\n"
            "错误信息：`%s`\n\n"
            "请先在项目根目录运行训练：\n\n"
            "```bash\npython src/train_cnn.py\n```"
            % predictor.error, icon=":material/error:")
    else:
        st.success(
            "模型已加载，可以开始识别菜品。"
            "到左侧「识菜」页上传或拍一张照片即可。",
            icon=":material/check_circle:")

    with st.expander("使用须知与局限", icon=":material/warning:"):
        st.markdown(
            "- 本系统只支持上表中的 **%d 道菜**。喂给它别的菜（比如番茄炒蛋），"
            "它也会强行归到最像的一类里 —— 这是闭集分类器的固有局限，"
            "不是 bug。界面上会用置信度提示这种情况。\n"
            "- 营养数据来自公开食物成分数据库，**仅供参考**，"
            "不能作为医疗或营养学建议。\n"
            "- 份量是影响估算精度最大的因素。同样一盘菜，小碟和大盘能差一倍，"
            "所以界面上把份量做成了可调参数，不确定时建议按标准份估算。"
            % len(predictor.classes))


# 导航栏名称按"面向用户"的思路定，不用项目文档里的说法：
#   原来的"项目概览""模型与数据"是给评委看的词，用户看了不知道能干嘛。
#   文件名（app_pages/*.py）保持不动 —— 改名要同步改 git 历史和文档引用，
#   收益不大，容易漏。文件名和界面标签本来就不必一致。
#
# 页面路径统一走 ui_pages 里的常量，避免各处手写字符串 ——
# 手写的路径在改名后不会报错，只会静默失效。
pages = [
    st.Page(home_page, title="主页介绍", icon=":material/home:",
            default=True),
    st.Page(ui_pages.RECOGNIZE, title="识菜",
            icon=":material/photo_camera:"),
    # AI 识别紧随「识菜」之后 —— 它是识菜的补充手段：
    # 本地模型认不出内置 16 道菜之外的菜品时，走这条路。
    st.Page(ui_pages.AI_RECOGNIZE, title="AI 识别",
            icon=":material/auto_awesome:"),
    st.Page(ui_pages.MEAL_PLAN, title="膳食管家",
            icon=":material/restaurant_menu:"),
    # "权威所在"这个名字是刻意选的：这个页面要回答的是
    # "你这些数字是真的吗、凭什么信你"。比"模型与数据"更贴近
    # 用户带着怀疑点进来时想找的东西。
    # 图标配 verified（带勾的盾牌）——盾牌有"有背书、可信"的含义，
    # 比原来那个 source（三层方块）更贴语义。
    st.Page(ui_pages.MODEL_INFO, title="权威所在",
            icon=":material/verified:"),
    # 放在导航最底部：用户遇到问题时会往下找。
    # 图标用 help（问号）—— 和「识菜」「AI 识别」那些动作型图标区分开，
    # 表明这是一页辅助内容，不是功能入口。
    st.Page(ui_pages.FEEDBACK, title="帮助与反馈",
            icon=":material/help:"),
]

nav = st.navigation(pages)
nav.run()
