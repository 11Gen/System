"""
菜品识别页。

一页里完成"输入图片 -> 看识别结果 -> 调份量 -> 看营养估算"这条主线。
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
import nutrition as nut
from infer import load_sample_images
from ui_table import html_table


@st.cache_resource(show_spinner="正在加载识别模型…")
def get_predictor():
    from infer import Predictor
    return Predictor()


@st.cache_resource(show_spinner=False)
def get_nutrition_db():
    return nut.NutritionDB()


def confidence_chart(topk):
    """用条形图展示 top-k 概率。比只显示一个数字更能说明问题。"""
    df = pd.DataFrame({
        "菜品": ["%s %s" % (t["cn_name"], t["key"]) for t in topk],
        "概率": [t["prob"] for t in topk],
    })
    st.bar_chart(df, x="菜品", y="概率", horizontal=True, height=210)


def show_source_note(est):
    """把营养数据来源摆出来。这是这个项目和"随便出个数字"的区别所在。"""
    with st.expander("这个数字是怎么来的", icon=":material/verified:"):
        st.markdown("**数据来源**：%s" % est["source"])
        st.markdown("**份量口径**：%s（%.0f g）"
                    % (est["serving_desc"] or "—", est["grams"]))
        if est["note"]:
            st.markdown("**备注**：%s" % est["note"])
        if est["source_type"] == "D" and est["recipe"]:
            st.markdown("**配料配方**（按此配方与配料营养成分合成计算）：")
            # 用 HTML 表格而不是 st.dataframe —— 后者是 canvas 渲染，
            # 网格线颜色改不了，配米白背景几乎看不见（详见 src/ui_table.py）
            html_table(["配料", "重量(g)"],
                       [[k, "%g" % g] for k, g in est["recipe"].items()],
                       align_right=[False, True], zebra=True)
        st.caption(
            "所有营养数值均来自公开食物成分数据库，可用 "
            "`python src/check_nutrition.py` 重新抓取复核。")


st.title("菜品识别", icon=":material/photo_camera:")

predictor = get_predictor()
db = get_nutrition_db()

if not predictor.ready:
    st.error("模型尚未就绪：`%s`\n\n请先运行 `python src/train_cnn.py`"
             % predictor.error, icon=":material/error:")
    st.stop()

# ---------------------------------------------------------------- 输入
tab_upload, tab_camera, tab_sample = st.tabs(
    [":material/upload: 上传图片", ":material/photo_camera: 拍照", ":material/collections: 示例图片"])

image = None
with tab_upload:
    f = st.file_uploader("选择一张菜品照片", type=["jpg", "jpeg", "png", "webp"],
                         label_visibility="collapsed")
    if f is not None:
        from PIL import Image
        image = Image.open(f)

with tab_camera:
    st.caption("用摄像头对准菜品拍照。光线均匀、主体占画面大部分时效果最好。")
    shot = st.camera_input("拍照", label_visibility="collapsed")
    if shot is not None:
        from PIL import Image
        image = Image.open(shot)

with tab_sample:
    samples = load_sample_images(limit=8)
    if not samples:
        st.caption("测试集还没准备好，先运行 `python src/prepare_dataset.py --source dishes16`")
    else:
        st.caption("下面这些图取自测试集，模型训练时没有见过。")
        cols = st.columns(4)
        for i, s in enumerate(samples):
            with cols[i % 4]:
                st.image(str(s["path"]), caption=s["cn_name"], width="stretch")
                if st.button("用它试试", key="sample_%d" % i):
                    st.session_state["picked_sample"] = str(s["path"])
        if st.session_state.get("picked_sample"):
            from PIL import Image
            image = Image.open(st.session_state["picked_sample"])

# ---------------------------------------------------------------- 结果
if image is None:
    st.caption("上传或拍一张照片，也可以从示例图片里挑一张。")
    st.stop()

st.space("small")
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.image(image, caption="待识别的图片", width="stretch")
    st.caption("图片尺寸 %d × %d" % image.size)

with right:
    with st.spinner("识别中…"):
        result = predictor.predict(image, topk=5)

    top1 = result["top1"]
    st.subheader("识别结果")

    # 拒识判断：用能量分，不用最大置信度。
    # 为什么不用置信度：softmax 会把 logits 差距压平，实测已知样本的
    # 最大置信度中位数是 1.000，用 0.5 当阈值几乎不触发（这是早期版本的 bug）。
    # 能量分阈值由 src/open_set.py 在 527 张已知 + 596 张未知样本上标定，
    # 取"已知接受率 95%"那一档，对应未知拒识率约 47%。
    if result["uncertain"]:
        # 注意加粗的写法：标点放在 ** 外面（`**……**。` 而不是 `**……。**`）。
        # Markdown 的强调标记要求闭标记前一个字符不是标点，
        # 中文的 。！？」）都算标点，写成 **xxx。** 有把星号原样显示出来的风险。
        #
        # 提示文案只说"发生了什么 + 该怎么办"，不报能量分和阈值 ——
        # 那是内部实现，用户看不懂也不需要知道。
        # 想了解拒识原理的可以到「权威所在」页看，那里本来就要摊开讲。
        st.error(
            "**这看起来不是系统支持的菜品**。\n\n"
            "这个系统目前只认识 %d 道中国名菜。可能的原因："
            "① 拍的不是这 %d 道菜之一；"
            "② 照片角度或光线与训练数据差别较大。\n\n"
            "**建议换一张主体更清晰的照片重试**。"
            % (len(predictor.classes), len(predictor.classes)),
            icon=":material/report:")
    else:
        st.success("最有可能是 **%s**（%s，置信度 %.1f%%）"
                   % (top1["cn_name"], top1["cuisine"], top1["prob"] * 100),
                   icon=":material/check_circle:")

    st.caption("%s · %s" % (top1["key"], top1["desc"]))

    m1, m2, m3 = st.columns(3)
    m1.metric("类别置信度", "%.1f%%" % (top1["prob"] * 100))
    # 归一化熵：越接近 0 越确定，越接近 1 越犹豫
    m2.metric("不确定度", "%.2f" % result["entropy"],
              help="预测分布的归一化熵，0 表示完全确定，1 表示完全随机")
    m3.metric("能量分", "%.2f" % result["energy"],
              help="logsumexp(logits)，越高越像模型学过的菜。"
                   "低于 %.2f 判为未知类别" % result["threshold"])

    st.markdown("**候选菜品概率**")
    confidence_chart(result["topk"])

    # 这里原来还有一个「拒识机制是怎么定的」展开框，里面写了标定脚本路径
    # 和结果文件路径。那是给开发者和答辩用的，不该出现在用户界面上 ——
    # 用户看不懂 src/open_set.py 是什么，也不需要知道。
    # 拒识原理的说明已经放在「权威所在」页，那一页本来就是要摊开讲这些的。

# ---------------------------------------------------------------- 营养估算
st.space("small")
st.subheader("营养摄入估算", icon=":material/monitor_heart:")

# 默认用识别出来的那一类查营养库
default_key = top1["key"]
keys = [k for k in predictor.classes if db.get(k)]
if default_key not in keys and keys:
    default_key = keys[0]

if not keys:
    st.warning("营养库里还没有这些菜的记录，请先运行 "
               "`python src/build_nutrition_db.py`", icon=":material/warning:")
    st.stop()

# 允许用户改菜品：识别错了可以手动纠正。
# 这不是"掩盖错误"，而是承认视觉识别会错，并给出补救途径。
# 报告里会把"识别错误 + 人工纠正"当成一个正常的交互流程来讨论。
dish_key = st.selectbox(
    "菜品", options=keys, index=keys.index(default_key) if default_key in keys else 0,
    format_func=lambda k: "%s（%s）" % (config.CN_NAME.get(k, k), k),
    help="如果识别结果不对，可以在这里手动改成正确的菜")

sel = db.get(dish_key)
# 重量上限按"该菜一份重量"动态计算，不写死。
# 起因：写死 2000 g 时，北京烤鸭（官方标准一份 3130 g）的默认值
# 超过了上限，number_input 直接抛异常，整页崩溃。
# 现在至少留 3000 g 余量，且允许调到一份重量的 2 倍。
max_grams = max(3000.0, sel.serving_g * 2.0)

c1, c2 = st.columns([2, 1])
with c1:
    preset = st.segmented_control(
        "份量", options=list(nut.PORTION_PRESETS.keys()),
        default="标准份", key="portion_preset",
        help="标准份 = 营养库里记录的一份重量：%s（%.0f g）"
             % (sel.serving_desc or "—", sel.serving_g))
with c2:
    grams = st.number_input(
        "或直接指定重量 (g)", min_value=10.0, max_value=max_grams,
        value=float(sel.serving_g), step=10.0, key="grams_input",
        help="上限按这道菜一份的重量放宽到 %.0f g" % max_grams)

# 用户改了重量数字就以数字为准，否则按档位换算。
# 这两个控件会互相打架，所以明确一个优先级：数字框优先。
if grams != float(sel.serving_g):
    est = nut.estimate_dish(dish_key, db, portion_g=grams)
else:
    est = nut.estimate_dish(dish_key, db,
                            portion_scale=nut.PORTION_PRESETS[preset or "标准份"])

# 识别给出的类别必须在营养库里有记录，否则 estimate_dish 返回 None。
# 不判空的话下一行取 est["nutrients"] 就直接崩，而且崩得没有提示。
if est is None:
    st.error("营养库里没有「%s」这条记录，无法估算营养。"
             "识别结果仍然有效，只是缺这道菜的营养数据。"
             % config.CN_NAME.get(dish_key, dish_key),
             icon=":material/error:")
    st.stop()

n = est["nutrients"]
score = nut.nutriscore_like(n)
ratio = n.macro_energy_ratio()

d1, d2, d3, d4 = st.columns(4)
d1.metric("能量", "%.0f kcal" % n.energy_kcal)
d2.metric("蛋白质", "%.1f g" % n.protein_g)
d3.metric("脂肪", "%.1f g" % n.fat_g)
d4.metric("碳水化合物", "%.1f g" % n.carb_g)

e1, e2, e3 = st.columns(3)
e1.metric("膳食纤维", "%.1f g" % n.fiber_g)
e2.metric("钠", "%.0f mg" % n.sodium_mg,
          help="每日建议不超过 2000 mg（约合食盐 5 g）")
e3.metric("膳食评分", "%.1f / 10" % score["score"], delta=score["grade"],
          delta_color="off")

st.caption("每 100 g：%.0f kcal ｜ 蛋白 %.1f g ｜ 脂肪 %.1f g ｜ 碳水 %.1f g"
           % (est["per_100g"].energy_kcal, est["per_100g"].protein_g,
              est["per_100g"].fat_g, est["per_100g"].carb_g))

# 供能比：比绝对克数更能说明这餐结构合不合理
st.markdown("**三大营养素供能比**")
ratio_df = pd.DataFrame({
    "营养素": ["蛋白质", "脂肪", "碳水化合物"],
    "供能占比": [ratio["protein"], ratio["fat"], ratio["carb"]],
})
st.bar_chart(ratio_df, x="营养素", y="供能占比", height=200)
st.caption("《中国居民膳食指南》建议：蛋白质 10%~20%、脂肪 20%~30%、"
           "碳水化合物 50%~65%。")

if score["reasons"]:
    st.markdown("**评分依据**：%s" % "；".join(score["reasons"]))
st.caption("说明：这里的「膳食评分」是借用 Nutri-Score 思路做的**简化指标**，"
           "用于横向比较不同菜品的相对优劣，并非欧盟官方的 Nutri-Score 算法。")

show_source_note(est)

# 把这次结果存进会话状态，供"一餐分析"页使用
if st.button(":material/add_circle: 加入本餐", type="primary"):
    st.session_state.setdefault("meal_items", []).append(est)
    st.toast("已加入本餐：%s（%.0f g）" % (est["cn_name"], est["grams"]),
             icon=":material/check_circle:")
    st.caption("已加入 %d 道菜，去「一餐分析」页查看汇总。"
               % len(st.session_state["meal_items"]))
