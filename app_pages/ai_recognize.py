"""
AI 识别页。

和「识菜」页的分工：
    识菜     —— 本地模型，只认识训练过的 16 道菜，毫秒级、免费、离线可用
    AI 识别  —— 调大模型视觉接口，理论上任何菜都能认，但慢、要联网、要花钱

为什么要做成两个页面而不是一个：
    两条路径的代价差别很大（延迟、成本、联网依赖），
    合并成一个会让用户不清楚自己在用什么。
    分开之后，「识菜」可以放心当主力（永远可用），
    而「AI 识别」明确是"认不出来时"的补充手段。

关于 API Key：
    只存在浏览器会话里（st.session_state），不写入任何文件。
    刷新页面或重启服务后需要重填 —— 这是有意的取舍，
    安全性优先于便利性。
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import nutrition as nut
from ai_nutrition import estimate_from_ingredients, to_estimate_dict
from vision_api import (DEFAULT_PROVIDER, PROVIDERS, DishResult,
                        VisionAPIError, VisionClient, match_known_dish)
from ui_table import html_table


# ---------------------------------------------------------------- 缓存资源
@st.cache_resource(show_spinner=False)
def get_nutrition_db():
    return nut.NutritionDB()


@st.cache_resource(show_spinner=False)
def get_ingredient_table():
    """配料营养表。AI 估算营养时要用，读一次缓存起来。"""
    from build_nutrition_db import load_ingredients
    return load_ingredients()


@st.cache_resource(show_spinner="正在加载本地模型…")
def get_local_predictor():
    """
    本地模型。用途是"AI 认出的菜如果在内置 16 道里，
    就直接用本地库里更权威的营养数据"。
    """
    from infer import Predictor
    return Predictor()


# ---------------------------------------------------------------- 会话状态
# 注意：API Key 只在这里存，不落盘。
def _init_state():
    ss = st.session_state
    ss.setdefault("ai_provider", DEFAULT_PROVIDER)
    ss.setdefault("ai_api_key", "")
    ss.setdefault("ai_base_url", PROVIDERS[DEFAULT_PROVIDER]["base_url"])
    ss.setdefault("ai_model", PROVIDERS[DEFAULT_PROVIDER]["model"])
    ss.setdefault("ai_connected", False)
    ss.setdefault("ai_last_result", None)
    # 「膳食管家」页读的是这个 key，AI 识别的结果也加进同一个餐
    ss.setdefault("meal_items", [])


_init_state()


# ---------------------------------------------------------------- 页面
st.title("AI 识别", icon=":material/auto_awesome:")

st.info(
    "这一页用大模型的视觉能力识别菜品，**不受内置 16 道菜的限制**。"
    "代价是需要联网、需要 API Key、每次识别有几秒延迟并产生少量费用。"
    "如果只是识别常见菜，用「识菜」页更快也更省。",
    icon=":material/info:")

db = get_nutrition_db()
ing_table = get_ingredient_table()
local_pred = get_local_predictor()


# ---------------------------------------------------------------- 连接设置
with st.container(border=True):
    st.subheader("AI 服务设置", icon=":material/settings:")

    c1, c2 = st.columns([1.2, 2])
    with c1:
        provider = st.selectbox(
            "服务商", list(PROVIDERS.keys()),
            index=list(PROVIDERS.keys()).index(st.session_state["ai_provider"]),
            key="ai_provider_widget")

    with c2:
        st.caption("两家的接口都是 OpenAI 兼容格式，用同一套代码调用。")

    # 切换服务商时自动填默认地址和模型名。
    # 用 session_state 记录上一次的选择来判断是否真的换了 ——
    # 直接比较 provider 和输入框的值会误判（用户手改过一次之后
    # 每次重跑都会把改动冲掉）。
    if provider != st.session_state.get("ai_provider_applied"):
        st.session_state["ai_base_url"] = PROVIDERS[provider]["base_url"]
        st.session_state["ai_model"] = PROVIDERS[provider]["model"]
        st.session_state["ai_provider_applied"] = provider
        st.session_state["ai_connected"] = False

    c3, c4 = st.columns(2)
    with c3:
        base_url = st.text_input(
            "接口地址 base_url", key="ai_base_url",
            help="已按服务商自动填好。如果厂家换了域名，直接在这里改。")
    with c4:
        model = st.text_input(
            "模型名称", key="ai_model",
            help="厂家更新模型名后，在这里改成新的即可，不用改代码。")

    api_key = st.text_input(
        "API Key", key="ai_api_key", type="password",
        placeholder=PROVIDERS[provider]["key_hint"],
        help="只保存在当前会话内存里，不写入任何文件。刷新页面需重填。")

    info = PROVIDERS[provider]
    if info.get("note"):
        st.caption("**%s**：%s" % (provider, info["note"]))
    tips = []
    if info.get("verified"):
        tips.append("默认配置核实于 %s" % info["verified"])
    if info.get("doc"):
        tips.append("官方文档：%s" % info["doc"])
    if tips:
        st.caption("　".join(tips))

    b1, b2 = st.columns([1, 4])
    if b1.button(":material/cable: 测试连接", type="primary"):
        try:
            VisionClient(api_key=api_key, base_url=base_url, model=model,
                         provider=provider)
            st.session_state["ai_connected"] = True
            st.session_state["ai_connected_cfg"] = (provider, base_url, model)
            st.success("配置已填好。下面上传图片试一次就能验证是否连通。",
                       icon=":material/check_circle:")
        except VisionAPIError as e:
            st.session_state["ai_connected"] = False
            st.error(str(e), icon=":material/error:")

    with b2:
        st.caption("点上面的按钮只是检查配置是否填全，"
                   "真正验证连通性要上传一张图片试一次。")


def make_client():
    return VisionClient(api_key=st.session_state["ai_api_key"],
                        base_url=st.session_state["ai_base_url"],
                        model=st.session_state["ai_model"],
                        provider=st.session_state["ai_provider"])


# ---------------------------------------------------------------- 输入图片
st.space("small")
st.subheader("上传图片", icon=":material/photo_camera:")

tab_upload, tab_camera, tab_sample = st.tabs(
    [":material/upload: 上传图片", ":material/photo_camera: 拍照",
     ":material/collections: 示例图片"])

image = None
with tab_upload:
    f = st.file_uploader("选择一张菜品照片",
                         type=["jpg", "jpeg", "png", "webp"],
                         label_visibility="collapsed", key="ai_upload")
    if f is not None:
        from PIL import Image
        image = Image.open(f)

with tab_camera:
    st.caption("用摄像头对准菜品拍照。")
    shot = st.camera_input("拍照", label_visibility="collapsed",
                           key="ai_camera")
    if shot is not None:
        from PIL import Image
        image = Image.open(shot)

with tab_sample:
    from infer import load_sample_images
    samples = load_sample_images(limit=8)
    if not samples:
        st.caption("测试集还没准备好。")
    else:
        st.caption("这些图取自测试集，模型训练时没见过。"
                   "可以拿它们对比「识菜」页和这一页的结果差异。")
        cols = st.columns(4)
        for i, s in enumerate(samples):
            with cols[i % 4]:
                st.image(str(s["path"]), caption=s["cn_name"], width="stretch")
                if st.button("用它试试", key="ai_sample_%d" % i):
                    st.session_state["ai_picked"] = str(s["path"])
        if st.session_state.get("ai_picked"):
            from PIL import Image
            image = Image.open(st.session_state["ai_picked"])

if image is None:
    st.caption("上传或拍一张照片，也可以从示例图片里挑一张。")
    st.stop()


# ---------------------------------------------------------------- 识别
st.space("small")
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.image(image, caption="待识别的图片", width="stretch")
    st.caption("图片尺寸 %d × %d（上传前会缩到长边 1024 以内）"
               % image.size)

with right:
    st.subheader("识别结果")

    if st.button(":material/auto_awesome: 开始识别", type="primary",
                 width="stretch"):
        try:
            client = make_client()
        except VisionAPIError as e:
            st.error(str(e), icon=":material/error:")
            st.stop()

        with st.spinner("正在调用 %s…" % st.session_state["ai_model"]):
            try:
                t0 = time.time()
                res = client.recognize(image)
                # 顺便记一下本地模型对同一张图的判断，
                # 用来对比两条路径的差异 —— 这个对比本身挺有意思
                local_top1 = None
                if local_pred.ready:
                    lr = local_pred.predict(image, topk=1)
                    local_top1 = lr["top1"]
                st.session_state["ai_last_result"] = (res, local_top1)
            except VisionAPIError as e:
                st.error(str(e), icon=":material/error:")
                st.stop()

    payload = st.session_state.get("ai_last_result")
    if payload:
        res, local_top1 = payload

        if not res.name:
            st.warning("模型没有按预期格式返回结果，请重试。",
                       icon=":material/warning:")
            with st.expander("查看模型原始输出", icon=":material/code:"):
                st.text(res.raw[:1500])
        else:
            if res.name == "非菜品":
                st.error("这张图看起来不是菜品。请换一张菜品照片。",
                         icon=":material/report:")
            else:
                conf_txt = ("%.0f%%" % (res.confidence * 100)
                            if res.confidence is not None else "未提供")
                st.success("识别结果：**%s**（AI 自报把握度 %s）"
                           % (res.name, conf_txt),
                           icon=":material/check_circle:")
                if res.cuisine or res.description:
                    st.caption("　".join(x for x in [res.cuisine,
                                                     res.description] if x))

            m1, m2, m3 = st.columns(3)
            m1.metric("调用耗时", "%.1f 秒" % res.elapsed)
            m2.metric("使用模型", res.model)
            m3.metric("本地模型判断",
                      local_top1["cn_name"] if local_top1 else "—")

            # 两条路径对比：这是这个页面独有的、也是挺有说服力的信息
            if local_top1:
                if local_top1["cn_name"] == res.name:
                    st.caption("两个模型判断一致，结果可信度较高。")
                else:
                    st.caption(
                        "本地模型认为是「%s」（把握度 %.0f%%），"
                        "与 AI 的判断不同。本地模型只认识内置的 %d 道菜，"
                        "遇到范围外的菜会强行归类，所以这里以 AI 的结果为准；"
                        "但如果本地模型的把握度很高，也值得留意。"
                        % (local_top1["cn_name"], local_top1["prob"] * 100,
                           len(local_pred.classes)))

    else:
        st.caption("点上面的按钮开始识别。")


# ---------------------------------------------------------------- 营养
st.space("small")
st.subheader("营养估算", icon=":material/monitor_heart:")

payload = st.session_state.get("ai_last_result")
if not payload or not payload[0].name or payload[0].name == "非菜品":
    st.caption("识别出菜品后，这里会显示营养估算。")
    st.stop()

res = payload[0]

# 先看内置营养库里有没有这道菜。
# 有的话直接用库里的数据 —— 那是权威来源，比 AI 估算准得多。
known_key = match_known_dish(res.name, list(db.keys()))
if known_key:
    st.success(
        "「%s」在内置的 16 道菜里，直接使用营养库中的权威数据"
        "（配方来自地方标准，逐条可溯源），不用 AI 估算。"
        % config.CN_NAME.get(known_key, known_key),
        icon=":material/verified:")
    est = nut.estimate_dish(known_key, db)
    st.session_state["ai_estimate"] = est
    st.session_state["ai_estimate_is_ai"] = False
else:
    st.warning(
        "「%s」不在内置的 16 道菜里，营养库里没有它的数据。"
        "下面**让 AI 给出配料和克重**，再用和内置菜完全相同的算法"
        "（配料营养值加权）算出营养。\n\n"
        "**注意：配料克重是 AI 估算的，不是权威数据，仅供参考。**"
        % res.name,
        icon=":material/warning:")

    if st.button(":material/calculate: 让 AI 估算营养", type="primary"):
        try:
            client = make_client()
            with st.spinner("正在让 %s 给出配料…" % st.session_state["ai_model"]):
                ings, total_g, raw = client.estimate_recipe(res.name)
        except VisionAPIError as e:
            st.error(str(e), icon=":material/error:")
            st.stop()

        if not ings:
            st.error("没能从 AI 的回答里解析出配料。", icon=":material/error:")
            with st.expander("查看原始输出", icon=":material/code:"):
                st.text(raw[:1500])
            st.stop()

        est_ai = estimate_from_ingredients(res.name, ings, total_g, ing_table)
        st.session_state["ai_estimate"] = to_estimate_dict(est_ai) \
            if est_ai.ok else None
        st.session_state["ai_estimate_raw"] = est_ai
        st.session_state["ai_estimate_is_ai"] = True

# 展示估算结果
est = st.session_state.get("ai_estimate")
is_ai = st.session_state.get("ai_estimate_is_ai", False)

if est is None and is_ai:
    raw_est = st.session_state.get("ai_estimate_raw")
    if raw_est and raw_est.message:
        st.error("营养估算失败：%s" % raw_est.message,
                 icon=":material/error:")
    st.caption("可以换一张更清晰的图片，或换一道更常见的菜再试。")
    st.stop()

if est:
    n = est["nutrients"]
    per100 = est["per_100g"]

    c = st.columns(4)
    c[0].metric("能量", "%.0f kcal" % n.energy_kcal)
    c[1].metric("蛋白质", "%.1f g" % n.protein_g)
    c[2].metric("脂肪", "%.1f g" % n.fat_g)
    c[3].metric("碳水化合物", "%.1f g" % n.carb_g)

    c2 = st.columns(3)
    c2[0].metric("膳食纤维", "%.1f g" % n.fiber_g)
    c2[1].metric("钠", "%.0f mg" % n.sodium_mg)
    c2[2].metric("估算份量", "%.0f g" % est["grams"])

    st.caption("每 100 g：%.0f kcal ｜ 蛋白 %.1f g ｜ 脂肪 %.1f g ｜ 碳水 %.1f g"
               % (per100.energy_kcal, per100.protein_g, per100.fat_g,
                  per100.carb_g))

    ratio = n.macro_energy_ratio()
    st.markdown("**三大营养素供能比**")
    st.bar_chart(pd.DataFrame({
        "营养素": ["蛋白质", "脂肪", "碳水化合物"],
        "供能占比": [ratio["protein"], ratio["fat"], ratio["carb"]],
    }), x="营养素", y="供能占比", height=200)

    # 数据来源说明 —— 这一块是刻意做详细的。
    # 让用户一眼看出哪些数字是权威来源、哪些是 AI 估的。
    with st.expander("这些数字是怎么来的", icon=":material/verified:",
                     expanded=is_ai):
        st.markdown("**数据来源**：%s" % est["source"])
        if is_ai:
            st.error(
                "这道菜的营养数据**全部来自 AI 估算**，"
                "与内置 16 道菜的数据不是一个可信等级。\n\n"
                "具体地说：配料克重是 AI 估的，营养值是从中国食物成分表"
                "查表加权算的。所以不确定性集中在「克重估得准不准」上。",
                icon=":material/warning:")
        if est.get("note"):
            st.markdown("**备注**：%s" % est["note"])
        if est.get("recipe"):
            st.markdown("**配料与克重**：")
            html_table(["配料", "克重(g)"],
                       [[k, "%g" % v] for k, v in est["recipe"].items()],
                       align_right=[False, True],
                       caption="这些克重由 AI 估算，不是权威数据")

    if st.button(":material/add_circle: 加入本餐", type="primary"):
        st.session_state["meal_items"].append(est)
        st.toast("已加入本餐：%s" % est["cn_name"],
                 icon=":material/check_circle:")
        st.caption("已加入 %d 道菜，去「膳食管家」页查看汇总。"
                   % len(st.session_state["meal_items"]))
