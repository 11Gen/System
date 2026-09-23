"""
一餐分析页。

把「菜品识别」页加进来的菜汇总成一餐，对照膳食指南给出评价，
再合成一份一天三餐的报告。
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
from ui_table import guess_right_align, html_table, ratio_bar_table


@st.cache_resource(show_spinner=False)
def get_nutrition_db():
    return nut.NutritionDB()


st.title("膳食管家", icon=":material/restaurant_menu:")
st.caption("把识别到的菜汇总成一顿饭，对照《中国居民膳食指南（2022）》"
           "给出供能比和用盐提示。")

db = get_nutrition_db()

# 会话状态必须在这里显式初始化。不初始化的话，
# 用户直接打开这一页（没先去识别页加菜）会 KeyError。
if "meal_items" not in st.session_state:
    st.session_state["meal_items"] = []
if "day_meals" not in st.session_state:
    st.session_state["day_meals"] = {}

meal_items = st.session_state["meal_items"]

# ---------------------------------------------------------------- 本餐
st.subheader("本餐菜品", icon=":material/list_alt:")

if not meal_items:
    st.info("还没有添加菜品。请到「菜品识别」页识别后点「加入本餐」，"
            "也可以直接在下面手动添加。", icon=":material/info:")

    with st.expander("手动添加菜品", icon=":material/add:"):
        keys = [k for k in db.keys() if k in config.CLASS_INFO]
        if keys:
            c1, c2, c3 = st.columns([2, 1, 1])
            k = c1.selectbox("菜品", keys,
                             format_func=lambda x: config.CN_NAME.get(x, x))
            # 上限不能拍脑袋定。北京烤鸭一份按官方标准是 3130 g（整鸭），
            # 之前写死 2000 导致默认值超过上限，页面直接报错。
            # 这里按"该菜一份重量的 1.5 倍"动态给上限，至少留出 3000 g 的余量。
            base_g = float(db.get(k).serving_g)
            max_g = max(3000.0, base_g * 1.5)
            g = c2.number_input("重量(g)", 10.0, max_g, base_g, 10.0,
                                key="man_g")
            if c3.button("添加", key="man_add"):
                est = nut.estimate_dish(k, db, portion_g=g)
                meal_items.append(est)
                st.rerun()
else:
    # 用 data_editor 让用户直接改重量和删除，比一堆小控件清爽。
    # num_rows="dynamic" 允许删行。
    rows = []
    for i, it in enumerate(meal_items):
        rows.append({
            "序号": i,
            "菜品": it["cn_name"],
            "重量(g)": it["grams"],
            "能量(kcal)": round(it["nutrients"].energy_kcal, 1),
            "蛋白质(g)": round(it["nutrients"].protein_g, 1),
            "脂肪(g)": round(it["nutrients"].fat_g, 1),
            "碳水(g)": round(it["nutrients"].carb_g, 1),
            "钠(mg)": round(it["nutrients"].sodium_mg),
        })
    df = pd.DataFrame(rows).set_index("序号")

    # 这个表格**必须保留 st.data_editor**：用户可以在这里直接改份量、
    # 删行，改完下面的指标会跟着重算。换成静态 HTML 表格就没法编辑了。
    # 代价是它的网格线仍然偏淡（canvas 渲染，见 src/ui_table.py 的说明），
    # 但"能改"比"边框清楚"重要得多。
    edited = st.data_editor(
        df, width="stretch", num_rows="dynamic", key="meal_editor",
        column_config={
            "重量(g)": st.column_config.NumberColumn(
                min_value=10.0, max_value=2000.0, step=10.0,
                help="改这里会立刻重新计算下面所有指标"),
        })

    # 根据编辑结果重建列表
    new_items = []
    changed = False
    for idx, row in edited.iterrows():
        if not isinstance(idx, (int, float)):
            continue                       # 新增行没有序号，忽略
        i = int(idx)
        if i >= len(meal_items):
            continue
        it = meal_items[i]
        if abs(row["重量(g)"] - it["grams"]) > 0.01:
            it = nut.estimate_dish(it["key"], db, portion_g=row["重量(g)"])
            changed = True
        new_items.append(it)

    if len(new_items) != len(meal_items):
        changed = True
    if changed:
        st.session_state["meal_items"] = new_items
        meal_items = new_items

    b1, b2 = st.columns([1, 5])
    if b1.button(":material/delete: 清空", type="secondary"):
        st.session_state["meal_items"] = []
        st.rerun()

    # 保存到某一天
    with b2:
        save_cols = st.columns([2, 2, 3, 2])
        day = save_cols[0].text_input("日期", value="今天", key="save_day")
        meal_name = save_cols[1].selectbox("餐次", list(nut.MEAL_SHARE.keys()),
                                           index=1, key="save_meal")
        if save_cols[3].button(":material/save: 存入当日", type="primary"):
            key = "%s-%s" % (day, meal_name)
            st.session_state["day_meals"][key] = nut.summarize_meal(
                meal_items, meal_name)
            st.toast("已保存 %s" % key, icon=":material/check_circle:")

# ---------------------------------------------------------------- 本餐汇总
if meal_items:
    st.space("small")
    st.subheader("本餐合计", icon=":material/calculate:")

    # 用最近一次保存的餐次来估算参考值；没保存过就按午餐
    meal_name = st.session_state.get("save_meal", "午餐")
    summary = nut.summarize_meal(meal_items, meal_name)
    t = summary["total"]
    ratio = summary["macro_ratio"]

    c = st.columns(4)
    c[0].metric("总能量", "%.0f kcal" % t.energy_kcal)
    c[1].metric("蛋白质", "%.1f g" % t.protein_g)
    c[2].metric("脂肪", "%.1f g" % t.fat_g)
    c[3].metric("碳水化合物", "%.1f g" % t.carb_g)

    c2 = st.columns(3)
    c2[0].metric("膳食纤维", "%.1f g" % t.fiber_g)
    c2[1].metric("钠", "%.0f mg" % t.sodium_mg)
    # 本餐能量占每日参考的比例
    share = nut.MEAL_SHARE.get(meal_name, 0.33)
    c2[2].metric("占每日能量参考",
                 "%.0f%%" % (t.energy_kcal / nut.DAILY_REFERENCE["energy_kcal"] * 100),
                 help="本餐按「%s」占每日 %d%% 计" % (meal_name, share * 100))

    st.markdown("**本餐三大营养素供能比**")
    st.bar_chart(pd.DataFrame({
        "营养素": ["蛋白质", "脂肪", "碳水化合物"],
        "供能占比": [ratio["protein"], ratio["fat"], ratio["carb"]],
    }), x="营养素", y="供能占比", height=200)

    if summary["advice"]:
        st.markdown("**膳食提示**")
        for a in summary["advice"]:
            st.markdown("- %s" % a)
    else:
        st.success("三大营养素供能比落在膳食指南推荐区间内，钠也未超标。",
                   icon=":material/thumb_up:")

    with st.expander("成分明细（每道菜）", icon=":material/table_chart:"):
        detail = []
        for it in meal_items:
            detail.append({
                "菜品": it["cn_name"],
                "重量(g)": it["grams"],
                "能量(kcal)": round(it["nutrients"].energy_kcal, 1),
                "蛋白(g)": round(it["nutrients"].protein_g, 2),
                "脂肪(g)": round(it["nutrients"].fat_g, 2),
                "碳水(g)": round(it["nutrients"].carb_g, 2),
                "纤维(g)": round(it["nutrients"].fiber_g, 2),
                "钠(mg)": round(it["nutrients"].sodium_mg, 1),
                "来源": it["source"].split("｜")[0],
            })
        # 只读汇总，换 HTML 表格让边框看得见（st.dataframe 是 canvas）
        cols = ["菜品", "重量(g)", "能量(kcal)", "蛋白(g)", "脂肪(g)",
                "碳水(g)", "纤维(g)", "钠(mg)", "来源"]
        html_table(cols, [[r[c] for c in cols] for r in detail],
                   align_right=guess_right_align(cols))

# ---------------------------------------------------------------- 全天
st.space("medium")
st.subheader("全天汇总", icon=":material/calendar_today:")

day_meals = st.session_state["day_meals"]
if not day_meals:
    st.caption("还没有保存任何餐次。在上面的表格里调好份量后点「存入当日」。")
else:
    report = nut.daily_report(day_meals)
    t = report["total"]
    pct = report["percent_of_reference"]

    cols = st.columns(4)
    cols[0].metric("全天能量", "%.0f kcal" % t.energy_kcal,
                   "%.0f%% 参考值" % (pct["energy_kcal"] * 100))
    cols[1].metric("蛋白质", "%.1f g" % t.protein_g,
                   "%.0f%% 参考值" % (pct["protein_g"] * 100))
    cols[2].metric("脂肪", "%.1f g" % t.fat_g,
                   "%.0f%% 参考值" % (pct["fat_g"] * 100))
    cols[3].metric("钠", "%.0f mg" % t.sodium_mg,
                   "%.0f%% 参考值" % (pct["sodium_mg"] * 100))

    st.markdown("**各餐次能量分布**")
    st.bar_chart(pd.DataFrame({
        "餐次": list(day_meals.keys()),
        "能量(kcal)": [round(m["total"].energy_kcal) for m in day_meals.values()],
    }), x="餐次", y="能量(kcal)", height=230)

    st.markdown("**各项营养素占每日参考摄入量的比例**")
    ref_rows = []
    label_map = {
        "energy_kcal": "能量", "protein_g": "蛋白质", "fat_g": "脂肪",
        "carb_g": "碳水化合物", "fiber_g": "膳食纤维", "sodium_mg": "钠",
    }
    for k, lab in label_map.items():
        ref_rows.append((lab, pct[k]))
    # 用自带的条形图表格：既保留 ProgressColumn 那种"一眼看出超标"的效果，
    # 又有清晰边框。用 st.dataframe 的 ProgressColumn 做不到后者（canvas 渲染）。
    ratio_bar_table([r[0] for r in ref_rows], [r[1] for r in ref_rows],
                    max_value=2.0, warn_over=1.0)

    st.caption(
        "参考值取自《中国居民膳食指南（2022）》成人一般水平：能量 2000 kcal、"
        "蛋白质 60 g、脂肪 60 g、碳水 270 g、膳食纤维 25 g、钠 2000 mg。"
        "实际需求随年龄、性别、体力活动差异很大，此处仅作粗略对照。")

    over = [k for k, v in pct.items() if k == "sodium_mg" and v > 1.0]
    if over:
        st.warning("全天钠摄入已达参考值的 %.0f%%，建议减少咸菜、"
                   "酱料和加工食品。" % (pct["sodium_mg"] * 100),
                   icon=":material/warning:")

    with st.expander("已保存的餐次明细", icon=":material/receipt_long:"):
        saved = []
        for k, m in day_meals.items():
            saved.append({
                "餐次": k, "菜品数": m["n_dishes"],
                "能量(kcal)": round(m["total"].energy_kcal),
                "蛋白(g)": round(m["total"].protein_g, 1),
                "脂肪(g)": round(m["total"].fat_g, 1),
                "钠(mg)": round(m["total"].sodium_mg),
            })
        cols = ["餐次", "菜品数", "能量(kcal)", "蛋白(g)", "脂肪(g)", "钠(mg)"]
        html_table(cols, [[r[c] for c in cols] for r in saved],
                   align_right=guess_right_align(cols))
        if st.button(":material/delete_sweep: 清空全天记录"):
            st.session_state["day_meals"] = {}
            st.rerun()
