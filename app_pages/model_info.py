"""
模型与数据页。

把训练配置、数据集构成、评估结果、营养数据来源都摊开给答辩老师看。
这一页是"可复现性"的体现：所有数字都能在代码和文件里对上。
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from ui_table import html_table

st.title("权威所在", icon=":material/verified:")
st.caption("这一页把系统里每个数字的来路都摊开：数据集怎么来的、模型怎么训的、"
           "结果怎么测的、营养数据出自哪里。"
           "所有指标都可以用仓库里的脚本重新跑一遍验证。")

# ---------------------------------------------------------------- 快速自查
# 把"可核查"这件事放在最前面。别人点进这一页多半是带着疑问来的，
# 所以第一眼就应该看到最硬的证据，而不是先读一堆背景介绍。
_ev = config.MODEL_DIR / "eval_summary.json"
_os = config.MODEL_DIR / "open_set.json"
if _ev.exists():
    import json as _json
    with open(_ev, encoding="utf-8") as _f:
        _e = _json.load(_f)
    c = st.columns(4)
    c[0].metric("测试集准确率", "%.2f%%" % (_e["accuracy"] * 100))
    c[1].metric("测试样本数", "%d 张" % _e["n_samples"])
    c[2].metric("Top-3 准确率", "%.2f%%" % (_e["top3_accuracy"] * 100))
    if _os.exists():
        with open(_os, encoding="utf-8") as _f:
            _o = _json.load(_f)
        _best = max(_o["methods"], key=lambda m: m["auroc"])
        c[3].metric("拒识 AUROC", "%.4f" % _best["auroc"])
    st.caption("以上数字可用 `python tools/verify_repro.py` 独立复算，"
               "该脚本会重新加载模型、重新推理测试集，逐项核对。")

tab_data, tab_model, tab_nutrition = st.tabs(
    [":material/dataset: 数据集", ":material/model_training: 模型与训练",
     ":material/verified: 营养数据来源"])

# ---------------------------------------------------------------- 数据集
with tab_data:
    st.subheader("数据集构成")

    split_dir = config.SPLIT_DIR
    classes = []
    cp = split_dir / "classes.json"
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            classes = json.load(f)

    if not classes:
        st.caption("数据还没准备好。请先运行 `python src/prepare_dataset.py "
                   "--source dishes16` 和 `python src/split_dataset.py`")
    else:
        rows = []
        total = {"train": 0, "val": 0, "test": 0}
        for i, cls in enumerate(classes):
            cn, cuisine, _ = config.CLASS_INFO.get(cls, (cls, "", ""))
            r = {"序号": i, "菜名": cn, "英文标识": cls, "菜系": cuisine}
            for split in ("train", "val", "test"):
                d = split_dir / split / cls
                n = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.jpeg"))) \
                    if d.is_dir() else 0
                r[{"train": "训练", "val": "验证", "test": "测试"}[split]] = n
                total[split] += n
            rows.append(r)
        # 16 行 × 7 列，只读 —— 换 HTML 表格让边框可见
        cols = ["序号", "菜名", "英文标识", "菜系", "训练", "验证", "测试"]
        html_table(cols, [[r[c] for c in cols] for r in rows],
                   align_right=[True, False, False, False, True, True, True])

        c = st.columns(4)
        c[0].metric("菜品类别", "%d 道" % len(classes))
        c[1].metric("训练集", "%d 张" % total["train"])
        c[2].metric("验证集", "%d 张" % total["val"])
        c[3].metric("测试集", "%d 张" % total["test"])

        st.caption("按 8:1:1 分层划分，保证每类在三份里的比例一致。"
                   "随机猜测基线为 %.2f%%。" % (100.0 / len(classes)))

        st.markdown("**各菜系样本数**")
        by_cuisine = {}
        for cls in classes:
            cuisine = config.CLASS_INFO.get(cls, ("", "未知", ""))[1]
            by_cuisine[cuisine] = by_cuisine.get(cuisine, 0) + 1
        st.bar_chart(pd.DataFrame({
            "菜系": list(by_cuisine.keys()),
            "菜品数": list(by_cuisine.values()),
        }), x="菜系", y="菜品数", height=220)

        with st.expander("数据处理流程", icon=":material/account_tree:"):
            st.markdown(
                "原始数据集是 **YOLO 目标检测格式**（5806 张图 + 5793 个标注文件），"
                "不能直接当分类数据用。处理步骤：\n\n"
                "1. **按标注框筛选**：只保留画面里恰好有 1 道菜的图。"
                "有多个框说明这是一桌菜的照片，标签会有歧义，剔除 387 张\n"
                "2. **按框裁剪**：去掉菜品周围多余的桌面、餐具，"
                "让模型专注于菜本身\n"
                "3. **近纯色图过滤**：剔除下载失败留下的占位图\n"
                "4. **去重**：用文件 MD5 指纹去掉重复样本，"
                "避免同一张图同时出现在训练集和测试集里导致指标虚高\n"
                "5. **分层划分**：8:1:1 分成训练/验证/测试\n\n"
                "清洗后保留 **%d 张**，剔除率约 %.1f%%。"
                % (sum(total.values()),
                   100.0 * (1 - sum(total.values()) / 5806.0)))

# ---------------------------------------------------------------- 模型
with tab_model:
    st.subheader("模型结构")

    c = st.columns(3)
    c[0].metric("主干网络", "MobileNetV2")
    c[1].metric("预训练权重", "ImageNet-1K")
    c[2].metric("输入尺寸", "%d × %d" % (config.IMG_SIZE, config.IMG_SIZE))

    st.markdown("""
**分类头结构**：`Dropout(0.2)` → `Linear(1280, 16)`

主干沿用 MobileNetV2 在 ImageNet 上的预训练权重，把原本的
`Linear(1280, 1000)` 换成 16 类输出。因为本项目的训练集只有四千多张，
相比 ImageNet 的百万级小两个数量级，所以在分类头前加了 Dropout 抑制过拟合。
""")

    st.markdown("**两阶段迁移学习**")
    # 这几个参数量是训练时实测打印出来的，不是估算：
    # 阶段一 0.02M / 2.24M；阶段二解冻 features[14:] 后是 1.70M / 2.24M。
    # MobileNetV2 后面的 block 通道数多，参数量集中在那里，
    # 所以只解冻最后 5 个 block 就占到了七成以上参数。
    stage_df = pd.DataFrame([
        {"阶段": "阶段一", "策略": "冻结主干，只训练分类头",
         "可训练参数": "0.02M / 2.24M（0.9%）",
         "学习率": config.STAGE1_LR, "轮数": config.STAGE1_EPOCHS},
        {"阶段": "阶段二", "策略": "解冻 features[14:] 微调",
         "可训练参数": "1.70M / 2.24M（75.8%）",
         "学习率": config.STAGE2_LR, "轮数": config.STAGE2_EPOCHS},
    ])
    html_table(list(stage_df.columns), stage_df.values.tolist(),
               align_right=[False, False, False, True, True])
    st.caption("MobileNetV2 的 features 共 19 个 block，越靠后通道数越多、"
               "参数量越集中。所以只解冻最后 5 个 block，"
               "就已经占到全部参数的四分之三。")

    st.info(
        "**为什么要分两阶段**？换掉分类头之后，新头的权重是随机初始化的，"
        "一开始会产生很大的梯度。如果这时整个网络一起训练，这些大梯度会顺着"
        "反向传播把预训练好的主干权重冲乱 —— 实测直接把主干解冻一起训，"
        "前两个 epoch 准确率掉到 20% 多。先冻结主干让新头收敛，"
        "再用小一个数量级的学习率解冻后段微调，训练才稳定。",
        icon=":material/lightbulb:")

    st.markdown("**训练配置**")
    cfg_rows = [
        ["优化器", "AdamW (weight_decay=1e-4)"],
        ["学习率调度", "余弦退火 CosineAnnealingLR"],
        ["损失函数", "交叉熵 CrossEntropyLoss"],
        ["批大小", str(config.BATCH_SIZE)],
        ["随机种子", str(config.RANDOM_SEED)],
        ["权重选择", "按验证集准确率保存最优，而非最后一个 epoch"],
    ]
    html_table(["项目", "设置"], cfg_rows)

    st.markdown("**数据增强**")
    st.markdown("""
- `RandomResizedCrop(scale=(0.6, 1.0))` —— 让模型对菜品在画面中的大小不敏感。
  下限取 0.6 而不是常见的 0.08：裁太狠会只剩一块看不出是什么的局部
- `RandomHorizontalFlip` —— 左右翻转后仍是同一道菜
- `ColorJitter(0.2)` —— 幅度给得较小。菜品识别很依赖颜色，
  抖动过强会破坏有效线索
- 不做垂直翻转 —— 一碗面倒过来不符合真实拍摄场景
""")

    # 训练曲线
    # 文件名（train_result_full.json 这类）是内部命名，不该展示给用户，
    # 所以这里用 tag 映射成人话。准确率也从 0.9602 这种原始小数改成百分比 ——
    # 用户扫一眼就能懂，不用自己做换算。
    _RUN_LABEL = {
        "full": "完整方法（预训练 + 两阶段训练 + 数据增强）",
        "noaug": "对照实验：关闭数据增强",
        "stage1only": "对照实验：只训练分类头，不做微调",
        "scratch": "对照实验：不用预训练，从头训练",
        "main": "主模型",
    }
    result_files = sorted(config.MODEL_DIR.glob("train_result*.json"))
    if result_files:
        st.markdown("**训练曲线**")
        for rf in result_files:
            with open(rf, encoding="utf-8") as f:
                res = json.load(f)
            hist = res.get("history", [])
            if not hist:
                continue
            _tag = res.get("tag") or res.get("name") or \
                rf.stem.replace("train_result_", "").replace("train_result", "main")
            _label = _RUN_LABEL.get(_tag, _tag)
            st.caption("**%s**　测试集准确率 %.2f%%　训练设备 %s"
                       % (_label, res.get("test_acc", 0.0) * 100,
                          res.get("device", "cpu")))
            df = pd.DataFrame(hist)
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("*准确率*")
                st.line_chart(df, x="epoch", y=["train_acc", "val_acc"],
                              height=240)
            with ch2:
                st.markdown("*损失*")
                st.line_chart(df, x="epoch", y=["train_loss", "val_loss"],
                              height=240)

    # 评估结果与图表
    eval_file = config.MODEL_DIR / "eval_summary.json"
    if eval_file.exists():
        with open(eval_file, encoding="utf-8") as f:
            ev = json.load(f)
        st.markdown("**测试集评估结果**")
        c = st.columns(3)
        c[0].metric("Top-1 准确率", "%.2f%%" % (ev["accuracy"] * 100))
        c[1].metric("Top-3 准确率", "%.3f" % ev["top3_accuracy"])
        c[2].metric("宏平均 F1", "%.3f" % ev["macro_f1"])

        if "confidence_stats" in ev:
            cs = ev["confidence_stats"]
            st.caption(
                "错分样本的平均置信度 %.3f，正确样本 %.3f。"
                "置信度高于 0.7 的错误有 %d 个，占全部错误的 %.1f%%。"
                % (cs["wrong_mean_conf"], cs["correct_mean_conf"],
                   cs["wrong_above_0.7"],
                   100.0 * cs["wrong_above_0.7"] / max(cs["wrong_total"], 1)))

        figs = [
            ("confusion_matrix.png", "混淆矩阵（按行归一化）"),
            ("per_class_f1.png", "各类别 F1 分数"),
            ("top_confusions.png", "最易混淆的类别对"),
            ("confidence.png", "预测置信度分布"),
            ("misclassified.png", "高置信度错分样本"),
        ]
        shown = 0
        for fn, title in figs:
            pth = config.FIGURE_DIR / fn
            if pth.exists():
                st.markdown("**%s**" % title)
                st.image(str(pth), width="stretch")
                shown += 1
        if shown == 0:
            st.caption("还没有生成图表。运行 `python src/evaluate.py` 生成。")

    # 跨数据集泛化测试结果（这一段不能写在上面那个循环里，
    # 否则只有存在某张图时才会被渲染到）
    cr = config.MODEL_DIR / "eval_summary_cross.json"
    if cr.exists():
        with open(cr, encoding="utf-8") as f:
            cros = json.load(f)
        st.markdown("**跨数据集泛化测试**")
        st.caption("在另一个来源的 5 类家常菜数据上直接测试（模型未见过这批图）："
                   "Top-1 准确率 %.2f%%（全部 %d 张）"
                   % (cros["accuracy"] * 100, cros.get("n_samples", 0)))
        st.caption("这批图来自 `jiezh2/common-chinese-food` 数据集，"
                   "拍摄风格、构图、光照都与训练集不同，"
                   "用来检验模型是不是只记住了训练集的表面特征。")

    # ---------------- 开集识别（拒识）----------------
    os_p = config.MODEL_DIR / "open_set.json"
    if os_p.exists():
        with open(os_p, encoding="utf-8") as f:
            osr = json.load(f)
        st.markdown("**开集识别：拒识陌生输入**")
        st.markdown(
            "模型只认识训练过的 %d 道菜。输入别的菜时，普通分类器也会给出"
            "一个概率最高的答案——第 4.6 节已经量化过这个问题。"
            "为此系统加了拒识机制：用能量分 `logsumexp(logits)` 判断输入"
            "是否属于模型学过的范围。" % len(classes))

        html_table(
            ["评分方法", "AUROC", "未知拒识率", "已知接受率"],
            [[m["name"], m["auroc"], m["novel_reject_at_95_accept"],
              m["known_accept_rate"]] for m in osr["methods"]],
            align_right=[False, True, True, True])
        st.caption(
            "「未知拒识率」在固定「已知接受率 95%%」的前提下测得："
            "只让 5%% 的正常菜品被要求重拍，能挡掉多少陌生输入。"
            "评测用 %d 张已知测试图 + %d 张未知类图（模型没学过的 4 类家常菜）。"
            % (osr["n_known"], osr["n_novel"]))

        c1, c2, c3 = st.columns(3)
        c1.metric("最优方法", osr["best_method"])
        c2.metric("拒识阈值（能量分）", "%.2f" % osr["reject_threshold"])
        best_m = max(osr["methods"], key=lambda m: m["auroc"])
        c3.metric("未知拒识率", "%.1f%%"
                  % (best_m["novel_reject_at_95_accept"] * 100))

        st.caption("阈值与拒识判断已在「菜品识别」页生效："
                   "能量分低于该值时会提示「这看起来不是系统支持的菜品」。")

        with st.expander("阈值选择对照表", icon=":material/tune:"):
            sw = osr["threshold_sweep"]
            cols = list(sw[0].keys()) if sw else []
            html_table(cols, [[r[c] for c in cols] for r in sw],
                       align_right=[True, True, True])
            st.caption("能量分越高越像模型学过的菜。阈值定得高更保守，"
                       "会拒掉更多陌生输入，但正常菜品被误拒的概率也上升。")

        fig = config.FIGURE_DIR / "open_set_roc.png"
        if fig.exists():
            st.image(str(fig), width="stretch")

# ---------------------------------------------------------------- 营养来源
with tab_nutrition:
    st.subheader("营养数据来源")
    try:
        import nutrition as nut
        db = nut.NutritionDB()
    except Exception as e:
        st.error("营养库加载失败：%s" % e)
        st.stop()

    src_map = {
        "A": ("USDA 官方 API 直连", "green"),
        "B": ("USDA 镜像站", "blue"),
        "C": ("中国食物成分表", "orange"),
        "D": ("按配料配方合成", "violet"),
    }

    rows = []
    for k, d in db.dishes.items():
        n = d.per_100g
        rows.append({
            "菜名": d.cn_name,
            "标识": k,
            "kcal/100g": n.energy_kcal,
            "蛋白(g)": n.protein_g,
            "脂肪(g)": n.fat_g,
            "碳水(g)": n.carb_g,
            "钠(mg)": n.sodium_mg,
            "一份(g)": d.serving_g,
            "来源": src_map.get(d.source_type, ("未知", "gray"))[0],
            "来源编号": d.source_id,
        })
    # 18 行 × 10 列。列多，HTML 表格会横向滚动；
    # 这是只读汇总，不需要排序，所以换过来让边框清晰。
    cols = ["菜名", "标识", "kcal/100g", "蛋白(g)", "脂肪(g)", "碳水(g)",
            "钠(mg)", "一份(g)", "来源", "来源编号"]
    html_table(cols, [[r[c] for c in cols] for r in rows],
               align_right=[False, False, True, True, True, True, True,
                            True, False, False])

    st.markdown("**来源说明**")
    st.markdown("""
- **USDA 官方 API 直连** —— 通过 `api.nal.usda.gov/fdc/v1` 取得，
  返回 HTTP 200 的原始响应
- **USDA 镜像站** —— USDA 官方 API 的免费额度是 30 次/小时/IP，
  取数时被限流，改用 nutritionvalue.org（该站声明数据来自
  USDA National Nutrient Database）。**属二手来源，已逐条标注**
- **中国食物成分表** —— 中国疾病预防控制中心营养与健康所官方查询平台。
  引用限制：该平台能量只印 kJ（本表 kcal 为按 ÷4.184 换算值）、
  不显示食物编码、"未检测"与"0"含义不同
- **按配料配方合成** —— 中国食物成分表没有"宫保鸡丁""东坡肉"这类
  菜肴条目（已对该库 259~1580 全部 1341 条做过穷举核验），
  因此改为取各配料每 100 g 的官方值，再按配方加权求和
""")

    st.markdown("**独立复核**")
    st.markdown("""
项目里带了 `src/check_nutrition.py`，会做两件事：

1. **重新抓取比对** —— 对镜像来源的条目重新访问页面，逐字段核对
2. **Atwater 自洽性检查** —— 用 4/9/4 系数从三大营养素反算能量，
   和标注能量比对。这个检查独立于数据来源，能发现抄录错误
""")

    # 现场跑一次 Atwater 检查，比写"已检查"有说服力
    st.markdown("**Atwater 能量自洽性检查：现场计算**")
    chk = []
    for k, d in db.dishes.items():
        n = d.per_100g
        if n.energy_kcal <= 0:
            continue
        calc = n.protein_g * 4 + n.fat_g * 9 + n.carb_g * 4
        dev = (calc - n.energy_kcal) / n.energy_kcal * 100
        chk.append({
            "菜名": d.cn_name,
            "标注能量": round(n.energy_kcal, 1),
            "按4/9/4反算": round(calc, 1),
            "偏差": "%.1f%%" % dev,
            "结论": "通过" if abs(dev) <= 12 else "偏差偏大，需复查",
        })
    cols = ["菜名", "标注能量", "按4/9/4反算", "偏差", "结论"]
    html_table(cols, [[r[c] for c in cols] for r in chk],
               align_right=[False, True, True, True, False])
    n_bad = sum(1 for r in chk if r["结论"] != "通过")
    if n_bad == 0:
        st.success("全部 %d 条通过自洽性检查（偏差均 ≤ 12%%）。" % len(chk),
                   icon=":material/check_circle:")
    else:
        st.warning("有 %d 条偏差偏大，需要复查。" % n_bad,
                   icon=":material/warning:")
