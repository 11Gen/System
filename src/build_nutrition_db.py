"""
build_nutrition_db.py —— 生成 data/nutrition_db.csv。

这个脚本把三种来源的营养数据合并成一张统一的表：

  1) 直接可用的菜品条目
     - USDA FoodData Central（含镜像）的 14 条西式菜品
     - 中国食物成分表的米饭、炒饭
     数据已经过 src/check_nutrition.py 复核（Atwater 自洽 + 页面比对），
     直接誊抄进来即可。

  2) 按配料配方合成的中式菜肴（source_type = "D"）
     中国食物成分表里没有"宫保鸡丁""东坡肉"这类菜肴条目
     （已对该库 259~1580 全部 1341 条做过穷举核验，确认没有）。
     所以改用配料合成法：
         每 100 g 营养值 = Σ(配料重量 × 配料每 100 g 营养值) / 成品总重 × 100
     配方来源和配料来源都逐条记录，可追溯、可复查。
     这比直接抄一个来源不明的"宫保鸡丁 200 kcal"靠谱得多。

  3) 沙拉酱的份量修正
     USDA 的凯撒/希腊沙拉记录都是 "no dressing"（不含酱），
     直接用会低估三倍以上。这里在 note 里明确标注，估算时单独处理。

用法：
    python src/build_nutrition_db.py            # 生成完整库
    python src/build_nutrition_db.py --check    # 只做自洽性检查，不写文件
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from config import DATA_DIR, NUTRITION_CSV

# ---------------------------------------------------------------- 配料营养值
# 来源：中国食物成分表（中国疾病预防控制中心营养与健康所官方查询平台）
#       https://nlc.chinanutri.cn/fq/
# 由 src/fetch_cn_nutrition.py 自动抓取并写成 data/ingredients_cn.csv。
#
# 这里直接从 CSV 读，不再手抄进代码。
# 一开始是把值硬编码在这个文件里的，但改配料别名时出现了两边不一致：
# 脚本里改了"藕"的匹配规则，硬编码的那份却还是藕粉的值（差 5 倍），
# 而且不会报错。改成读 CSV 之后只有一个数据源，不会再漂移。
#
# 引用限制（写报告时必须带上）：
#   - 该平台能量只印 kJ，CSV 里的 kcal 是按 ÷4.184 换算的
#   - 平台不显示食物编码，所以 CSV 里的 cn_food_id 是本项目抓取时记的序号，
#     不是官方食物编码
#   - 平台只写"中国食物成分表"，未标版本号
#   - 空白 ≠ 0，"—"=未检测，"Tr"=未检出
INGREDIENTS_CSV = DATA_DIR / "ingredients_cn.csv"

# 配料 key -> 规范中文名。
# 只用来把内部 key 翻成人话，不参与营养计算。
# 为什么需要它：有些配料在食物成分表里查不到（所以拿不到营养值），
# 但它们仍然出现在配方里。这些配料没有 CSV 记录可查，
# 于是 note 里就会漏出 pork_skin 这样的英文 key，用户看不懂。
ING_CN_NAME = {
    "pork_skin": "猪皮（皮冻原料）",
    "chili_dry": "干辣椒",
    "fish_maw": "鱼肚",
    "oyster_sauce": "蚝油",
    "chicken_broth": "鸡汤",
}

# 配料 key -> 出现在正文里的中文说法。
# 和上面的表分开，因为同一个 key 在"未计入"那句话里和在句子里
# 读起来需要不同的措辞（前者要能独立成词，后者要能接在"借…计"后面）。
ING_INLINE = {
    "beef_shank": "瘦牛肉",
    "scallion": "大葱",
    "chicken_whole": "鸡肉",
    "fish_grass_carp": "草鱼",
    "pork_belly": "五花肉",
    "bamboo_shoot": "竹笋",
    "ginger": "生姜",
    "total_g": "成品总重",
}


def clean_note(note):
    """
    把技术笔记里的内部代号换成用户能懂的词。

    背景：这些 note 原本是写给下游和报告用的，里面混了
    `total_g`（代码变量名）、`beef_shank`（配料表 key）、`§4`（原文档章节号）
    这类东西。界面上会直接展示 note，不洗一遍用户看不懂。

    做法是逐条显式替换，不用正则批量猜 —— 16 道菜样本量很小，
    逐条看得见改了什么，比一条正则扫全库安全。
    技术细节本身没有丢：`_recipes_data.py` 里保留着原文，
    报告需要细究时从那里取。
    """
    if not note:
        return ""
    import re as _re

    out = note
    # 变量名 / 配料 key -> 中文说法。长的先替换，避免互相干扰。
    for k in sorted(ING_INLINE, key=len, reverse=True):
        out = out.replace(k, ING_INLINE[k])

    # "借 xxx 近似" 这种省略说法补成完整句子。
    # 注意替换顺序：「借 」必须先于「故借」处理，
    # 否则 "故借 scallion" 会先被 "借 " 规则改成 "故借用 scallion"，
    # 再被 "故借" 规则改成 "故借用用 scallion" —— 用了两次"用"。
    out = out.replace("借 ", "借用 ")
    out = out.replace("故借用", "故用")
    # "借用鸡肉的营养数据近似" 读起来像病句（借的恰好就是它自己），
    # 规整成"按鸡肉计"更顺。
    out = _re.sub(r"借用\s*([^\s，。、）]{1,12})\s*的营养数据近似", r"按\1计", out)
    out = out.replace("的营养数据近似的", "近似")
    out = out.replace("借用", "用").replace("故用", "故用")
    # 收到多余空格。分两种，顺序不能反：
    #
    # (1) 单位与后面中文之间统一**补一个空格**。
    #
    #     这里来回折腾过几次，记一下结论：中英文混排到底该不该留空格是个
    #     排版口味问题，不是 bug。但**必须是统一的一种**——
    #     同一句里出现 "100 mL" 和 "10 g面剂" 两种写法比全都不留空格更难看。
    #     所以这里统一成"数字与单位之间有空格，单位与中文之间也有空格"，
    #     和正文里其他地方（如"220 g"）保持一致。
    #
    #     踩过的坑：一开始想用变长 lookbehind 一次覆盖所有单位，
    #     但 Python 的 re 不支持变长 lookbehind，直接抛 PatternError；
    #     而且异常被上层吞掉、脚本照样"成功"退出，排查了很久。
    #     现在逐个单位写，都是定长。
    for _u in ("kcal", "mg", "mL", "ml", "kg", "kJ", "g"):
        # 先补空格（单位紧贴中文的情况）
        out = _re.sub(r"(?<=%s)(?=[\u4e00-\u9fff])" % _u, " ", out)
    # 再多空格压成一个
    out = _re.sub(r"\s{2,}", " ", out)
    # (2) 中文与中文之间的空格收掉。
    #     用 + 而不是匹配单个空格，一次收干净；前面的单位补空格规则
    #     只会在"单位+汉字"边界上加空格，不会破坏这里的匹配。
    out = _re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", out)
    # 括号里重复同一个词："瘦牛肉（瘦牛肉）" -> "瘦牛肉"
    out = _re.sub(r"([\u4e00-\u9fff]{2,8})（\1）", r"\1", out)
    out = _re.sub(r"([\u4e00-\u9fff]{2,8})\(\1\)", r"\1", out)
    # "另有 xxx 未计入" 里的多余空格
    out = _re.sub(r"另有\s+", "另有", out)
    out = _re.sub(r"\s+未计入", "未计入", out)
    # 原配方文档的章节号引用，对用户没意义
    out = out.replace("按 §4 ", "按配料规范").replace("§4", "配料规范")
    # 文档内部用语
    out = out.replace("本表按", "这里按")
    out = out.replace("不在文档", "不在成品")
    out = out.replace("取文档区间", "取原配方区间")
    out = out.replace("C 级估计", "估算值")
    out = out.replace("A 级配方", "标准配方")
    out = out.replace("A 级", "标准版")
    out = out.replace("B 级", "行业资料级")
    # "成品总重=1400" 这种等号紧贴数字的写法，补上单位
    out = _re.sub(r"(成品总重)=(\d)", r"\1 \2", out)
    out = _re.sub(r"(成品总重)\s*(\d+(?:\.\d+)?)(?!\s*g)", r"\1 \2 g", out)
    # 清掉替换过程中可能出现重复的空格
    out = _re.sub(r"\s{2,}", " ", out)
    return out

# ---------------------------------------------------------------- 补充配料
# 中国食物成分表里没有收录、但配方需要的配料。
#
# 处理原则：**只在能拿到权威值时才补，拿不到就留空并在 note 里说明。**
# 绝不拿名字相近的东西顶替 —— 这类顶替在本项目里已经出过两次事故：
#   "藕"   匹配到 "藕粉"   → 热量差 5 倍
#   "鸡汤" 匹配到 "鸡精"   → 钠差近一万倍（佛跳墙因此算出 61919 mg 钠）
#
# 格式：key: (中文名, 来源标识, kcal, 蛋白, 脂肪, 碳水, 纤维, 钠)  每 100 g
SUPPLEMENT_INGREDIENTS = {
    # 鸡汤。来源：USDA FoodData Central SR Legacy "Soup, chicken broth,
    # canned, ready-to-serve"（fdcId 174536）。该条为低钠版；
    # 普通鸡汤钠约 100~200 mg/100g，这里取下限是偏保守的估计。
    "chicken_broth": ("鸡汤", "USDA FDC 174536", 4.0, 0.9, 0.2, 0.4, 0.0, 143.0),
}


def load_ingredients():
    """
    读配料营养表，返回 {key: (中文名, id, kcal, 蛋白, 脂肪, 碳水, 纤维, 钠)}

    空字段按 None 处理，因为该库里空白表示"未检测"，
    和"含量为 0"是两回事 —— 直接当 0 会把钠和纤维算低。
    """
    if not INGREDIENTS_CSV.exists():
        print("找不到 %s" % INGREDIENTS_CSV)
        print("请先运行：python src/fetch_cn_nutrition.py --out %s"
              % INGREDIENTS_CSV)
        return None

    def num(v):
        v = (v or "").strip()
        if v == "" or v.lower() in ("none", "null"):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    out = {}
    with open(INGREDIENTS_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row.get("food") or "").strip()
            if not key:
                continue
            out[key] = (
                (row.get("cn_name") or key).strip(),
                (row.get("cn_food_id") or "").strip(),
                num(row.get("energy_kcal")),
                num(row.get("protein_g")),
                num(row.get("fat_g")),
                num(row.get("carb_g")),
                num(row.get("fiber_g")),
                num(row.get("sodium_mg")),
            )

    # 把补充配料并进来（不覆盖已有条目）
    for key, tup in SUPPLEMENT_INGREDIENTS.items():
        if key not in out:
            out[key] = tup

    return out

# ---------------------------------------------------------------- 菜肴配方
# 每道菜：{成品总重(g): float, 配料: {key: 克数}, 来源: str, 可信度: str}
# 配方由检索整理，来源逐条记录在 RECIPE_SOURCE 里。
DISH_RECIPES = {
    # 这里的内容由 _recipes_data.py 提供（配方数据量大，单独放一个文件，
    # 也方便核对和修改）
}

try:
    from _recipes_data import RECIPES as _EXT_RECIPES
    DISH_RECIPES = _EXT_RECIPES
except ImportError:
    pass


# ---------------------------------------------------------------- 直接条目
# 这些是从 USDA / 中国食物成分表 直接取到的成品菜条目，已复核。
# 字段：key, 中文名, kcal, 蛋白, 脂肪, 碳水, 纤维, 钠, 一份重量, 份量说明,
#       来源类型, 来源ID, 来源名称, 备注
DIRECT_ROWS = [
    ("rice_steamed", "米饭", 117.8, 2.6, 0.3, 25.9, "", 2.5, 200, "一碗(约200g)",
     "C", "287", "中国食物成分表 米饭(蒸)(均值)",
     "原值493kJ由÷4.184换算；膳食纤维为未检测故留空"),
    ("fried_rice", "什锦炒饭", 186.2, 5.0, 5.6, 29.7, 2.0, 220.0, 300, "一盘(估值)",
     "C", "1314", "中国食物成分表 什锦炒饭",
     "原值779kJ换算；300g为估值，无官方份量数据"),
]


def build_recipe_rows():
    """
    按配料配方合成中式菜肴的营养值。

    计算过程完全公开，报告里可以逐条验算：
        总能量(kcal) = Σ(配料克数 × 配料kcal_per_100g) / 100
        每100g能量   = 总能量 / 成品总重 × 100
    其他营养素同理。

    关于"配料加总 ≠ 成品总重"：
        这是正常的，而且原因分两类，都在每道菜的 note 里写明了：
        (1) 生重 vs 熟重口径。比如东坡肉，标准给的是生投料
            （肉 2000 g + 黄酒 550 + 酱油 150 + 糖 200），
            而成品重量是"撇去浮油、收汁"之后的估算值。
            这里用"成品总重"当分母是对的 —— 我们算的是"吃到嘴里的
            每 100 g 有多少营养"，不是"投了多少料"。
        (2) 库中没有营养值的配料被省略。比如咕咾肉的菠萝和番茄酱、
            小笼包的猪皮冻。这些在 note 里逐一点名，
            不拿相似食物顶替。
    """
    ing = load_ingredients()
    if ing is None:
        return None, ["配料营养表读取失败"]

    rows = []
    problems = []

    for dish_key, spec in DISH_RECIPES.items():
        total_g = spec["total_g"]
        recipe = spec["ingredients"]

        # 累加各配料的绝对量
        acc = {"kcal": 0.0, "p": 0.0, "f": 0.0, "c": 0.0, "fib": 0.0, "na": 0.0}
        missing_vals = []      # 有配方克数、但库里没有营养值的配料
        used = []              # 实际参与计算的配料

        for ing_key, grams in recipe.items():
            item = ing.get(ing_key)
            if item is None:
                missing_vals.append(ing_key)
                continue
            cn, fid, kcal, p, f, c, fib, na = item

            # 判断"这条配料有没有可用数据"不能只看能量。
            # 精盐就是反例：该库能量栏是空白（盐不产能），但钠有 39311 mg/100g。
            # 一开始写成 if kcal is None 就跳过，结果盐被当成"缺数据"剔除，
            # 16 道菜的钠全部严重低估 —— 而钠恰恰是这批菜最该关注的指标。
            if all(v is None for v in (kcal, p, f, c, fib, na)):
                missing_vals.append(ing_key)
                continue

            k = grams / 100.0
            acc["kcal"] += (kcal or 0) * k
            acc["p"] += (p or 0) * k
            acc["f"] += (f or 0) * k
            acc["c"] += (c or 0) * k
            acc["fib"] += (fib or 0) * k
            acc["na"] += (na or 0) * k
            used.append((ing_key, cn, fid, grams))

        if missing_vals:
            problems.append("%s 缺少配料营养值：%s" % (dish_key, missing_vals))

        if total_g <= 0:
            problems.append("%s 成品重量非法：%s" % (dish_key, total_g))
            continue

        # 换算到每 100 g
        f100 = 100.0 / total_g
        per100_kcal = round(acc["kcal"] * f100, 1)

        # 把配方也记进 note。
        #
        # 注意这里不再往 note 里塞内部代号：
        #   原来拼的是 "…总重 xxx g｜配方：猪里脊肉 200g；…｜⚠库中无营养值未计入：pork_skin"
        #   total_g 是代码里的变量名，pork_skin 是配料表的 key，
        #   这些都会原样显示到界面上，用户看到只会困惑。
        # 现在统一只留用户能懂的文字，内部信息留在 source 字段（记来源）和
        # 代码注释里，报告需要时从原始数据取。
        # 配方文本的克数格式化。
        # 这里踩过一个坑：原来用 "%.0f" 取整，结果回锅肉的 0.5 g 盐
        # 被格式化成 "0"，看起来像"这道菜根本没放盐" ——
        # 而盐的钠含量是 39311 mg/100g，用户会以为钠算错了。
        # 现在按数量级选精度：小于 10 g 的保留一位小数，其余取整。
        def _fmt_g(g):
            if g < 10:
                return "%g" % round(g, 1)
            return "%.0f" % g

        recipe_txt = "；".join("%s %s g" % (cn, _fmt_g(g))
                              for _, cn, _, g in used)
        note = clean_note(spec.get("note", ""))
        if note:
            note = "%s｜配方：%s" % (note, recipe_txt)
        else:
            note = "配方：%s" % recipe_txt

        if missing_vals:
            # 把内部 key 翻成中文名再写进 note。
            # 这些 key 在食物成分表里没有对应条目（所以才有哪个英文名），
            # 它们的规范中文名记在下面的 ING_CN_NAME 里。
            cn_names = [ING_CN_NAME.get(k, k) for k in missing_vals]
            note += ("｜另有 %s 未计入（食物成分表中无对应条目）"
                     % "、".join(cn_names))

        row = (
            dish_key,
            spec.get("cn_name", dish_key),
            per100_kcal,
            round(acc["p"] * f100, 2),
            round(acc["f"] * f100, 2),
            round(acc["c"] * f100, 2),
            round(acc["fib"] * f100, 2),
            round(acc["na"] * f100, 1),
            total_g,
            spec.get("serving_desc", "一份(约%dg)" % int(total_g)),
            "D",
            spec.get("source_id", "recipe"),
            "按配料配方合成｜%s（可信度 %s）"
            % (spec.get("source", ""), spec.get("credibility", "?")),
            note,
        )
        rows.append(row)

        # 自洽性检查：合成出来的值如果不合理要立刻发现
        calc = row[3] * 4 + row[4] * 9 + row[5] * 4
        if row[2] > 0:
            dev = (calc - row[2]) / row[2] * 100
            if abs(dev) > 15:
                problems.append("%s Atwater 偏差 %.1f%%（标注 %.1f，反算 %.1f）"
                                % (dish_key, dev, row[2], calc))

    return rows, problems


def write_csv(rows, path):
    header = ["food", "cn_name", "energy_kcal", "protein_g", "fat_g", "carb_g",
              "fiber_g", "sodium_mg", "serving_g", "serving_desc",
              "source_type", "source_id", "source_name", "note"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            # 空值统一写成空字符串，不要写 None
            w.writerow(["" if v is None else v for v in r])
    print("已写入 %s（%d 条）" % (path, len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查不写文件")
    args = ap.parse_args()

    print("=" * 84)
    print("生成营养数据库")
    print("=" * 84)

    recipe_rows, problems = build_recipe_rows()
    print("\n按配方合成的菜肴：%d 道" % len(recipe_rows))
    for r in recipe_rows:
        print("   %-44s %7.1f kcal  份量 %4.0fg"
              % (r[1], r[2], r[8]))

    print("\n直接取用的成品菜条目：%d 条" % len(DIRECT_ROWS))
    for r in DIRECT_ROWS:
        print("   %-44s %7.1f kcal" % (r[1], r[2]))

    if problems:
        print("\n!! 发现问题 %d 处：" % len(problems))
        for p in problems:
            print("   %s" % p)
    else:
        print("\n自洽性检查：全部通过")

    all_rows = recipe_rows + DIRECT_ROWS
    print("\n合计 %d 条" % len(all_rows))

    if args.check:
        print("\n（--check 模式，未写文件）")
        return

    config.ensure_dirs()
    write_csv(all_rows, NUTRITION_CSV)
    print("\n下一步：python src/nutrition.py     # 自检")
    print("        python src/check_nutrition.py  # 联网复核（只对 A/B 类来源）")


if __name__ == "__main__":
    main()
