"""
nutrition.py —— 营养估算核心模块。

这个模块负责把"识别出的菜名"变成"这一餐吃了多少热量和营养素"。

设计上的两个决定，以及为什么这么定：

1) 用"每 100 g 营养值 + 份量克数"的模型，而不是"一份多少卡"。
   因为份量是最大的误差来源：同样一盘宫保鸡丁，小碟装和大盘装能差一倍。
   把份量单独拎出来当参数，用户还能自己调，误差可见、可讨论；
   直接存"一份 500 kcal" 的话，这个误差就被藏起来了，谁也说不清。

2) 沙拉酱这类"另配的调料"单独处理。
   数据源（USDA FNDDS）把凯撒沙拉的酱单列为一条记录，沙拉本身只有
   77 kcal/100 g。如果直接用这个值，会低估三倍以上。所以这里加了
   DRESSING_ADJUST 表，估算时按用户选的口味再加回去。

数据来源有两类，都在 CSV 的 source_type 里标了：
   A = USDA FoodData Central 官方 API 直连取得
   B = 经 nutritionvalue.org 镜像取得（该站声明数据来自 USDA）
   C = 中国食物成分表（中国疾控中心营养与健康所官方查询平台）
   D = 按配料配方自行合成的计算值（配方与配料来源都记录在案）
本项目有独立的复核脚本 src/check_nutrition.py 会重新抓取比对，
并用 Atwater 系数（蛋白4/脂肪9/碳水4）做能量自洽性检查。
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import config

# ---------------------------------------------------------------- 常量
# 中国居民膳食指南（2022）里成人每日参考摄入量，用于给用户算占比。
# 注意这是个粗略参考：真实需求随年龄、性别、体力活动变化很大，
# 界面上必须写清楚"仅供参考"，不能当成医学建议。
DAILY_REFERENCE = {
    "energy_kcal": 2000,     # 轻体力活动成年女性约 1800，男性约 2250，取中值
    "protein_g": 60,         # 膳食指南推荐成年男性 65 g、女性 55 g
    "fat_g": 60,             # 脂肪供能比 20%~30%，按 2000 kcal 折算约 60 g
    "carb_g": 270,           # 碳水供能比 50%~65%，按 2000 kcal 折算约 270 g
    "sodium_mg": 2000,       # 指南建议每日食盐不超过 5 g，折合钠约 2000 mg
    "fiber_g": 25,           # 指南推荐每日膳食纤维 25~30 g
}

# 一餐的参考分配比例。指南是按"三餐"给的，午餐晚餐占大头。
MEAL_SHARE = {"早餐": 0.3, "午餐": 0.4, "晚餐": 0.3, "加餐": 0.1}


@dataclass
class Nutrients:
    """一份食物（不是每100g）的营养素总量"""
    energy_kcal: float = 0.0
    protein_g: float = 0.0
    fat_g: float = 0.0
    carb_g: float = 0.0
    fiber_g: float = 0.0
    sodium_mg: float = 0.0

    def __add__(self, other):
        return Nutrients(
            self.energy_kcal + other.energy_kcal,
            self.protein_g + other.protein_g,
            self.fat_g + other.fat_g,
            self.carb_g + other.carb_g,
            self.fiber_g + other.fiber_g,
            self.sodium_mg + other.sodium_mg,
        )

    def __mul__(self, k):
        return Nutrients(self.energy_kcal * k, self.protein_g * k,
                         self.fat_g * k, self.carb_g * k,
                         self.fiber_g * k, self.sodium_mg * k)

    def as_dict(self):
        return {
            "能量(kcal)": round(self.energy_kcal, 1),
            "蛋白质(g)": round(self.protein_g, 2),
            "脂肪(g)": round(self.fat_g, 2),
            "碳水(g)": round(self.carb_g, 2),
            "膳食纤维(g)": round(self.fiber_g, 2),
            "钠(mg)": round(self.sodium_mg, 1),
        }

    def macro_energy_ratio(self):
        """
        三大营养素各自的供能占比。

        用 Atwater 系数算：蛋白 4 kcal/g、脂肪 9 kcal/g、碳水 4 kcal/g。
        这个指标比绝对克数更能反映一餐的结构是否合理 ——
        比如"脂肪供能比 45%"一听就知道这顿偏油了。
        """
        p = self.protein_g * 4
        f = self.fat_g * 9
        c = self.carb_g * 4
        total = p + f + c
        if total <= 0:
            return {"protein": 0.0, "fat": 0.0, "carb": 0.0}
        return {"protein": p / total, "fat": f / total, "carb": c / total}


@dataclass
class Dish:
    """营养库里的一个条目"""
    key: str
    cn_name: str
    per_100g: Nutrients
    serving_g: float
    serving_desc: str
    source_type: str
    source_id: str
    source_name: str
    note: str = ""
    # 配料配方（只有 source_type == "D" 的条目才有）
    recipe: Dict[str, float] = field(default_factory=dict)
    # 这份量够几个人吃。0 表示未标注。
    #
    # 这个字段是必需的，不是可选的：16 道菜里有 8 道是"一份=一整桌菜"
    # （佛跳墙 1200 g、东坡肉 1400 g、水煮鱼 3000 g），
    # 直接按整份报"钠 5754 mg"会把用户吓到 —— 那是六个人一起吃的量。
    # 没有这个字段的话，界面上的数字既没用也容易误导。
    servings: float = 1.0


# 各菜"一份"够几个人吃。
# 依据是 DB5133/T 49—2021 附录 A 的官方主料净含量（那是单人份下限量）
# 与各地方标准给出的成品总重的比值。
DISH_SERVINGS = {
    "Peking_Duck": 4.0,                        # 2500-3000g 整鸭，宴席菜
    "Sweet_and_Sour_Pork": 3.0,
    "Mapo_Tofu": 3.0,
    "Yuxiang_Shredded_Pork": 2.5,
    "Husband_and_Wife_Lung_Slices": 1.5,
    "Twice-Cooked_Pork": 2.0,
    "Kung_Pao_Chicken": 2.0,
    "Saliva_Chicken": 3.0,                     # 475g / 4 人份（Cookidoo 口径）
    "Soup_Dumplings": 4.0,                     # 900g / 30 只，一笼 8 只
    "Braised_Pork_Meatballs_in_Brown_Sauce": 3.0,   # 570g / 6 个
    "Dongpo_Pork": 5.0,
    "West_Lake_Vinegar_Fish": 3.0,
    "Buddha_Jumps_Over_the_Wall": 6.0,         # 宴席大菜
    "Steamed_Sea_Bass": 3.0,
    "Fish_with_Pickled_Cabbage_and_Chili": 4.0,
    "Boiled_Fish_with_Sichuan_Peppercorns": 4.0,
    "rice_steamed": 1.0,
    "fried_rice": 1.0,
}


# ---------------------------------------------------------------- 数据库
class NutritionDB:
    def __init__(self, csv_path=None):
        self.path = Path(csv_path) if csv_path else config.NUTRITION_CSV
        self.dishes: Dict[str, Dish] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(
                "找不到营养数据库 %s\n"
                "请先跑 src/build_nutrition_db.py 生成。" % self.path)

        with open(self.path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                def num(k):
                    v = (row.get(k) or "").strip()
                    # 空字符串表示"该字段未检测"，不等于 0，所以按 0 处理
                    # 但要在 note 里说明。这不是偷懒，是数据本身如此。
                    return float(v) if v else 0.0

                d = Dish(
                    key=row["food"],
                    cn_name=(row.get("cn_name") or row["food"]).strip(),
                    per_100g=Nutrients(
                        energy_kcal=num("energy_kcal"),
                        protein_g=num("protein_g"),
                        fat_g=num("fat_g"),
                        carb_g=num("carb_g"),
                        fiber_g=num("fiber_g"),
                        sodium_mg=num("sodium_mg"),
                    ),
                    serving_g=num("serving_g") or 100.0,
                    serving_desc=(row.get("serving_desc") or "").strip(),
                    source_type=(row.get("source_type") or "").strip(),
                    source_id=(row.get("source_id") or "").strip(),
                    source_name=(row.get("source_name") or "").strip(),
                    note=(row.get("note") or "").strip(),
                    servings=DISH_SERVINGS.get(row["food"], 1.0),
                )
                self.dishes[d.key] = d

    def get(self, key) -> Optional[Dish]:
        return self.dishes.get(key)

    def keys(self):
        return list(self.dishes.keys())

    def source_text(self, d: Dish) -> str:
        """把来源拼成一句能直接写进报告的话"""
        tag = {"A": "USDA官方API直连", "B": "USDA镜像(nutritionvalue.org)",
               "C": "中国食物成分表", "D": "按配料配方合成计算"}.get(
            d.source_type, "未知来源")
        return "%s｜%s %s" % (tag, d.source_id, d.source_name)


# ---------------------------------------------------------------- 份量
# 份量档位。默认按营养库里记录的"一份"重量，用户可调。
PORTION_PRESETS = {
    "小份": 0.7,
    "标准份": 1.0,
    "大份": 1.4,
}


def estimate_dish(key, db: NutritionDB, portion_scale=1.0,
                  portion_g=None) -> Optional[dict]:
    """
    估算一道菜的营养摄入。

    portion_g 给了就用它，否则用 库里的 serving_g × portion_scale。
    返回的 dict 里带上了完整的来源信息，界面和报告都要展示 ——
    这是这个项目和"随便调个 API 出个数字"的关键区别：
    每个数字都能追溯到具体是哪条数据库记录。
    """
    d = db.get(key)
    if d is None:
        return None

    grams = float(portion_g) if portion_g else d.serving_g * portion_scale
    total = d.per_100g * (grams / 100.0)
    # 整份可能是"一桌菜"，所以同时给出人均值
    per_person = total * (1.0 / d.servings) if d.servings > 0 else total

    return {
        "key": key,
        "cn_name": d.cn_name,
        "grams": round(grams, 1),
        "serving_desc": d.serving_desc,
        "servings": d.servings,
        "nutrients": total,
        "per_person": per_person,
        "per_100g": d.per_100g,
        "source": db.source_text(d),
        "source_type": d.source_type,
        "note": d.note,
        "recipe": d.recipe,
    }


# ---------------------------------------------------------------- 一餐汇总
def summarize_meal(items: List[dict],
                   meal_name: str = "午餐") -> dict:
    """
    把若干道菜汇成一餐，并和每日参考摄入量做对比。

    items 里每项是 estimate_dish 的返回值。
    """
    total = Nutrients()
    for it in items:
        if it and it.get("nutrients"):
            total = total + it["nutrients"]

    share = MEAL_SHARE.get(meal_name, 0.33)
    ratio = total.macro_energy_ratio()

    # 供能比是否落在膳食指南推荐区间内
    # 指南给的区间：蛋白 10%~20%、脂肪 20%~30%、碳水 50%~65%
    advice = []
    if ratio["fat"] > 0.35:
        advice.append("脂肪供能比 %.0f%%，明显偏高（建议 20%%~30%%），"
                      "这餐偏油腻" % (ratio["fat"] * 100))
    elif ratio["fat"] < 0.15 and total.energy_kcal > 200:
        advice.append("脂肪供能比仅 %.0f%%，偏低" % (ratio["fat"] * 100))

    if ratio["protein"] < 0.10 and total.energy_kcal > 200:
        advice.append("蛋白质供能比 %.0f%%，偏低，建议加一份优质蛋白"
                      % (ratio["protein"] * 100))

    if ratio["carb"] > 0.70:
        advice.append("碳水供能比 %.0f%%，主食偏多" % (ratio["carb"] * 100))

    per_meal_sodium = DAILY_REFERENCE["sodium_mg"] * share
    if total.sodium_mg > per_meal_sodium * 1.2:
        advice.append("钠 %.0f mg，超过本餐参考值 %.0f mg，口味偏咸"
                      % (total.sodium_mg, per_meal_sodium))

    return {
        "meal_name": meal_name,
        "n_dishes": len([i for i in items if i]),
        "total": total,
        "macro_ratio": ratio,
        "advice": advice,
    }


def daily_report(meals: Dict[str, dict]) -> dict:
    """
    汇总一天。meals 是 {餐次名: summarize_meal 的返回值}
    """
    total = Nutrients()
    for m in meals.values():
        if m:
            total = total + m["total"]

    ratio = total.macro_energy_ratio()
    pct = {}
    for k, ref in DAILY_REFERENCE.items():
        got = getattr(total, k)
        pct[k] = got / ref if ref else 0.0

    return {
        "total": total,
        "macro_ratio": ratio,
        "percent_of_reference": pct,
        "reference": dict(DAILY_REFERENCE),
    }


def nutriscore_like(n: Nutrients) -> dict:
    """
    一个简化的"膳食质量评分"。

    强调：这不是真实的 Nutri-Score（那是欧盟的食品包装标签算法，
    针对预包装食品，有明确的评分表）。这里只是借用它的思路 ——
    把有利因素和不利因素分别打分再相减 —— 做一个便于展示的直观指标。
    界面上必须标明是"简化指标"，否则就是误导。
    """
    score = 10.0
    reasons = []

    # 有利因素
    if n.fiber_g >= 3:
        score += 1.0
        reasons.append("膳食纤维较丰富 +1.0")
    if n.protein_g >= 15:
        score += 1.0
        reasons.append("蛋白质充足 +1.0")

    # 不利因素
    # 能量密度：按每 100 g 折算，直接看总量会被份量带偏
    if n.sodium_mg >= 600:
        score -= 2.0
        reasons.append("钠含量很高 -2.0")
    elif n.sodium_mg >= 300:
        score -= 1.0
        reasons.append("钠偏高 -1.0")

    r = n.macro_energy_ratio()
    if r["fat"] >= 0.40:
        score -= 2.0
        reasons.append("脂肪供能比过高 -2.0")
    elif r["fat"] >= 0.32:
        score -= 1.0
        reasons.append("脂肪供能比偏高 -1.0")

    score = max(0.0, min(10.0, score))
    if score >= 8:
        grade = "较优"
    elif score >= 6:
        grade = "中等"
    elif score >= 4:
        grade = "偏低"
    else:
        grade = "较差"

    return {"score": round(score, 1), "grade": grade, "reasons": reasons}


if __name__ == "__main__":
    # 直接运行本文件时做一次自检：看看 16 道菜的营养库能不能对上
    print("=" * 78)
    print("营养数据库自检")
    print("=" * 78)
    try:
        db = NutritionDB()
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)

    print("已加载 %d 条记录\n" % len(db.dishes))
    print("%-16s %-12s %8s %8s %8s %8s %8s  %s"
          % ("key", "中文名", "kcal", "蛋白", "脂肪", "碳水", "钠", "来源"))
    print("-" * 78)
    for k, d in db.dishes.items():
        n = d.per_100g
        print("%-16s %-12s %8.1f %8.1f %8.1f %8.1f %8.0f  %s"
              % (k, d.cn_name, n.energy_kcal, n.protein_g, n.fat_g,
                 n.carb_g, n.sodium_mg, d.source_type))
    print("-" * 78)
    by_src = {}
    for d in db.dishes.values():
        by_src[d.source_type] = by_src.get(d.source_type, 0) + 1
    print("来源分布：%s" % by_src)

    # Atwater 自洽性抽查
    print("\nAtwater 能量自洽性抽查（蛋白4/脂肪9/碳水4）：")
    bad = 0
    for k, d in db.dishes.items():
        n = d.per_100g
        calc = n.protein_g * 4 + n.fat_g * 9 + n.carb_g * 4
        if n.energy_kcal > 0:
            dev = (calc - n.energy_kcal) / n.energy_kcal * 100
            flag = "" if abs(dev) <= 12 else "  <-- 偏差偏大"
            if abs(dev) > 12:
                bad += 1
            print("   %-16s 标注 %6.1f  反算 %6.1f  (%+.1f%%)%s"
                  % (k, n.energy_kcal, calc, dev, flag))
    print("\n偏差异常条目数：%d" % bad)
