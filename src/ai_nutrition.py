"""
ai_nutrition.py —— 用 AI 给出的配料估算营养。

这个模块解决的是：AI 认出了一道本地营养库里没有的菜（比如"番茄炒蛋"），
营养数据从哪来。

做法**刻意和本地 16 道菜保持一致**：
    AI 给出配料和克重 → 映射到中国食物成分表的配料营养值 →
    按同一个公式加权求和。

    每 100 g 营养值 = Σ(配料重量 × 配料每 100 g 营养值) ÷ 成品总重 × 100

为什么不让 AI 直接报热量数字：
    那样得到的数字无从核查，而且会和本地那批可溯源的数据混在一起，
    把整个项目好不容易建立起来的可信度拉低。
    走"AI 估配料 + 本地权威值计算"这条路，至少营养值是查表得来的，
    不确定性被限制在"配料克重"这一个环节上，可以明确标注。
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from nutrition import Nutrients

# 配料名匹配的最低相似度。
# 0.75 是权衡后的取值：太低会误配（"猪肉"配到"猪油"），
# 太高则大量正常表述匹配不上。配不上的后果只是这一项不计入 + 明确标注，
# 比错配安全得多，所以宁可定高一点。
SIMILARITY_THRESHOLD = 0.75


# ---------------------------------------------------------------- 同义词表
# AI 说的是日常叫法，而中国食物成分表用的是规范名，两边常常对不上：
#   "食用油" vs "豆油"、"生抽" vs "酱油"、"料酒" vs "黄酒"、"白糖" vs "白砂糖"
# 这类是**同物异名**，不是相似度问题，靠字符串匹配永远救不了，
# 所以显式列出来。
#
# 注意这里是**精确匹配**（键必须与 AI 的说法完全一致），
# 不做模糊 —— 同义词表一旦模糊起来，很容易把"猪肉"配到"猪油"。
ING_ALIAS = {
    # 油
    "食用油": "cooking_oil", "植物油": "cooking_oil", "菜油": "cooking_oil",
    "花生油": "cooking_oil", "色拉油": "cooking_oil", "油": "cooking_oil",
    "香油": "sesame_oil", "芝麻油": "sesame_oil",
    "红油": "chili_oil", "辣椒油": "chili_oil",
    # 酱油类
    "生抽": "soy_sauce", "老抽": "soy_sauce", "豉油": "soy_sauce",
    "蒸鱼豉油": "soy_sauce",
    # 糖
    "白糖": "sugar", "白砂糖": "sugar", "砂糖": "sugar",
    "冰糖": "sugar", "糖": "sugar",
    # 酒
    "料酒": "cooking_wine", "黄酒": "cooking_wine", "绍酒": "cooking_wine",
    # 淀粉
    "生粉": "starch", "水淀粉": "starch", "淀粉": "starch",
    "玉米淀粉": "starch", "红薯淀粉": "starch",
    # 盐
    "盐": "salt", "食盐": "salt", "精盐": "salt",
    # 猪肉
    "猪里脊肉": "pork_lean", "里脊肉": "pork_lean", "猪瘦肉": "pork_lean",
    "瘦肉": "pork_lean", "猪里脊": "pork_lean",
    "五花肉": "pork_belly", "猪五花肉": "pork_belly", "带皮五花肉": "pork_belly",
    "猪绞肉": "pork_belly", "猪肉末": "pork_belly", "猪肉馅": "pork_belly",
    "猪肉": "pork_belly",
    "排骨": "pork_rib", "猪排": "pork_rib",
    # 鸡
    "鸡肉": "chicken_whole", "鸡腿肉": "chicken_whole", "整鸡": "chicken_whole",
    "鸡胸肉": "chicken_breast", "鸡脯肉": "chicken_breast",
    # 牛
    "牛腱子": "beef_shank", "牛腱": "beef_shank", "牛肉": "beef_shank",
    "牛肉末": "beef_shank", "牛腩": "beef_shank",
    # 水产
    "三文鱼": "fish_salmon", "鲑鱼": "fish_salmon",
    "鱿鱼": "squid", "干贝": "scallop_dried", "瑶柱": "scallop_dried",
    "虾米": "shrimp_dried", "海米": "shrimp_dried",
    # 蛋
    "鸡蛋": "egg", "蛋": "egg", "鸡蛋液": "egg", "蛋液": "egg",
    "鸡蛋清": "egg", "蛋清": "egg",
    # 蔬菜
    "西红柿": "tomato", "土豆": "potato", "马铃薯": "potato",
    "青椒": "green_pepper", "尖椒": "green_pepper", "柿子椒": "green_pepper",
    "青辣椒": "green_pepper", "杭椒": "green_pepper",
    # 红椒/彩椒/甜椒：食物成分表里是"甜椒[灯笼椒，柿子椒]"（id 410）。
    # AI 一般说"红椒"，靠它匹配
    "红椒": "bell_pepper", "彩椒": "bell_pepper", "甜椒": "bell_pepper",
    "灯笼椒": "bell_pepper", "红甜椒": "bell_pepper", "黄椒": "bell_pepper",
    "黄瓜": "cucumber", "胡萝卜": "carrot", "竹笋": "bamboo_shoot",    "冬笋": "bamboo_shoot", "春笋": "bamboo_shoot",
    "木耳": "wood_ear", "黑木耳": "wood_ear",
    "莲藕": "lotus_root", "藕": "lotus_root",
    "荸荠": "water_chestnut", "马蹄": "water_chestnut",
    "大蒜": "garlic", "蒜": "garlic", "蒜瓣": "garlic", "蒜末": "garlic",
    "生姜": "ginger", "姜": "ginger", "姜末": "ginger", "姜片": "ginger",
    "葱": "scallion", "大葱": "scallion", "小葱": "scallion",
    "香葱": "scallion", "蒜苗": "scallion", "青蒜": "scallion",
    "香菜": "cilantro", "芫荽": "cilantro",
    # 豆制品
    "豆腐": "tofu", "嫩豆腐": "tofu", "老豆腐": "tofu", "北豆腐": "tofu",
    "豆腐皮": "tofu_skin", "油皮": "tofu_skin",
    # 坚果
    "花生": "peanut", "花生米": "peanut", "花生仁": "peanut",
    "芝麻": "sesame", "白芝麻": "sesame", "黑芝麻": "sesame",
    "核桃": "walnut", "核桃仁": "walnut",
    # 其他调味
    "豆瓣酱": "doubanjiang", "郫县豆瓣": "doubanjiang", "豆瓣": "doubanjiang",
    "甜面酱": "sweet_bean_sauce",
    "蚝油": "oyster_sauce",
    "醋": "vinegar", "香醋": "vinegar", "米醋": "vinegar",
    "陈醋": "vinegar", "白醋": "vinegar",
    "鸡精": "chicken_essence", "鸡粉": "chicken_essence",
    "味精": "chicken_essence",
    "面粉": "flour", "小麦粉": "flour",
    "米饭": "rice", "白米饭": "rice",
}


@dataclass
class EstIngredient:
    """一个配料的估算结果"""
    ai_name: str                 # AI 报出来的名字
    grams: float                 # AI 报的克重
    matched_key: Optional[str] = None   # 匹配到的配料表 key
    matched_cn: str = ""                # 匹配到的规范中文名
    matched_by: str = ""                # 靠什么匹配上的（用于排查）
    nutrients: Optional[Nutrients] = None   # 这一项的绝对营养量
    skipped: bool = False        # 属于"无热量、本就不该计入"的项（如水）


# 这些配料没有热量和营养素，计不计入结果一样，但 AI 常会把它们写进配方。
# 如果当成"未匹配"报给用户，会让人误以为营养算漏了 ——
# 所以单独识别出来静默跳过，只在说明里提一句"水不计入"。
NO_NUTRIENT_NAMES = {
    "清水", "水", "凉水", "温水", "热水", "开水", "纯净水", "自来水",
    "冰水", "高汤", "鲜汤", "清汤", "鸡汤", "骨汤", "煮肉水",
    "荷叶", "粽叶", "牙签", "保鲜膜",
}


@dataclass
class AIEstimate:
    """一道菜的 AI 估算结果"""
    dish: str
    total_g: float
    ingredients: List[EstIngredient] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)   # 没匹配上的配料名
    per_100g: Optional[Nutrients] = None
    ok: bool = False
    message: str = ""
    # 配料的加总重量，用来判断 AI 给的克重是否自相矛盾
    sum_g: float = 0.0


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_ingredient(name: str, ing_table: Dict) -> Tuple[Optional[str], str, str]:
    """
    把 AI 报的配料名映射到配料营养表的 key。

    返回 (key, 规范中文名, 匹配方式)。

    匹配顺序：同义词表 -> 完全相等 -> 包含 -> 相似度。

    同义词表放在最前面，因为 AI 说的是日常叫法（"食用油""生抽""料酒"），
    而食物成分表用规范名（"豆油""酱油""黄酒"），
    这类同物异名靠字符串匹配永远救不了，必须显式列出。

    **任何一步都不允许"拿相近的东西顶替"**。相似度匹配最容易出事，
    所以阈值定得高、且限定两边长度接近，并把匹配方式记下来 ——
    界面上可以把"这项是靠相似度猜的"单独标出来给用户判断。
    """
    if not name:
        return None, "", ""

    q = name.strip()

    # 零轮：同义词表（精确匹配，不做模糊）
    key = ING_ALIAS.get(q)
    if key and key in ing_table:
        return key, ing_table[key][0], "同义词表"

    # 一轮：规范中文名完全相等
    for key, item in ing_table.items():
        if item[0] == q:
            return key, item[0], "完全匹配"

    # 二轮：包含关系，取最长的那条（长的更具体，不容易配错）
    best, best_cn, best_len = None, "", 0
    for key, item in ing_table.items():
        cn = item[0]
        if len(cn) < 2:
            continue
        if cn in q or q in cn:
            if len(cn) > best_len:
                best, best_cn, best_len = key, cn, len(cn)
    if best:
        return best, best_cn, "包含匹配"

    # 三轮：相似度。只在两边长度接近时才算，避免"油"配到"猪油"
    best, best_cn, best_score = None, "", 0.0
    for key, item in ing_table.items():
        cn = item[0]
        if len(cn) < 2:
            continue
        if abs(len(cn) - len(q)) > 2:
            continue
        s = _similarity(q, cn)
        if s > best_score:
            best, best_cn, best_score = key, cn, s
    if best and best_score >= SIMILARITY_THRESHOLD:
        return best, best_cn, "相似度%.2f" % best_score

    return None, "", ""


def estimate_from_ingredients(dish: str, ingredients: List[Dict],
                              total_g: Optional[float],
                              ing_table: Dict = None) -> AIEstimate:
    """
    按配料算营养。

    ingredients: [{"name": "猪里脊肉", "grams": 200}, ...]
    total_g:     成品总重（克）。AI 没给就用配料加总当分母。
    """
    if ing_table is None:
        from build_nutrition_db import load_ingredients
        ing_table = load_ingredients()
    if not ing_table:
        return AIEstimate(dish=dish, total_g=0.0, ok=False,
                          message="配料营养表读取失败。")

    if not ingredients:
        return AIEstimate(dish=dish, total_g=0.0, ok=False,
                          message="没能从 AI 的回答里解析出配料。")

    acc = Nutrients()
    result = []
    unmatched = []
    skipped = []
    sum_g = 0.0

    for it in ingredients:
        name = (it.get("name") or "").strip()
        grams = float(it.get("grams") or 0)
        if grams <= 0:
            continue
        sum_g += grams

        # 无热量配料（水、汤等）静默跳过。
        # 注意 sum_g 里已经算进了它们 —— 这是对的，
        # 因为成品总重本来就包含水，分母不该把它们排除。
        if name in NO_NUTRIENT_NAMES:
            skipped.append(name)
            result.append(EstIngredient(ai_name=name, grams=grams,
                                        skipped=True))
            continue

        key, cn, how = match_ingredient(name, ing_table)
        if key is None:
            unmatched.append(name)
            result.append(EstIngredient(ai_name=name, grams=grams))
            continue

        item = ing_table[key]
        # item = (中文名, id, kcal, 蛋白, 脂肪, 碳水, 纤维, 钠)
        _, _, kcal, p, f, c, fib, na = item
        k = grams / 100.0
        nut = Nutrients(
            energy_kcal=(kcal or 0) * k,
            protein_g=(p or 0) * k,
            fat_g=(f or 0) * k,
            carb_g=(c or 0) * k,
            fiber_g=(fib or 0) * k,
            sodium_mg=(na or 0) * k,
        )
        acc = acc + nut
        result.append(EstIngredient(ai_name=name, grams=grams,
                                    matched_key=key, matched_cn=cn,
                                    matched_by=how, nutrients=nut))

    # 分母
    if not total_g or total_g <= 0:
        total_g = sum_g
    if total_g <= 0:
        return AIEstimate(dish=dish, total_g=0.0,
                          ingredients=result, unmatched=unmatched,
                          ok=False, message="配料总重为 0，无法计算。",
                          sum_g=sum_g)

    per100 = acc * (100.0 / total_g)

    # 用一个宽松的检查发现明显反常的结果。
    # 热量密度 900 kcal/100g 是纯油脂的水平，正常菜品不可能超过；
    # 低于 5 kcal/100g 说明配料基本没匹配上。
    if per100.energy_kcal > 900:
        return AIEstimate(dish=dish, total_g=total_g, ingredients=result,
                          unmatched=unmatched, per_100g=per100, ok=False,
                          sum_g=sum_g,
                          message="算出的热量密度异常高（超过纯油脂水平），"
                                  "AI 给的配料克重可能有问题。")
    if per100.energy_kcal < 5:
        return AIEstimate(dish=dish, total_g=total_g, ingredients=result,
                          unmatched=unmatched, per_100g=per100, ok=False,
                          sum_g=sum_g,
                          message="算出的热量密度异常低，"
                                  "可能配料都没匹配上。")

    return AIEstimate(dish=dish, total_g=total_g, ingredients=result,
                      unmatched=unmatched, per_100g=per100, ok=True,
                      sum_g=sum_g,
                      message="")


def to_estimate_dict(est: AIEstimate, keys=None) -> dict:
    """
    转成和 nutrition.estimate_dish 相同结构的 dict，
    这样界面和「膳食管家」页不用区分数据来源就能统一处理。

    关键区别在 source_type：本地库是 C/D，AI 估算是 "AI"。
    界面靠这个字段决定要不要打"AI 估算"的标记。
    """
    n = est.per_100g or Nutrients()
    total = n * (est.total_g / 100.0)

    # 配料表：给用户看"这道菜由什么构成"
    recipe = {}
    for it in est.ingredients:
        if it.skipped:
            # 水之类不用列进配方，会显得很啰嗦
            continue
        label = it.matched_cn or it.ai_name
        if it.matched_key is None:
            label = "%s（未能匹配，未计入）" % it.ai_name
        recipe[label] = it.grams

    note_parts = [
        "配料克重由 AI 估算，不是权威数据。热量是按中国食物成分表的"
        "配料营养值加权计算，算法与内置的 16 道菜一致。",
    ]
    if est.unmatched:
        note_parts.append(
            "有 %d 项配料在食物成分表中查不到对应条目，未计入：%s"
            % (len(est.unmatched), "、".join(est.unmatched)))
    if abs(est.sum_g - est.total_g) > max(1.0, est.total_g * 0.02):
        note_parts.append(
            "配料加总 %.0f g 与成品总重 %.0f g 不一致，"
            "计算时以成品总重为分母。" % (est.sum_g, est.total_g))

    return {
        "key": "ai:%s" % est.dish,
        "cn_name": est.dish,
        "grams": round(est.total_g, 1),
        "serving_desc": "一份（AI 估算，约 %.0f g）" % est.total_g,
        "servings": 1.0,
        "nutrients": total,
        "per_person": total,
        "per_100g": n,
        # source_type 用 "AI" 标记，界面和报告据此区分数据来源等级
        "source_type": "AI",
        "source": "AI 估值｜配料由 %s 估算，营养值取自中国食物成分表"
                  % est.dish,
        "note": " ".join(note_parts),
        "recipe": recipe,
        "is_ai_estimated": True,
    }
