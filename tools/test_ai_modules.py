"""临时测试：AI 识别模块的解析与容错逻辑（不联网）

视觉 API 的关键风险不在"能不能调通"，而在"模型返回的东西不符合预期时会不会崩"。
模型经常加上 ```json 代码块、多写一句话、或者干脆不按格式返回，
这些情况必须都能处理。这里用假数据把这些路径都走一遍。
"""
import sys

sys.path.insert(0, "src")

from ai_nutrition import estimate_from_ingredients, match_ingredient, to_estimate_dict
from build_nutrition_db import load_ingredients
from vision_api import _extract_json, image_to_data_url, match_known_dish

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print("  [通过] %s" % name)
    else:
        fail += 1
        print("  [失败] %s  %s" % (name, detail))


print("=" * 74)
print("一、模型输出的 JSON 解析（最容易出问题的地方）")
print("=" * 74)

cases = [
    ("标准 JSON", '{"name": "宫保鸡丁", "confidence": 0.9}',
     lambda d: d and d.get("name") == "宫保鸡丁"),
    ("被 ```json 包裹", '```json\n{"name": "麻婆豆腐", "confidence": 0.8}\n```',
     lambda d: d and d.get("name") == "麻婆豆腐"),
    ("被 ``` 包裹（无语言标记）",
     '```\n{"name": "水煮鱼"}\n```',
     lambda d: d and d.get("name") == "水煮鱼"),
    ("前后有多余解释",
     '好的，我识别出这是：\n{"name": "回锅肉", "confidence": 0.75}\n希望有帮助！',
     lambda d: d and d.get("name") == "回锅肉"),
    ("完全没有 JSON", '这是一道川菜，看起来像宫保鸡丁。',
     lambda d: d is None),
    ("JSON 语法错误", '{"name": "宫保鸡丁", "confidence":}',
     lambda d: d is None),
    ("空字符串", '', lambda d: d is None),
    ("嵌套 JSON（配料格式）",
     '{"ingredients": [{"name": "鸡蛋", "grams": 150}], "total_g": 300}',
     lambda d: d and len(d.get("ingredients", [])) == 1),
]
for name, raw, pred in cases:
    got = _extract_json(raw)
    check(name, pred(got), "解析结果=%r" % (got,))

print()
print("=" * 74)
print("二、配料名匹配")
print("=" * 74)
ing = load_ingredients()
match_cases = [
    ("鸡蛋", "egg"), ("食用油", "cooking_oil"), ("生抽", "soy_sauce"),
    ("猪里脊肉", "pork_lean"), ("五花肉", "pork_belly"), ("料酒", "cooking_wine"),
    ("白糖", "sugar"), ("淀粉", "starch"), ("盐", "salt"),
    ("红椒", "bell_pepper"), ("菠萝", "pineapple"), ("番茄", "tomato"),
    ("完全不认识的东西", None), ("", None),
]
for name, expect in match_cases:
    k, cn, how = match_ingredient(name, ing)
    check("「%s」-> %s" % (name or "(空)", expect or "不匹配"),
          k == expect, "实际得到 %r" % k)

print()
print("=" * 74)
print("三、营养估算的边界情况")
print("=" * 74)

# 正常
e = estimate_from_ingredients("测试菜", [{"name": "鸡蛋", "grams": 100}], 200)
check("正常输入能算出结果", e.ok and e.per_100g is not None)
check("  每100g 热量为正", e.per_100g.energy_kcal > 0)

# 空配料
e = estimate_from_ingredients("测试菜", [], 200)
check("空配料列表 -> 失败但不崩", not e.ok and "解析" in e.message)

# 配料全不认识
e = estimate_from_ingredients("测试菜", [{"name": "外星食材", "grams": 100}], 200)
check("配料全不认识 -> 判定异常并给出原因",
      not e.ok and "异常低" in e.message)
check("  未匹配项被记录", "外星食材" in e.unmatched)

# 克重为 0 / 负数
e = estimate_from_ingredients("测试菜", [{"name": "鸡蛋", "grams": 0}], 100)
check("克重为0 -> 不崩", not e.ok)

# 总重缺失 -> 用配料加总当分母
e = estimate_from_ingredients("测试菜", [{"name": "鸡蛋", "grams": 100}], None)
check("总重缺失时用配料加总当分母", e.ok and abs(e.total_g - 100) < 0.01)

# 热量密度异常高（纯油水平）
e = estimate_from_ingredients(
    "测试菜", [{"name": "食用油", "grams": 950}, {"name": "盐", "grams": 50}], 100)
check("热量密度超过纯油脂 -> 报异常", not e.ok and "异常高" in e.message)

# 水应当被静默跳过
e = estimate_from_ingredients(
    "测试菜", [{"name": "鸡蛋", "grams": 100}, {"name": "清水", "grams": 500}], 600)
check("清水被静默跳过（不算未匹配）", "清水" not in e.unmatched)
check("  但水的重量仍计入分母", abs(e.sum_g - 600) < 0.01)

# 转成界面用的 dict
d = to_estimate_dict(e)
check("转 dict 后 source_type 是 AI", d["source_type"] == "AI")
check("转 dict 后带 is_ai_estimated 标记", d.get("is_ai_estimated") is True)
check("配方里不含水", not any("清水" in k or "水" == k for k in d["recipe"]))

print()
print("=" * 74)
print("四、本地菜名匹配（决定要不要走 AI 估算营养）")
print("=" * 74)
import config
keys = list(config.CLASSES)
name_cases = [
    ("宫保鸡丁", "Kung_Pao_Chicken"),
    ("麻婆豆腐", "Mapo_Tofu"),
    ("水煮鱼", "Boiled_Fish_with_Sichuan_Peppercorns"),
    ("番茄炒蛋", None),          # 不在 16 道里
    ("北京烤鸭", "Peking_Duck"),
    ("", None),
]
for name, expect in name_cases:
    got = match_known_dish(name, keys)
    check("「%s」-> %s" % (name or "(空)", expect or "不匹配（走 AI）"),
          got == expect, "实际得到 %r" % got)

print()
print("=" * 74)
print("五、图片转 base64")
print("=" * 74)
from PIL import Image
im = Image.new("RGB", (3000, 2000), (200, 100, 50))
url = image_to_data_url(im)
check("返回 data URL 格式", url.startswith("data:image/jpeg;base64,"))
import base64
from io import BytesIO
raw = base64.b64decode(url.split(",", 1)[1])
out = Image.open(BytesIO(raw))
check("大图被缩到长边 1024 以内", max(out.size) <= 1024,
      "实际尺寸 %s" % (out.size,))
check("长宽比保持不变", abs(out.size[0] / out.size[1] - 1.5) < 0.02,
      "实际比例 %.3f" % (out.size[0] / out.size[1]))
check("体积远低于各家 32 MiB 上限", len(raw) < 2 * 1024 * 1024,
      "实际 %.1f KB" % (len(raw) / 1024))

print()
print("=" * 74)
print("结果：%d 项通过，%d 项失败" % (ok, fail))
print("=" * 74)

import os
sys.stdout.flush()
os._exit(1 if fail else 0)
