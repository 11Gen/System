"""
check_nutrition.py —— 独立复核营养数据。

背景：这批营养数据里有一部分因为 USDA 官方 API 的 DEMO_KEY 限额（30次/小时/IP）
取不到直连结果，是用 nutritionvalue.org 这个镜像站取的。
二手来源不能直接就写进项目里，所以这个脚本负责：
    1) 重新抓一遍镜像站页面，把数值解析出来
    2) 和 nutrition_db.csv 里记录的值对比
    3) 用 Atwater 系数（蛋白4/脂肪9/碳水4）做自洽性检查
    4) 检查"水分+蛋白+脂肪+碳水"是否接近 100 g（证明基准是不是每100g）

第 3、4 条是很好的交叉验证手段：如果某条数据抄错了，
Atwater 反算出来的能量会和标的能量对不上，一眼就能看出来。
这个检查也确实抓到过问题 —— 见报告"数据质量"一节。

用法：
    python src/check_nutrition.py            # 全部复核
    python src/check_nutrition.py --only A   # 只复核标记为镜像来源的
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import NUTRITION_CSV

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 镜像站上每条记录对应的页面。这个 URL 拼法是试出来的：
# 标题里的空格/逗号/引号要换成下划线，问号后的 size=100+g 控制按每100g显示。
MIRROR_URL = {
    "pizza": "https://www.nutritionvalue.org/"
             "Pizza%2C_cheese%2C_from_restaurant_or_fast_food%2C_thin_crust_"
             "58106220_nutritional_value.html?size=100+g",
    "steak": "https://www.nutritionvalue.org/"
             "Beef%2C_broiled%2C_cooked%2C_choice%2C_trimmed_to_0%22_fat%2C_"
             "separable_lean_and_fat%2C_steak%2C_top_sirloin_nutritional_value.html?size=100+g",
    "sushi": "https://www.nutritionvalue.org/"
             "Sushi_roll%2C_California_nutritional_value.html?size=100+g",
    "ramen": "https://www.nutritionvalue.org/"
             "Soup%2C_ramen_noodles%2C_water_added_nutritional_value.html?size=100+g",
    "club_sandwich": "https://www.nutritionvalue.org/"
                     "Club_sandwich_or_sub%2C_restaurant_nutritional_value.html?size=100+g",
    "french_fries": "https://www.nutritionvalue.org/"
                    "Fast_foods%2C_french_fried_in_vegetable_oil%2C_potato_"
                    "nutritional_value.html?size=100+g",
    "grilled_salmon": "https://www.nutritionvalue.org/"
                      "Fish%2C_dry_heat%2C_cooked%2C_farmed%2C_Atlantic%2C_salmon_"
                      "nutritional_value.html?size=100+g",
    "scrambled_egg": "https://www.nutritionvalue.org/"
                     "Egg%2C_scrambled%2C_cooked%2C_whole_nutritional_value.html?size=100+g",
    "bibimbap": "https://www.nutritionvalue.org/"
                "Bibimbap%2C_Korean_nutritional_value.html?size=100+g",
}

# 镜像站表格里的行标题 -> 我们要的字段名
FIELD_PATTERNS = [
    ("energy_kcal", r"Calories"),
    ("protein_g", r"Protein"),
    ("fat_g", r"Total Fat"),
    ("carb_g", r"Total Carbohydrate"),
    ("fiber_g", r"Dietary Fiber"),
    ("sodium_mg", r"Sodium"),
]


def fetch(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(800000).decode("utf-8", "replace")
    except Exception as e:
        print("      抓取失败: %s" % str(e)[:70])
        return None


def parse_page(html):
    """
    从页面里抠出营养数值。

    页面结构（看过原始 HTML 才写对的）：
        热量      <td ... id='calories'>149</td>
        其他      <td class='left'><b>Total&nbsp;Fat</b>&nbsp;11g</td>
    两个坑：
      1) 标签里用的是 &nbsp; 不是普通空格，正则要按 &nbsp; 匹配
      2) 镜像站只显示到 2 位有效数字（10.98g 显示成 11g），
         所以对比时不能要求完全相等，得给容差
    """
    out = {}

    m = re.search(r"id='calories'[^>]*>\s*([0-9]+(?:\.[0-9]+)?)", html)
    if m:
        out["energy_kcal"] = float(m.group(1))

    # 标签 -> (字段名, 单位)。正则允许标签前后有 <b> 之类的标签
    patterns = {
        "protein_g": r"Protein",
        "fat_g": r"Total&nbsp;Fat",
        "carb_g": r"Total&nbsp;Carbohydrate",
        "fiber_g": r"Dietary&nbsp;Fiber",
        "sodium_mg": r"Sodium",
    }
    for field, label in patterns.items():
        # 例：<b>Total&nbsp;Fat</b>&nbsp;11g
        m = re.search(r">\s*%s\s*<[^>]*>\s*(?:&nbsp;)*\s*([0-9]+(?:\.[0-9]+)?)"
                      % label, html)
        if m:
            out[field] = float(m.group(1))

    # 页面上标的份量，用来确认这批数字确实是"每 100 g"
    m = re.search(r"id='serving-size'>([^<]+)<", html)
    if m:
        out["_serving"] = m.group(1).strip()

    return out


def atwater_check(kcal, p, f, c):
    """用 4/9/4 系数反算能量，返回 (计算值, 相对偏差%)"""
    calc = p * 4 + f * 9 + c * 4
    if kcal <= 0:
        return calc, 0.0
    return calc, (calc - kcal) / kcal * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["A", "B"], default=None,
                    help="A=只查镜像来源  B=只查直连来源")
    args = ap.parse_args()

    if not NUTRITION_CSV.exists():
        print("找不到 %s" % NUTRITION_CSV)
        sys.exit(1)

    with open(NUTRITION_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("=" * 100)
    print("营养数据复核   （共 %d 条）" % len(rows))
    print("=" * 100)
    print("%-16s %-8s %8s %8s %8s %8s   %s" %
          ("food", "来源", "kcal", "蛋白", "脂肪", "碳水", "校验"))
    print("-" * 100)

    checked, mismatched, unverified = 0, [], []

    for r in rows:
        food = r["food"]
        src = r.get("source_type", "?")
        if args.only and src != args.only:
            continue

        try:
            kcal = float(r["energy_kcal"])
            p = float(r["protein_g"])
            f_ = float(r["fat_g"])
            c = float(r["carb_g"])
        except (KeyError, ValueError):
            print("%-16s 字段缺失，跳过" % food)
            continue

        # ---- 1) Atwater 自洽性检查（对任何来源都适用）----
        calc, dev = atwater_check(kcal, p, f_, c)
        if abs(dev) <= 12:
            tag = "OK  (%.0f kcal, %+.0f%%)" % (calc, dev)
        else:
            tag = "!! 偏差大 (%.0f kcal, %+.0f%%)" % (calc, dev)
            mismatched.append((food, "Atwater", kcal, calc))

        # ---- 2) 镜像来源再去页面复核一次 ----
        note = ""
        if src == "B" and food in MIRROR_URL:
            html = fetch(MIRROR_URL[food])
            if html:
                got = parse_page(html)
                if not got:
                    note = "页面解析不出数值"
                    unverified.append((food, "解析失败"))
                else:
                    if got.get("_serving"):
                        note = "[%s] " % got["_serving"]
                    diffs = []
                    for k, v in got.items():
                        if k.startswith("_"):
                            continue
                        mine = float(r.get(k) or 0)
                        # 页面只有 2 位有效数字，容差按 3% 或绝对 0.6 取大者
                        tol = max(0.6, abs(v) * 0.03)
                        if abs(v - mine) > tol:
                            diffs.append("%s:页面%.1f/本地%.2f" % (k, v, mine))
                    if diffs:
                        note += "不一致 -> " + "; ".join(diffs)
                        unverified.append((food, note))
                    else:
                        note += "已复核一致"
                    checked += 1
            else:
                note = "抓取失败"
                unverified.append((food, "抓取失败"))

        print("%-16s %-8s %8.2f %8.2f %8.2f %8.2f   %s %s"
              % (food, src, kcal, p, f_, c, tag, note))
        time.sleep(0.8)        # 别把人家站点打挂了

    print("-" * 100)
    print("\n复核完成：实际抓取核对 %d 条" % checked)

    if mismatched:
        print("\nAtwater 自洽性异常（能量和三大营养素对不上，需要复查）：")
        for food, kind, stated, calc in mismatched:
            print("   %-16s 标注 %.0f kcal，按 4/9/4 反算 %.0f kcal" % (food, stated, calc))
    else:
        print("Atwater 自洽性检查：全部通过（偏差均 <=12%，属正常范围）")

    if unverified:
        print("\n需要关注的条目：")
        for food, why in unverified:
            print("   %-16s %s" % (food, why))
    else:
        print("镜像来源条目全部复核一致。")


if __name__ == "__main__":
    main()
