"""
fetch_cn_nutrition.py —— 从中国疾控中心营养与健康所官方平台取中式菜品营养数据。

数据源：https://nlc.chinanutri.cn/fq/   中国食物成分表在线查询
    主办单位：中国疾病预防控制中心营养与健康所

为什么单独写一个脚本而不是手抄：
    1) 该平台能量只印 kJ、不印 kcal，手工换算容易出错，脚本统一按 ÷4.184 换算
    2) 需要留痕。报告里要说明每个数字从哪来，脚本把原始 kJ 值也记下来
    3) 该平台没有提供接口，只能按 foodinfo/<id>.html 逐条抓页面；
       分类列表页是 AJAX 空壳，拿不到完整 ID 清单，所以这里用"按名称在
       已抓取的候选 ID 范围内扫描"的办法定位。

已知限制（引用数据时必须带上）：
    - 页面不显示食物编码，所以本项目不提供食物编码
    - 能量只有 kJ，kcal 是本脚本换算值
    - 站点只写"中国食物成分表"，未标版本号，不要宣称"第6版"
    - 空白格 ≠ 0，"—"表示未检测，"Tr"表示未检出
"""

import argparse
import csv
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR

BASE = "https://nlc.chinanutri.cn/fq"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

KJ_PER_KCAL = 4.184

# 要取的配料。该平台没有"宫保鸡丁""红烧肉"这类菜肴条目
# （已对该库 259~1580 全部 1341 个条目做过全库穷举核验，确认没有），
# 所以改用"配料合成法"：取各配料每 100 g 的官方值，再按配方加权求和。
# 这样算出来的菜肴营养值是可追溯、可复查的，比直接抄一个来源不明的
# "宫保鸡丁 200 kcal" 靠谱得多。
INGREDIENTS = {
    # --- 主料：畜禽肉 ---
    # 注意：别名顺序有讲究，而且必须用"完全匹配优先"的策略（见 match_targets）。
    # 踩过的坑：
    #   "藕" 会先撞上"藕粉"(378.6 kcal)，而鲜藕只有 73.6 kcal，差 5 倍
    #   "鸡蛋" 会撞上"鸡蛋白"(60.7 kcal) 或"鸡蛋黄"(325 kcal)
    #   所以这里直接把完整条目名写在第一位，靠完全匹配命中
    "egg": ["蛋（鸡蛋，均值)", "鸡蛋(红皮)", "鸡蛋(白皮)"],
    "pork_belly": ["猪肉(肥瘦)(均值)", "猪肉(肥瘦)"],
    "pork_lean": ["猪肉(瘦)", "猪里脊"],
    "pork_rib": ["猪大排", "猪肋排"],
    # 猪皮：中国食物成分表里没有收录（已确认 259~1580 全库无"猪皮"条目），
    # 所以小笼包的皮冻这一项无法取到官方值，只能忽略并在配方 note 里说明
    "pork_fat": ["猪油(炼)", "猪油"],
    "chicken_breast": ["鸡胸脯肉", "鸡胸肉"],
    "chicken_whole": ["鸡(均值)", "鸡肉(均值)"],
    "duck_meat": ["鸭(均值)", "鸭肉"],
    "beef_shank": ["牛肉(后腱)", "牛腱子"],
    "beef_offal": ["牛肚", "牛舌", "牛心"],
    # --- 主料：水产 ---
    "fish_bass": ["鲈鱼[鲈花]", "鲈鱼"],
    "fish_grass_carp": ["草鱼[白鲩，草包鱼]", "草鱼"],
    # 三文鱼：该库用"鲑鱼[大麻哈鱼]"这个名称收录，不是"三文鱼"
    "fish_salmon": ["鲑鱼[大麻哈鱼]", "鲑鱼"],
    "sea_cucumber": ["海参", "海参(水浸)"],
    "abalone": ["鲍鱼(干)", "鲍鱼"],
    # 鱼肚/鱼鳔：全库未收录，无法取官方值
    "scallop_dried": ["扇贝(干)[干贝]", "干贝"],
    "shrimp_dried": ["虾米[海米，虾仁]", "虾米"],
    "squid": ["乌贼(鲜)", "鱿鱼"],
    # --- 豆制品 / 面筋 ---
    "tofu": ["豆腐(均值)", "豆腐(北)", "豆腐"],
    "tofu_skin": ["豆腐皮", "油皮"],
    "gluten_ball": ["面筋", "烤麸"],
    # --- 蔬果 ---
    "potato": ["马铃薯", "土豆"],
    "tomato": ["番茄", "西红柿"],
    "cucumber": ["黄瓜"],
    "carrot": ["胡萝卜(红)[金笋，丁香萝卜]", "胡萝卜(红)"],
    "green_pepper": ["辣椒(青，尖)", "青椒"],
    # 甜椒 / 红椒 / 灯笼椒：该库用"甜椒[灯笼椒，柿子椒]"收录（id 410）。
    # AI 通常写"红椒""彩椒"，靠它匹配。
    "bell_pepper": ["甜椒[灯笼椒，柿子椒]", "甜椒"],
    # 菠萝：咕咾肉（菠萝咕咾肉）的主料之一。
    # 该库条目名是"菠萝[凤梨，地菠萝]"（id 716）。
    "pineapple": ["菠萝[凤梨，地菠萝]", "菠萝"],
    "bamboo_shoot": ["竹笋(鲜)", "竹笋"],
    "wood_ear": ["木耳(水发)", "黑木耳"],
    "lotus_root": ["藕[莲藕]", "藕(莲藕)"],
    "water_chestnut": ["荸荠[马蹄，地栗](鲜)", "荸荠"],
    "garlic": ["大蒜", "蒜头"],
    "ginger": ["姜", "生姜"],
    "scallion": ["大葱", "葱", "青蒜"],
    "cilantro": ["香菜", "芫荽"],
    # --- 坚果 / 籽 ---
    "peanut": ["花生仁(炒)", "花生仁(生)"],
    "sesame": ["芝麻(黑)", "芝麻(白)", "芝麻"],
    "walnut": ["核桃(干)", "核桃"],
    # --- 油 / 调味 ---
    "cooking_oil": ["豆油", "花生油", "菜籽油"],
    "sesame_oil": ["香油", "芝麻油"],
    "chili_oil": ["辣椒油", "红油"],
    "soy_sauce": ["酱油(均值)", "酱油"],
    "sugar": ["糖（白砂糖）", "白砂糖"],
    "starch": ["玉米淀粉", "淀粉(玉米)"],
    "vinegar": ["醋(均值)", "醋"],
    "salt": ["精盐", "食盐"],
    "cooking_wine": ["黄酒", "料酒"],
    "doubanjiang": ["豆瓣酱", "郫县豆瓣"],
    "sweet_bean_sauce": ["甜面酱"],
    # 蚝油：全库未收录。近似的只有"牡蛎"(73.4 kcal) 和"生蚝"(57.6 kcal)，
    # 但那是鲜牡蛎不是蚝油（蚝油是加糖加淀粉的调味酱，热量高得多），
    # 所以不拿它顶替，在配方 note 里说明这项缺失
    #
    # 高汤/鸡汤：全库同样未收录。
    # ⚠这里踩过一个后果很严重的坑：一开始把 chicken_broth 的别名写成
    #   ["鸡精", "鸡汤"]，结果模糊匹配命中了"鸡精"。
    #   鸡精是浓缩调味料，钠 18864 mg/100g；而鸡汤的钠只有 2 mg/100g 量级，
    #   差了将近一万倍。佛跳墙里 300 g "高汤"因此算出 56593 mg 钠，
    #   麻婆豆腐 150 g 算出 28297 mg —— 全都离谱到超出每日摄入上限十倍以上。
    #   这类"名字里带同一个字但完全不是一种东西"的匹配陷阱，
    #   是本项目做别名匹配时最需要防的（同类还有"藕"匹配到"藕粉"）。
    #   结论：宁可让它匹配不上、在 note 里说明缺失，也不要拿相似名字顶替。
    "chicken_broth": ["鸡汤(不加盐)", "鸡汤"],
    "chicken_essence": ["鸡精"],
    # --- 主食 / 粉 ---
    "flour": ["小麦粉(标准粉)", "面粉", "小麦粉"],
    "rice": ["米饭(蒸)(均值)", "米饭(蒸)"],
    "yeast": ["酵母(干)", "酵母(鲜)"],
    # --- 成品菜（该库确实有的）---
    "fried_rice": ["什锦炒饭"],
    "dumpling": ["水饺(猪肉白菜馅)"],
}

# 已知的候选 ID 区间。该站 ID 不连续，分类列表页又是 AJAX 空壳拿不到清单，
# 只能逐条抓详情页。这个范围是实测出来的：259 之前是 404，
# 1580 之后也是 404，中间 259~1580 共 1341 个有效条目。
SCAN_RANGES = [
    (259, 424),       # 谷类 / 薯类 / 淀粉
    (425, 688),       # 干豆类 / 蔬菜
    (689, 819),       # 菌藻 / 水果
    (820, 966),       # 畜肉 / 禽肉 / 乳类
    (967, 1149),      # 蛋类 / 鱼虾蟹贝
    (1150, 1330),     # 婴幼儿 / 小吃 / 速食
    (1331, 1580),     # 速食 / 糕点 / 酒 / 糖
]


def fetch_html(fid, timeout=15):
    url = "%s/foodinfo/%d.html" % (BASE, fid)
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(300000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return None if e.code == 404 else ""
    except Exception:
        return ""


def parse_food_page(html):
    """
    解析详情页。

    页面结构（看原始 HTML 才知道的）：
        <title>什锦炒饭-食物营养成分查询平台</title>
        <h1>什锦炒饭</h1>            <- 注意页头还有一个空的 <h1></h1>，要取非空的
        <td class="td_left">能量(Energy)</td><td>779kJ</td><td>55</td><td>1295kJ</td>...
                                        ^^^^ 只要第一列，第二列是"同类均值"

    坑：能量那一行后面还有同类均值的 1295kJ，所以必须只取紧跟标签的第一列，
    不能用"标签后第一个数字"那种宽松写法。
    """
    if not html:
        return None

    vals = {}

    # 食物名：所有 h1 里取第一个非空的（页头那个是空的）
    for m in re.finditer(r"<h1[^>]*>(.*?)</h1>", html, re.S):
        t = re.sub(r"<[^>]+>", "", m.group(1))
        t = re.sub(r"&nbsp;?", " ", t).strip()
        if t:
            vals["_name"] = t
            break

    # 分类：h3 里第一个是分类名（第二个是备注说明）
    for m in re.finditer(r"<h3[^>]*>(.*?)</h3>", html, re.S):
        t = re.sub(r"<[^>]+>", "", m.group(1))
        t = re.sub(r"&nbsp;?", " ", t).strip()
        if t and not t.startswith("备注"):
            vals["_category"] = t
            break

    # 营养值：<td class="td_left">水分(Water)</td> <td>58.7g</td>
    # 标签里同时有中文和英文，用中文名匹配即可
    field_pat = {
        "energy_kj": "能量",
        "protein_g": "蛋白质",
        "fat_g": "脂肪",
        "carb_g": "碳水化合物",
        "fiber_g": "膳食纤维",
        "sodium_mg": "钠",
        "water_g": "水分",
    }
    for key, label in field_pat.items():
        # 标签单元格 -> 紧跟的第一个数据单元格，取里面的数字
        m = re.search(
            r"td_left[^>]*>\s*%s[^<]*</td>\s*<td[^>]*>\s*([0-9]+\.?[0-9]*|[—\-Tr]+)"
            % label, html)
        if m:
            raw = m.group(1)
            # "—"=未检测  "Tr"=未检出，都不是 0，按缺失处理
            if raw not in ("—", "-", "Tr", "tr"):
                vals[key] = float(raw)

    return vals if len(vals) > 1 else None


def scan(ranges, verbose=True):
    """扫描 ID 区间，返回 {id: 解析结果}"""
    found = {}
    for lo, hi in ranges:
        for fid in range(lo, hi + 1):
            html = fetch_html(fid)
            if not html:
                continue
            p = parse_food_page(html)
            if p:
                found[fid] = p
                if verbose:
                    nm = p.get("_name", "?")
                    kj = p.get("energy_kj")
                    print("   [%4d] %-28s %s kJ" % (fid, nm[:28],
                                                    ("%.0f" % kj) if kj else "-"))
            time.sleep(0.12)      # 别把站点打挂了
    return found


def match_targets(found):
    """
    把扫描结果和要取的配料对上。

    匹配顺序很关键：别名列表是按优先级排的，第一个别名优先。
    而且必须用"完全匹配优先、包含匹配兜底"的策略 ——
    直接用包含匹配会踩坑：搜"猪肉"会先撞上"猪肉(肥瘦)"没问题，
    但搜"鸡蛋"如果不加限制，可能匹配到"鸡蛋黄"或"松花蛋"；
    搜"糖"会匹配到"糖（白砂糖）"也可能匹配到"水果味软糖"。
    所以先做一遍完全匹配，再做包含匹配，且记录下匹配到了哪个名字，
    方便人工复核有没有配错。
    """
    out = {}
    for key, aliases in INGREDIENTS.items():
        hit = None
        # 第一轮：完全相等
        for a in aliases:
            for fid, p in found.items():
                if p.get("_name", "") == a:
                    hit = (fid, p, a, "exact")
                    break
            if hit:
                break
        # 第二轮：包含
        if not hit:
            for a in aliases:
                for fid, p in found.items():
                    if a in p.get("_name", ""):
                        hit = (fid, p, a, "contains")
                        break
                if hit:
                    break
        if hit:
            out[key] = hit
    return out


def to_kcal(kj):
    return round(kj / KJ_PER_KCAL, 1) if kj else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="扫描 ID 区间找条目")
    ap.add_argument("--cache", default=None, help="扫描结果缓存 json")
    ap.add_argument("--out", default=None, help="输出 CSV")
    args = ap.parse_args()

    cache = Path(args.cache) if args.cache else (DATA_DIR / "raw" / "cn_food_scan.json")

    if args.scan:
        print("扫描 %s 的条目（%d 个 ID 区间，共 %d 个 ID）..."
              % (BASE, len(SCAN_RANGES),
                 sum(hi - lo + 1 for lo, hi in SCAN_RANGES)))
        found = scan(SCAN_RANGES)
        import json
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in found.items()}, f,
                      ensure_ascii=False, indent=1)
        print("\n共找到 %d 个条目，已缓存到 %s" % (len(found), cache))
    else:
        import json
        if not cache.exists():
            print("没有缓存，先跑 --scan")
            sys.exit(1)
        with open(cache, encoding="utf-8") as f:
            found = {int(k): v for k, v in json.load(f).items()}
        print("从缓存读到 %d 个条目" % len(found))

    matched = match_targets(found)
    print("\n配料匹配结果（请人工复核匹配到的名字对不对）：")
    rows = []
    for key, aliases in INGREDIENTS.items():
        if key in matched:
            fid, p, used, how = matched[key]
            kj = p.get("energy_kj")
            kc = to_kcal(kj)
            print("   %-16s -> [%4d] %-24s %6s kJ = %6s kcal  (%s匹配: %s)"
                  % (key, fid, p.get("_name"), kj, kc, how, used))
            rows.append({
                "food": key, "cn_name": p.get("_name"), "cn_food_id": fid,
                "energy_kj_per100g": kj, "energy_kcal": kc,
                "protein_g": p.get("protein_g"), "fat_g": p.get("fat_g"),
                "carb_g": p.get("carb_g"), "fiber_g": p.get("fiber_g"),
                "sodium_mg": p.get("sodium_mg"),
                "source": "中国食物成分表(中国疾控中心营养与健康所)",
                "note": "kcal 由 kJ÷4.184 换算；站点未印 kcal",
            })
        else:
            print("   %-16s -> 未找到（别名：%s）" % (key, "、".join(aliases)))

    if rows and args.out:
        out = Path(args.out)
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n已写入 %s（%d 条）" % (out, len(rows)))


if __name__ == "__main__":
    main()
