"""
config.py —— 全项目共享的配置。

单独抽出来是因为类别清单被好几个模块用到（数据划分、训练、评估、界面），
之前写在一处改一处忘一处，train 里 14 类、app 里 12 类，跑起来直接报错，
所以统一放这里。
"""

from pathlib import Path
import os

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SRC_IMG_DIR = DATA_DIR / "raw" / "common-chinese-food"   # 原始图（按类别分目录）
SPLIT_DIR = DATA_DIR / "dataset"           # 划分后的 train/val/test
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "figures"
STATIC_DIR = ROOT / "static"
CACHE_DIR = ROOT / ".cache"                # torch 权重缓存放项目内

# 把 torch / matplotlib 的缓存目录指到项目里。
# 起因是这台机器上默认的 C:\Users\<用户>\.cache\torch 写不进去，
# 直接报 PermissionError，而报错信息在 load_state_dict_from_url 里面，
# 不看堆栈根本想不到是缓存目录的问题。
# 放在项目内还有个好处：换机器/重装环境时预训练权重不用重新下。
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

# ---------------------------------------------------------------- 类别
# 16 道中国名菜。数据集来源：Kaggle "16 Famous Chinese Dishes"（Apache 2.0），
# 原始是 YOLO 目标检测格式，经 prepare_dataset.py 清洗+裁剪后用于分类。
CLASSES = [
    "Boiled_Fish_with_Sichuan_Peppercorns",
    "Braised_Pork_Meatballs_in_Brown_Sauce",
    "Buddha_Jumps_Over_the_Wall",
    "Dongpo_Pork",
    "Fish_with_Pickled_Cabbage_and_Chili",
    "Husband_and_Wife_Lung_Slices",
    "Kung_Pao_Chicken",
    "Mapo_Tofu",
    "Peking_Duck",
    "Saliva_Chicken",
    "Soup_Dumplings",
    "Steamed_Sea_Bass",
    "Sweet_and_Sour_Pork",
    "Twice-Cooked_Pork",
    "West_Lake_Vinegar_Fish",
    "Yuxiang_Shredded_Pork",
]

NUM_CLASSES = len(CLASSES)

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# 中文名 / 菜系 / 一句话描述。界面显示和报告制表都要用。
CLASS_INFO = {
    "Peking_Duck": ("北京烤鸭", "京菜", "果木炭火烤制，皮脆肉嫩，配薄饼甜面酱"),
    "Sweet_and_Sour_Pork": ("咕咾肉", "粤菜", "猪肉块挂糊油炸后裹酸甜汁"),
    "Mapo_Tofu": ("麻婆豆腐", "川菜", "豆腐配牛肉末，麻辣鲜香"),
    "Yuxiang_Shredded_Pork": ("鱼香肉丝", "川菜", "肉丝木耳笋丝，酸甜微辣的鱼香味"),
    "Husband_and_Wife_Lung_Slices": ("夫妻肺片", "川菜", "牛肉牛杂凉拌，红油麻辣"),
    "Twice-Cooked_Pork": ("回锅肉", "川菜", "五花肉先煮后炒，配蒜苗豆瓣酱"),
    "Kung_Pao_Chicken": ("宫保鸡丁", "川菜", "鸡丁配花生米，糊辣荔枝味"),
    "Saliva_Chicken": ("口水鸡", "川菜", "鸡肉凉拌，红油花椒调味"),
    "Soup_Dumplings": ("小笼汤包", "苏菜", "薄皮肉馅，内含汤汁"),
    "Braised_Pork_Meatballs_in_Brown_Sauce": ("红烧狮子头", "苏菜", "大肉丸配青菜红烧"),
    "Dongpo_Pork": ("东坡肉", "浙菜", "五花肉黄酒慢炖，肥而不腻"),
    "West_Lake_Vinegar_Fish": ("西湖醋鱼", "浙菜", "草鱼浇糖醋汁，酸甜带姜末"),
    "Buddha_Jumps_Over_the_Wall": ("佛跳墙", "闽菜", "海参鲍鱼等多种海味与肉类同炖"),
    "Steamed_Sea_Bass": ("清蒸鲈鱼", "粤菜", "清蒸保留原味，淋豉油"),
    "Fish_with_Pickled_Cabbage_and_Chili": ("酸菜鱼", "川菜", "鱼片配酸菜，酸辣开胃"),
    "Boiled_Fish_with_Sichuan_Peppercorns": ("水煮鱼", "川菜", "鱼片在麻辣红油中，配豆芽"),
}

# 界面上给用户看的中文名（从 CLASS_INFO 里取，保持单一数据源）
CN_NAME = {k: v[0] for k, v in CLASS_INFO.items()}

# ---------------------------------------------------------------- 训练超参
# 放这里是为了让报告里的"参数设置"章节有个唯一出处
IMG_SIZE = 224                 # MobileNetV2 的标准输入
BATCH_SIZE = 32
VAL_RATIO = 0.10               # 从官方 train 里切出来的验证集比例
TEST_RATIO = 0.10              # 同样从官方 train 切
RANDOM_SEED = 20250301         # 固定种子，保证结果可复现

STAGE1_EPOCHS = 10             # 阶段一：冻住主干，只训练分类头
STAGE1_LR = 1e-3
STAGE2_EPOCHS = 8              # 阶段二：解冻后 4 个 block 做微调
STAGE2_LR = 1e-4

NUTRITION_CSV = DATA_DIR / "nutrition_db.csv"


def ensure_dirs():
    """用到哪个建哪个，省得每个脚本都写一遍"""
    for d in (MODEL_DIR, FIGURE_DIR, SPLIT_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # 直接运行本文件时打印一下配置，方便自查
    print("项目根目录 : %s" % ROOT)
    print("类别数     : %d" % NUM_CLASSES)
    for i, c in enumerate(CLASSES):
        print("  [%2d] %-16s %s" % (i, c, CN_NAME.get(c, "?")))
    print("图像尺寸   : %d" % IMG_SIZE)
    print("Batch size : %d" % BATCH_SIZE)
