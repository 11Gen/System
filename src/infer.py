"""
infer.py —— 推理封装。

单独抽出来是因为界面（Streamlit）和命令行都要用同一套预处理逻辑。
如果两处各写一遍，很容易出现"界面里显示对了、命令行结果不对"的情况 ——
预处理里归一化参数写错一个数字就会这样，而且不会报错。

另外这里对 torch 做了延迟导入。原因是 Streamlit 在启动时会把所有页面
脚本都 import 一遍，如果页面文件在顶层 import torch，那么在还没训练出
模型、甚至没装 torch 的环境里，整个应用都起不来 —— 连"请先训练模型"的
提示都显示不出来，用户只会看到一堆报错。
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

import config

# ---------------------------------------------------------------- 预处理
# 必须和训练时 evaluation 用的完全一致。
# 训练用的是 ImageNet 的均值和标准差（因为主干是在 ImageNet 上预训练的），
# 这里手写而不是从 transforms 里取，是为了让界面不依赖 torchvision ——
# 见文件开头的说明。
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def energy_score(logits, temperature=1.0):
    """
    能量分：T · logsumexp(logits / T)

    logsumexp 相当于 softmax 去掉归一化，保留了 logits 的绝对大小。
    这个信息在判断"这个输入像不像训练分布"时很有用：
    对训练分布内的样本，模型通常给出整体更高的 logits。

    为什么不用最大置信度做拒识：softmax 是指数归一化，
    它会把不同样本之间的 logits 差距压平 —— 实测最大置信度
    在已知样本上的中位数是 1.000，几乎所有已知样本都顶格了，
    用 0.5 这种阈值根本区分不开。能量分没有这个问题。

    这里直接写在 infer.py 而不是 import open_set.py：
    避免界面模块去依赖一个只在离线评测时用的脚本。

    **不做任何符号变换，直接返回原始能量分。**
    判定规则：能量 >= 阈值 → 已知；能量 < 阈值 → 提示"不太确定"。

    踩过的坑：一开始习惯性取了负号当"异常分"，结果方向判反 ——
    标定出来的阈值是 -2.89，连正常的宫保鸡丁（能量 9.89）
    都被判成"不认识"，48 张已知样本只有 1 张被接受。
    """
    z = np.asarray(logits, dtype=np.float64) / temperature
    m = z.max()
    lse = m + np.log(np.exp(z - m).sum())
    return float(temperature * lse)


def load_reject_threshold():
    """
    读取由 src/open_set.py 标定出的能量分阈值。

    标定过程：在 527 张已知测试图 + 596 张未知类图上算 ROC，
    取「已知样本接受率 ≈95%」对应的能量分下限
    （实测 2.8921，对应未知拒识率约 47%）。

    找不到标定结果时返回 -inf，即不启用拒识 ——
    这样界面仍然可用，只是退化成普通闭集分类器。
    """
    p = config.MODEL_DIR / "open_set_threshold.json"
    if not p.exists():
        return float("-inf"), None
    try:
        import json
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return float(d.get("reject_threshold", float("-inf"))), d.get("method")
    except Exception:
        return float("-inf"), None


def preprocess(image, size=None):
    """
    PIL 图 -> 归一化后的 numpy 数组 (1, 3, H, W)

    只依赖 PIL 和 numpy，不依赖 torchvision。
    这样即使 torch 没装，界面上的"图片预处理预览"这类功能仍可用。
    """
    size = size or config.IMG_SIZE
    im = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)          # HWC -> CHW
    return arr[None, ...]                  # 加上 batch 维


class Predictor:
    """
    模型推理器。

    用法：
        p = Predictor()              # 自动找 models/best.pt
        p = Predictor("models/stage1_best.pt")
        if p.ready:
            res = p.predict(pil_image)
    """

    def __init__(self, ckpt_path=None, device=None, classes=None):
        self.ready = False
        self.error = None
        self.model = None
        self.device = None
        self.classes = list(classes) if classes else self._load_classes()
        self.ckpt_path = None
        # 拒识阈值，由 open_set.py 标定后写入 models/open_set_threshold.json
        self.reject_threshold, self.reject_method = load_reject_threshold()

        try:
            self._load(ckpt_path, device)
            self.ready = True
        except Exception as e:
            # 不抛异常，把错误记下来让界面显示。
            # 界面上明确告诉用户"模型还没训练好，请先跑 train_cnn.py"
            # 比抛一个堆栈有用得多。
            self.error = "%s: %s" % (type(e).__name__, e)

    # ------------------------------------------------------------ 内部
    @staticmethod
    def _load_classes():
        """类别顺序以 split_dataset.py 落盘的为准"""
        p = config.SPLIT_DIR / "classes.json"
        if p.exists():
            import json
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return list(config.CLASSES)

    def _load(self, ckpt_path, device):
        import torch

        from train_cnn import build_model, pick_device

        self.device = torch.device(device) if device else pick_device()

        # 找不到指定权重时退回默认候选，方便直接跑起来
        candidates = []
        if ckpt_path:
            candidates.append(Path(ckpt_path))
        candidates += [config.MODEL_DIR / "best.pt",
                       config.MODEL_DIR / "stage1_best.pt"]

        ckpt = next((c for c in candidates if c.exists()), None)
        if ckpt is None:
            raise FileNotFoundError(
                "找不到模型权重。请先运行：python src/train_cnn.py")

        model = build_model(num_classes=len(self.classes), pretrained=False)
        state = torch.load(ckpt, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device).eval()

        self.model = model
        self.ckpt_path = ckpt

    # ------------------------------------------------------------ 对外
    def predict(self, image, topk=5):
        """
        单张图预测。

        同时给出两种"异常分"，各有各的用处：
          - 最大置信度：人看得懂，用于界面展示
          - 能量分（energy）：开集识别里区分已知/未知更有效，
            由 src/open_set.py 在真实数据上标定了阈值，用于拒识判断

        返回 dict：
            probs      : 每个类别的概率（按 self.classes 顺序）
            topk       : [(类别key, 中文名, 概率), ...] 按概率降序
            top1       : topk[0]
            entropy    : 预测分布的熵（归一化到 0~1）
            energy     : 能量分（logsumexp 取负），越大越像已知
            uncertain  : 是否应提示"不太确定"（能量分低于标定阈值）
        """
        if not self.ready:
            raise RuntimeError("模型未就绪：%s" % self.error)

        import torch

        x = torch.from_numpy(preprocess(image)).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            logit_arr = logits[0].cpu().numpy()

        order = np.argsort(-probs)
        topk_list = []
        for i in order[:topk]:
            key = self.classes[i]
            cn, cuisine, desc = config.CLASS_INFO.get(key, (key, "", ""))
            topk_list.append({
                "key": key, "cn_name": cn, "cuisine": cuisine,
                "desc": desc, "prob": float(probs[i]),
            })

        # 归一化熵：H / ln(N)。全部概率相等时等于 1，完全确定时等于 0。
        eps = 1e-12
        h = -float(np.sum(probs * np.log(probs + eps)))
        h_norm = h / np.log(len(probs)) if len(probs) > 1 else 0.0

        energy = float(energy_score(logit_arr))

        return {
            "probs": probs,
            "topk": topk_list,
            "top1": topk_list[0],
            "entropy": h_norm,
            "energy": energy,
            "threshold": self.reject_threshold,
            "uncertain": energy < self.reject_threshold,
        }

    def predict_batch(self, images, topk=5):
        return [self.predict(im, topk) for im in images]

    @property
    def model_name(self):
        return self.ckpt_path.name if self.ckpt_path else "(未加载)"


# 这里原本写的是「最大置信度 < 0.5 就提示不确定」。
# 加了开集评测之后发现这个阈值是错的：已知样本的最大置信度中位数是 1.000，
# 用 0.5 当判据几乎永远不会触发，等于没做提示。
# 现在改用能量分 + open_set.py 标定出的阈值（见 load_reject_threshold）。


def load_sample_images(limit=None) -> List[dict]:
    """
    从测试集里挑几张图，给界面做"示例图片"用。

    从测试集取而不是训练集：这样界面上演示的图模型没见过，
    结果更能反映真实表现。
    """
    out = []
    test_dir = config.SPLIT_DIR / "test"
    if not test_dir.is_dir():
        return out

    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")))
        if not files:
            continue
        key = cls_dir.name
        cn = config.CN_NAME.get(key, key)
        out.append({"path": files[0], "key": key, "cn_name": cn,
                    "label": "%s（%s）" % (cn, key)})
        if limit and len(out) >= limit:
            break
    return out
