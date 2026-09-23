"""
train_cnn.py —— MobileNetV2 迁移学习训练主程序。

整体思路（报告里"技术实现"那章就是按这个写的）：

    阶段一  冻结主干网络，只训练自己接的分类头。
            这一步是在"让新头先学会怎么用已有的特征"。
            如果一上来就全网络微调，随机初始化的头会产生很大的梯度，
            顺着反向传播把预训练好的权重冲烂，这是踩过的坑 ——
            第一次跑的时候直接全解冻训练，前两个 epoch 准确率掉到 20% 多，
            后来改成两阶段才正常。

    阶段二  解冻后面几个 block，用很小的学习率（1e-4）微调。
            学习率必须比阶段一小一个数量级，否则同样会把预训练特征毁掉。

    最终以验证集准确率为准保存最优权重，而不是用最后一个 epoch 的。

超参和类别都在 config.py 里，避免多处定义不一致。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (BATCH_SIZE, CLASSES, IMG_SIZE, MODEL_DIR, NUM_CLASSES,
                    RANDOM_SEED, SPLIT_DIR, STAGE1_EPOCHS, STAGE1_LR,
                    STAGE2_EPOCHS, STAGE2_LR, ensure_dirs)


def pick_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(batch_size=BATCH_SIZE, workers=0, aug_strength="normal"):
    """
    数据增强。

    这里的选择是有针对性的，不是随便抄了一套：
      - RandomResizedCrop(scale=(0.6,1.0))：食物照片里主体的占比差别很大，
        有的整盘都是菜，有的只占中间一小块，随机裁剪能让模型对尺度不敏感。
        下限取 0.6 而不是常见的 0.08 —— 0.08 会裁出很多看不出是什么的碎片，
        食物类别本来就靠整体外观区分，裁太狠反而变成噪声。
      - ColorJitter 幅度给得比较小(0.2)：食物识别很依赖颜色，
        抖动太强会让"三文鱼是橙色的"这种线索被破坏。
      - RandomHorizontalFlip：安全，食物左右翻转仍是同一个东西。
      - 不做垂直翻转：一碗面倒过来不符合真实拍摄场景。
    """
    if aug_strength == "none":
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # 验证/测试一律只做缩放 + 中心裁剪，保证评估结果可复现
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    loaders = {}
    for split, tf in (("train", train_tf), ("val", eval_tf), ("test", eval_tf)):
        d = SPLIT_DIR / split
        if not d.is_dir():
            print("找不到 %s，先跑 split_dataset.py" % d)
            return None
        ds = datasets.ImageFolder(str(d), transform=tf)
        # 这里特意检查一下类别顺序，ImageFolder 是按文件夹名字母序排的，
        # 和 config.CLASSES 的顺序不一定一致，不一致的话标签就全错位了
        if ds.classes != CLASSES:
            print("!! 警告：目录里的类别顺序和 config.CLASSES 不一致")
            print("   目录: %s" % ds.classes)
            print("   配置: %s" % CLASSES)
            print("   以目录为准继续，但评估时的类别名要按目录顺序来。")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
        print("   %-6s %5d 张   %d 类" % (split, len(ds), len(ds.classes)))

    return loaders


def build_model(arch="mobilenet_v2", num_classes=NUM_CLASSES, pretrained=True):
    """
    加载预训练主干 + 换掉分类头。

    分类头的结构：Dropout(0.2) -> Linear(1280, num_classes)
    MobileNetV2 原本最后一层是 Linear(1280, 1000)。直接换成 14 类输出，
    再在它前面加个 Dropout 抑制过拟合 —— 我们的训练集只有几千张，
    相比 ImageNet 的百万级小太多，不加 dropout 验证集很快就掉。
    """
    if arch != "mobilenet_v2":
        raise ValueError("目前只支持 mobilenet_v2")

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model, trainable, unfreeze_from=None):
    """
    控制主干网络哪些部分参与训练。

    unfreeze_from 给一个 block 序号（如 14 表示 features[14:] 解冻），
    不传就整体冻结/解冻。
    MobileNetV2 的 features 一共 19 个 block，越靠后语义越高级，
    微调时通常只动后面几个。
    """
    for p in model.features.parameters():
        p.requires_grad = trainable

    if trainable and unfreeze_from is not None:
        for i, blk in enumerate(model.features):
            for p in blk.parameters():
                p.requires_grad = (i >= unfreeze_from)


def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def _pfx(prefix):
    """权重文件名前缀：给了就加下划线，没给返回空串"""
    return ("%s_" % prefix) if prefix else ""


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """跑一个 epoch，返回 (平均loss, 准确率)"""
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            out = model(x)
            loss = criterion(out, y)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)

    return total_loss / max(n, 1), correct / max(n, 1)


def train_stage(model, loaders, device, epochs, lr, stage_name,
                unfreeze_from=None, ckpt_name="best.pt", history=None):
    """一个训练阶段。返回验证集准确率最好的那个模型状态。"""
    print("\n" + "-" * 72)
    print(" %s   共 %d 个 epoch   lr=%g" % (stage_name, epochs, lr))
    print("-" * 72)

    total, trainable = count_trainable(model)
    print("  参数量：总计 %.2fM，本阶段可训练 %.2fM (%.1f%%)"
          % (total / 1e6, trainable / 1e6, 100.0 * trainable / total))

    criterion = nn.CrossEntropyLoss()
    # 只把需要梯度的参数交给优化器，不然 Adam 会给冻结参数也建状态，白占显存
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    # 余弦退火，配合小 epoch 数比阶梯衰减稳一些
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_state, best_epoch = 0.0, None, -1
    history = history if history is not None else []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion,
                                    optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion,
                                    optimizer, device, train=False)
        scheduler.step()
        dt = time.time() - t0

        history.append({
            "stage": stage_name, "epoch": ep,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "seconds": round(dt, 1),
        })

        mark = ""
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = "  <- 目前最好，已保存"

        print("  epoch %2d/%d  train loss %.4f acc %.4f | "
              "val loss %.4f acc %.4f | %.0fs%s"
              % (ep, epochs, tr_loss, tr_acc, va_loss, va_acc, dt, mark))

    if best_state is not None:
        path = MODEL_DIR / ckpt_name
        torch.save(best_state, path)
        print("  最优验证准确率 %.4f（第 %d 个 epoch），已存 %s"
              % (best_acc, best_epoch, path.name))

    return best_acc, history


def plot_training_curve(history, tag=""):
    """
    画训练过程的准确率与损失曲线，报告里要用。

    横轴用"累计 epoch 序号"而不是阶段内的序号：
    两阶段是连续训练的，如果各自从 1 开始编号，
    图上会看到两条重叠的曲线，看不出阶段切换点在哪。
    这里额外画一条竖线标出阶段二的起点。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  没有 matplotlib，跳过训练曲线")
        return

    for fam in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
        if fam in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [fam]
            break
    plt.rcParams["axes.unicode_minus"] = False

    eps = list(range(1, len(history) + 1))
    tr_acc = [h["train_acc"] for h in history]
    va_acc = [h["val_acc"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]

    # 阶段二的起点
    switch = None
    for i, h in enumerate(history):
        if "阶段二" in h.get("stage", ""):
            switch = i + 1
            break

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (a, b, title, ylab) in zip(
            axes,
            [(tr_acc, va_acc, "准确率", "准确率"),
             (tr_loss, va_loss, "损失", "损失")]):
        ax.plot(eps, a, "o-", label="训练集", color="#4f81bd", markersize=3.5)
        ax.plot(eps, b, "s-", label="验证集", color="#c0504d", markersize=3.5)
        if switch:
            ax.axvline(switch - 0.5, color="gray", linestyle="--", linewidth=1.2)
            ax.text(switch - 0.4, ax.get_ylim()[1] * 0.97, " 阶段二开始",
                    fontsize=8, color="gray", va="top")
        ax.set_xlabel("epoch（两阶段连续编号）")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = config.FIGURE_DIR / ("training_curve%s.png"
                               % (("_" + tag) if tag else ""))
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  训练曲线已保存 %s" % out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs1", type=int, default=STAGE1_EPOCHS)
    ap.add_argument("--epochs2", type=int, default=STAGE2_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr1", type=float, default=STAGE1_LR)
    ap.add_argument("--lr2", type=float, default=STAGE2_LR)
    ap.add_argument("--workers", type=int, default=0,
                    help="Windows 上多进程 DataLoader 容易出问题，默认 0")
    ap.add_argument("--unfreeze-from", type=int, default=14,
                    help="从第几个 block 开始解冻，默认 14")
    ap.add_argument("--cpu", action="store_true", help="强制用 CPU")
    ap.add_argument("--stage1-only", action="store_true",
                    help="只跑阶段一（做消融实验用）")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="不用 ImageNet 预训练权重，从随机初始化开始训练"
                         "（消融实验用，用来量化迁移学习本身的价值）")
    ap.add_argument("--aug", default="normal", choices=["normal", "none"],
                    help="数据增强开关（做消融实验用）")
    ap.add_argument("--tag", default="", help="实验标签，会写进结果文件名")
    ap.add_argument("--ckpt-prefix", default="",
                    help="权重文件名前缀。跑多组实验时必须给，"
                         "否则每次训练都写同一个 best.pt 互相覆盖，"
                         "最后想复现某组实验就找不到权重了")
    args = ap.parse_args()
    # tag 和 ckpt 前缀默认保持一致，省得每次给两个参数
    if args.ckpt_prefix == "":
        args.ckpt_prefix = args.tag

    ensure_dirs()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = pick_device(args.cpu)
    print("=" * 72)
    print(" 训练 MobileNetV2   设备：%s" % device)
    if device.type == "cuda":
        print(" GPU：%s   显存 %.1f GB"
              % (torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
    print("=" * 72)

    print("\n加载数据 ...")
    loaders = build_dataloaders(args.batch_size, args.workers, args.aug)
    if loaders is None:
        sys.exit(1)
    # 类别顺序以 ImageFolder 实际读到的为准
    actual_classes = loaders["train"].dataset.classes

    model = build_model(num_classes=len(actual_classes),
                        pretrained=not args.no_pretrained).to(device)
    if args.no_pretrained:
        print("\n  [消融实验] 未加载 ImageNet 预训练权重，主干从随机初始化开始训练")

    history = []
    # ---------- 阶段一：只训练分类头 ----------
    # 主干全冻，只留分类头可训练。
    # 这里必须精确到 classifier，因为 build_model 里换过的头是新的，
    # 不冻主干的话反向传播会连带更新预训练权重（见文件开头的说明）。
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    acc1, history = train_stage(
        model, loaders, device, args.epochs1, args.lr1,
        "阶段一：冻结主干，训练分类头", history=history,
        ckpt_name="%sstage1_best.pt" % _pfx(args.ckpt_prefix))

    acc2 = None
    if not args.stage1_only:
        # ---------- 阶段二：解冻后段微调 ----------
        set_backbone_trainable(model, trainable=True,
                               unfreeze_from=args.unfreeze_from)
        acc2, history = train_stage(
            model, loaders, device, args.epochs2, args.lr2,
            "阶段二：解冻 features[%d:] 微调" % args.unfreeze_from,
            history=history, ckpt_name="%sbest.pt" % _pfx(args.ckpt_prefix))

    # ---------- 测试集评估 ----------
    ckpt = MODEL_DIR / ("%sbest.pt" % _pfx(args.ckpt_prefix) if acc2 is not None
                        else "%sstage1_best.pt" % _pfx(args.ckpt_prefix))
    print("\n用 %s 在测试集上评估 ..." % ckpt.name)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    te_loss, te_acc = run_epoch(model, loaders["test"],
                                nn.CrossEntropyLoss(), None, device, train=False)
    print("  测试集 loss %.4f  acc %.4f" % (te_loss, te_acc))

    # ---------- 保存结果 ----------
    tag = ("_" + args.tag) if args.tag else ""
    result = {
        # 这两个字段是 run_experiments.py 汇总时要用的
        "tag": args.tag or "main",
        "name": args.tag or "main",
        "device": str(device),
        "classes": actual_classes,
        "stage1_best_val_acc": round(acc1, 4),
        "stage2_best_val_acc": round(acc2, 4) if acc2 is not None else None,
        "test_acc": round(te_acc, 4),
        "test_loss": round(te_loss, 4),
        "epochs1": args.epochs1, "epochs2": args.epochs2,
        "lr1": args.lr1, "lr2": args.lr2,
        "batch_size": args.batch_size,
        "pretrained": not args.no_pretrained,
        "aug": args.aug,
        "stage1_only": bool(args.stage1_only),
        "unfreeze_from": args.unfreeze_from,
        "history": history,
    }
    out = MODEL_DIR / ("train_result%s.json" % tag)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n训练记录已写入 %s" % out)

    # 训练曲线只给主线实验画（消融实验跑得多，省点时间）
    if not args.tag or args.tag == "main":
        plot_training_curve(history, "")

    print("下一步：python src/evaluate.py")


if __name__ == "__main__":
    main()
