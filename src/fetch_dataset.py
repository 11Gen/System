"""
fetch_dataset.py —— 从 Kaggle 公开接口下载数据集。

为什么走 Kaggle 而不是官方源：
    项目最初用的是 Food-101 官方地址 data.vision.ee.ethz.ch，
    实测国内访问只有 10~14 KB/s，4.65 GB 要下 141 小时，直接放弃。
    换了一圈：
        HuggingFace          DNS 被污染，完全不通
        hf-mirror.com        首页能开，但文件接口 404 / DNS 失败
        Zenodo                getaddrinfo failed，被墙
        Google Drive          不通
        GitHub               通，1.1 MB/s，但上面几乎没有带完整数据集的仓库
        Kaggle 公开下载接口   通，0.5~1.6 MB/s，而且部分数据集不需要登录
    所以最后落在 Kaggle。

    另外这里实现了断点续传：之前用 urlretrieve 下 ETH 那个源，
    断了就得从 0 开始，白等。用 Range 头续传能省很多时间。

用法：
    python src/fetch_dataset.py --slug jiezh2/common-chinese-food
    python src/fetch_dataset.py --slug xxx --list        # 只看文件清单
    python src/fetch_dataset.py --slug xxx --extract     # 只解压已下好的包
"""

import argparse
import io
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
API = "https://www.kaggle.com/api/v1"


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return "%.1f %s" % (n, u)
        n /= 1024
    return "%.1f TB" % n


def kaggle_json(path):
    req = urllib.request.Request("%s/%s" % (API, path), headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def show_info(slug):
    d = kaggle_json("datasets/view/%s" % slug)
    print("  标题    : %s" % d.get("title"))
    print("  大小    : %s" % human(d.get("totalBytes", 0)))
    print("  许可证  : %s" % d.get("licenseName"))
    print("  下载数  : %s" % d.get("downloadCount"))
    print("  更新时间: %s" % d.get("lastUpdated"))


def download(slug, dest, resume=True):
    """
    流式下载，边下边写盘。带 Range 断点续传。

    没用 urlretrieve 是因为：一是它不支持续传，二是进度回调里拿不到
    "已经写了多少"这个状态，断点续传没法算偏移量。
    """
    url = "%s/datasets/download/%s" % (API, slug)
    dest.parent.mkdir(parents=True, exist_ok=True)

    have = dest.stat().st_size if (resume and dest.exists()) else 0
    headers = dict(UA)
    if have:
        headers["Range"] = "bytes=%d-" % have
        print("  发现已下载 %s，尝试续传..." % human(have))

    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if have and r.status != 206:
                # 服务器不认 Range，只能从头来
                print("  服务器不支持续传（返回 %s），从头下载" % r.status)
                have = 0
            cl = r.headers.get("Content-Length")
            total = (int(cl) + have) if cl else None
            mode = "ab" if have else "wb"

            done = have
            last_print = 0.0
            with open(dest, mode) as f:
                while True:
                    chunk = r.read(512 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    # 每 2 秒刷新一次进度，刷太快日志没法看
                    if now - last_print > 2:
                        last_print = now
                        el = max(now - t0, 0.001)
                        rate = (done - have) / el
                        if total:
                            pct = done * 100.0 / total
                            eta = (total - done) / rate if rate > 0 else 0
                            sys.stdout.write(
                                "\r  %5.1f%%  %s / %s   %s/s   剩余 %s      "
                                % (pct, human(done), human(total),
                                   human(rate), human(eta * rate) if eta else "-"))
                        else:
                            sys.stdout.write("\r  %s   %s/s      "
                                             % (human(done), human(rate)))
                        sys.stdout.flush()
        print()
        return True
    except urllib.error.HTTPError as e:
        print("\n  下载失败 HTTP %s (%s)" % (e.code, e.reason))
        if e.code == 403:
            print("  403 说明这个数据集需要登录。需要配置 Kaggle API Token：")
            print("     1) 登录 kaggle.com -> 头像 -> Settings -> API -> Create New Token")
            print("     2) 把下载到的 kaggle.json 放到 %USERPROFILE%\\.kaggle\\ 下")
        return False
    except Exception as e:
        print("\n  下载中断：%s: %s" % (type(e).__name__, str(e)[:80]))
        print("  已下载的部分保留着，重跑本脚本可以续传。")
        return False


def peek_zip(zip_path, max_entries=25):
    """
    只看压缩包的结构，不解压。

    为什么不用 zipfile.ZipFile(zip_path) 直接读：
    如果文件是下载中断的半个包，直接读会在 namelist() 那里报 BadZipFile。
    这里先只读中央目录的尾部，能读就读，读不了就明确说"包不完整"。
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            print("\n  包内共 %d 个条目" % len(names))

            from collections import Counter
            # 顶两层目录做个统计，方便看类别分布
            def depth2(n):
                parts = n.split("/")
                return "/".join(parts[:2]) if len(parts) > 1 else parts[0]

            top = Counter(depth2(n) for n in names)
            print("  目录分布（前 20）：")
            for k, v in top.most_common(20):
                print("     %-56s %6d" % (k[:56], v))

            exts = Counter(Path(n).suffix.lower() for n in names if "." in n)
            print("  文件类型：%s" % dict(exts.most_common(8)))
            return names
    except zipfile.BadZipFile:
        print("\n  !! 这不是一个完整的 zip（下载可能没完成）")
        return None


def extract(zip_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n  解压到 %s ..." % out_dir)
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print("  解压完成，用时 %.1f 分钟" % ((time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="如 jiezh2/common-chinese-food")
    ap.add_argument("--out", default=None, help="zip 保存路径")
    ap.add_argument("--list", action="store_true", help="只看信息与包结构")
    ap.add_argument("--extract", action="store_true", help="下载后解压")
    ap.add_argument("--keep-zip", action="store_true", help="解压后保留 zip")
    args = ap.parse_args()

    name = args.slug.split("/")[-1]
    zip_path = Path(args.out) if args.out else (RAW_DIR / ("%s.zip" % name))
    out_dir = RAW_DIR / name

    print("=" * 74)
    print("数据集：%s" % args.slug)
    print("=" * 74)
    try:
        show_info(args.slug)
    except Exception as e:
        print("  取元信息失败：%s" % str(e)[:70])

    if args.list:
        if zip_path.exists():
            peek_zip(zip_path)
        else:
            print("\n  本地还没有 %s，先下载。" % zip_path)
        return

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print("\n开始下载 -> %s" % zip_path)
        if not download(args.slug, zip_path):
            sys.exit(1)
    else:
        print("\n本地已存在 %s (%s)" % (zip_path.name, human(zip_path.stat().st_size)))

    peek_zip(zip_path)

    if args.extract:
        extract(zip_path, out_dir)
        if not args.keep_zip:
            print("  删除 zip 以释放 %s ..." % human(zip_path.stat().st_size))
            zip_path.unlink()
        print("\n  数据目录：%s" % out_dir)
    else:
        print("\n  加 --extract 即可解压。")


if __name__ == "__main__":
    main()
