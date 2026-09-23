import io
import os
import re

import requests
from pypdf import PdfReader

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf_text")
os.makedirs(OUT, exist_ok=True)

TARGETS = [
    ("SAC_dbba_pk16132ff7",
     "https://dbba.sacinfo.org.cn/attachment/downloadStdFile?pk=16132ff757ea958e03976e65675614f1"),
]

KEYWORDS = ["麻婆豆腐", "鱼香肉丝", "咕咾", "咕噜", "烤鸭", "豆腐", "肉丝", "菠萝", "净含量"]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
}

for name, url in TARGETS:
    print("=" * 70)
    print("URL:", url)
    try:
        r = requests.get(url, headers=HEADERS, timeout=90, verify=False)
        print("HTTP", r.status_code, "bytes:", len(r.content))
        print("content-type:", r.headers.get("content-type"))
        if not r.content.startswith(b"%PDF"):
            print("NOT A PDF. first bytes:", r.content[:120])
            continue
        reader = PdfReader(io.BytesIO(r.content))
        print("pages:", len(reader.pages))
        chunks = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                txt = f"<<extract error: {exc}>>"
            chunks.append(f"----- page {i + 1} -----\n{txt}")
        full = "\n".join(chunks)
        txt_path = os.path.join(OUT, name + ".txt")
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(full)
        print("saved:", txt_path, "chars:", len(full))
        # report which keywords hit, with surrounding context
        for kw in KEYWORDS:
            for m in re.finditer(re.escape(kw), full):
                s = max(0, m.start() - 120)
                e = min(len(full), m.end() + 120)
                snippet = full[s:e].replace("\n", " | ")
                print(f"[HIT {kw}] ...{snippet}...")
    except Exception as exc:  # noqa: BLE001
        print("ERROR:", type(exc).__name__, exc)
