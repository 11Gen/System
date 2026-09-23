import io
import os
import sys

import requests
from pypdf import PdfReader

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf_text")
os.makedirs(OUT, exist_ok=True)

TARGETS = [
    ("BJCA001_kaoya_jishu_guifan",
     "http://bjprxh.cn/uploadfile/2018/1229/20181229021223299.pdf"),
    ("BJCA_tuantibiaozhun_other",
     "http://bjprxh.cn/uploadfile/2018/1229/20181229021224745.pdf"),
]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
}

for name, url in TARGETS:
    print("=" * 70)
    print("URL:", url)
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, verify=False)
        print("HTTP", r.status_code, "bytes:", len(r.content))
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            print("NOT A PDF / BAD STATUS. first bytes:", r.content[:80])
            continue
        pdf_path = os.path.join(OUT, name + ".pdf")
        with open(pdf_path, "wb") as fh:
            fh.write(r.content)
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
        print(full[:6000])
    except Exception as exc:  # noqa: BLE001
        print("ERROR:", type(exc).__name__, exc)
