"""
test_app.py —— 界面冒烟测试。

用 Streamlit 自带的 AppTest 框架在进程内跑一遍各页面，
不需要启动浏览器或服务器，就能发现语法错误、导入错误、
以及运行时异常（比如某个变量名写错、字典 key 不存在）。

这个测试在开发过程中确实抓到过问题：
model_info.py 里一段跨数据集展示代码被误缩进到 for 循环内部，
导致只有存在某张图时才会渲染；这种错误光看代码不容易发现。

用法：
    python test_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    ("streamlit_app.py", "主页介绍（入口）"),
    ("app_pages/recognize.py", "识菜"),
    ("app_pages/ai_recognize.py", "AI 识别"),
    ("app_pages/meal_plan.py", "膳食管家"),
    ("app_pages/model_info.py", "权威所在"),
    ("app_pages/feedback_page.py", "帮助与反馈"),
]

TIMEOUT = 180


def run_page(path, name):
    print("=" * 74)
    print("测试页面：%s  (%s)" % (name, path))
    print("=" * 74)
    try:
        at = AppTest.from_file(str(ROOT / path), default_timeout=TIMEOUT)
        at.run()
    except Exception as e:
        print("  [失败] 无法运行：%s: %s" % (type(e).__name__, e))
        return False

    # AppTest 把异常收集在 at.exception 里，不会直接抛出来
    if at.exception:
        print("  [失败] 页面抛出 %d 个异常：" % len(at.exception))
        for ex in at.exception:
            print("     %s" % ex.value)
        return False

    # 顺便看看渲染出了什么，确认不是"空页面"。
    # 注意 Streamlit 把 st.title / st.subheader / st.caption 各自归到
    # 不同的元素集合里，不算在 markdown 里 —— 只统计 markdown 会误判成空页面。
    counts = {
        "title": len(at.title),
        "subheader": len(at.subheader),
        "markdown": len(at.markdown),
        "caption": len(at.caption),
        "metric": len(at.metric),
        "tabs": len(at.tabs),
    }
    print("  渲染元素：" + "，".join("%s %d" % (k, v)
                                    for k, v in counts.items() if v))
    if sum(counts.values()) == 0:
        print("  [警告] 页面没有任何可见元素，可能是空页面")

    if at.error:
        print("  页面里有 %d 个 error 提示：" % len(at.error))
        for e in at.error:
            print("     %s" % str(e.value)[:100])
    if at.warning:
        print("  页面里有 %d 个 warning 提示：" % len(at.warning))
        for w in at.warning:
            print("     %s" % str(w.value)[:100])
    print("  [通过]")
    return True


def main():
    print("\nStreamlit 界面冒烟测试\n")
    results = {}
    for path, name in PAGES:
        if not (ROOT / path).exists():
            print("跳过（文件不存在）：%s" % path)
            results[name] = None
            continue
        results[name] = run_page(path, name)

    print("\n" + "=" * 74)
    print("测试结果汇总")
    print("=" * 74)
    ok = 0
    for name, r in results.items():
        mark = "通过" if r else ("跳过" if r is None else "失败")
        print("  %-24s %s" % (name, mark))
        if r:
            ok += 1
    total = sum(1 for r in results.values() if r is not None)
    print("\n  %d / %d 通过" % (ok, total))
    return 0 if ok == total else 1


if __name__ == "__main__":
    code = main()
    # 用 os._exit 跳过 Python 的正常退出流程。
    # 原因：Streamlit 测试会在系统临时目录里建文件，
    # 退出时 tempfile 模块会去清理它，而某些受限环境下会因权限不足抛
    # PermissionError，把测试结果淹在一大堆堆栈里。
    # 这里所有结果都已经打印完了，直接退出不影响结论。
    import os
    sys.stdout.flush()
    os._exit(code)
