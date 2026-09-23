"""
ui_pages.py —— 页面的路径常量。

为什么单独一个模块：
    页面之间要做跳转（st.page_link），需要目标页面的路径。
    如果各处都手写 "app_pages/model_info.py" 这种字符串，
    改了文件名或目录之后**不会报错，只会静默失效**（链接点了没反应）。
    集中在这里，改名时只改一处。

为什么链接用路径字符串而不是 Page 对象：
    st.page_link 也接受 Page 对象，但那要求目标页面已经注册在
    st.navigation 里 —— 而页面被单独运行（例如测试）时并没有注册过，
    会抛 StreamlitPageNotFoundError。传路径字符串没这个限制。
"""

# 首页是入口文件本身（在 streamlit_app.py 里定义）
HOME = "streamlit_app.py"

# 各功能页
RECOGNIZE = "app_pages/recognize.py"
AI_RECOGNIZE = "app_pages/ai_recognize.py"
MEAL_PLAN = "app_pages/meal_plan.py"
MODEL_INFO = "app_pages/model_info.py"
FEEDBACK = "app_pages/feedback_page.py"

# 导航栏顺序。改顺序只改这里。
ORDER = [HOME, RECOGNIZE, AI_RECOGNIZE, MEAL_PLAN, MODEL_INFO, FEEDBACK]
