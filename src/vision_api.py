"""
vision_api.py —— 调用 AI 视觉模型识别菜品。

为什么要有这个模块：
    本地 MobileNetV2 只认识训练过的 16 道菜，这是闭集分类器的固有限制。
    这一模块用大模型的视觉能力补上这块：任何菜品都能识别出名字。

设计上的几个决定，都和各厂家接口的实际情况有关：

1) 用 OpenAI 兼容格式，不引入各家的专用 SDK。
   DeepSeek、阿里云百炼都提供 OpenAI 兼容端点，用同一个 openai 包就能调。
   如果分别用各家 SDK，装一堆包不说，界面代码里还得写分支判断。

2) base_url 和模型名都做成**可编辑**的，不只是下拉选择。
   原因很实际：本模块开发期间（2026-09）核实官方文档时发现，
   阿里云百炼在推带业务空间 ID 的新域名，智谱的视觉模型已经从
   GLM-4V-Plus 换到了 GLM-5.3 系列。**厂家改域名和模型名的速度很快**，
   如果把配置写死，过一段时间就会失效，而且用户毫无办法。
   做成可编辑之后，厂家一改，用户在界面上改一下就能继续用。

3) 只做识别，营养估算另走一条路（见 estimate_nutrition）。
   因为营养的算法和 16 道菜那边必须保持一致 —— 都是"配料克重 ×
   配料营养值 ÷ 成品总重"。让视觉模型直接报热量数字是不可核查的，
   会把项目里精心建立的数据可追溯性毁掉。

4) API Key 只存内存（存在 st.session_state 里），不落盘。
   这是用户明确选的方案。安全性最高，代价是刷新页面要重填。
"""

import base64
import json
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- 厂商预设
# 每一项都注明核实日期与官方文档，方便日后核对是否过期。
#
# 注意：这些**只是默认值**，界面上可以改。厂家改了域名或模型名之后，
# 用户自己改一下就能继续用，不需要等代码更新。
PROVIDERS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "doc": "https://api-docs.deepseek.com/zh-cn/guides/vision",
        "note": "官方文档确认支持图像理解。"
                "旧模型名 deepseek-v4-flash-vision-exp 已下线。",
        "key_hint": "在 platform.deepseek.com 创建，形如 sk-...",
        "verified": "2026-09-22",
    },
    "通义千问（阿里云百炼）": {
        # 官方文档说明旧域名仍可正常使用，且不需要业务空间 ID。
        # 文档同时推荐迁移到 https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com
        # 那种带 WorkspaceId 的新域名 —— 如果你有自己的业务空间 ID，
        # 用新域名更稳，直接在界面上把 base_url 改掉即可。
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-vl-32b-thinking",
        "doc": "https://www.alibabacloud.com/help/zh/model-studio/"
               "compatibility-of-openai-with-dashscope",
        "note": "官方文档确认支持 OpenAI 兼容接口。"
                "模型名以文档『支持的模型列表』为准，可能已更新。",
        "key_hint": "在阿里云百炼控制台创建，形如 sk-...",
        "verified": "2026-09-22",
    },
    "自定义": {
        "base_url": "",
        "model": "",
        "doc": "",
        "note": "任何 OpenAI 兼容的视觉模型服务都可以填在这里。",
        "key_hint": "按你所用服务的说明获取",
        "verified": "",
    },
}

DEFAULT_PROVIDER = "DeepSeek"

# 请求超时。视觉模型通常几秒返回，超过这个时间基本就是网络或服务出问题了，
# 早点报错比让用户干等好。
DEFAULT_TIMEOUT = 60.0


class VisionAPIError(Exception):
    """调用失败。消息是给用户看的，所以不用英文原始报错直接抛。"""


@dataclass
class DishResult:
    """一次识别的结果"""
    name: str                              # 菜名，如"番茄炒蛋"
    confidence: Optional[float] = None     # 模型自报的置信度，可能为 None
    cuisine: str = ""                      # 菜系
    description: str = ""                  # 一句话描述
    ingredients: List[Dict] = field(default_factory=list)  # 配料（估算营养时才用）
    total_g: Optional[float] = None        # 成品总重（估算营养时才用）
    raw: str = ""                          # 模型原始输出，出错排查用
    elapsed: float = 0.0                   # 耗时（秒）
    provider: str = ""
    model: str = ""


# ---------------------------------------------------------------- 工具
def image_to_data_url(image, max_side: int = 1024, quality: int = 88) -> str:
    """
    PIL 图 -> base64 data URL。

    上传前先缩到 max_side 以内：手机照片动辄 4000×3000，
    原图上传既慢又没必要 —— 识别一道菜不需要这个分辨率。
    1024 是权衡后的取值：再小会丢细节，再大对识别准确率帮助有限。

    注意各家的限制：单张 base64 图片最大 32 MiB，
    缩放后远低于这个上限，不会触发。
    """
    from PIL import Image

    im = image.convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                       Image.LANCZOS)

    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/jpeg;base64,%s" % b64


def _extract_json(text: str) -> Optional[dict]:
    """
    从模型输出里抠出 JSON。

    为什么要这么麻烦：即使提示里明确要求"只输出 JSON"，
    模型仍可能加上 ```json 代码块标记，或在前后写一两句解释。
    直接 json.loads 会失败。
    策略是先从 ``` 代码块里找，再退回按第一个 { 到最后一个 } 截取。
    """
    if not text:
        return None

    # 情况一：被 ```json ... ``` 包起来
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 情况二：整段就是 JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 情况三：夹在别的文字里，取最外层的花括号
    a, b = text.find("{"), text.rfind("}")
    if 0 <= a < b:
        try:
            return json.loads(text[a:b + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------- 提示词
RECOGNIZE_PROMPT = """请识别这张图片里的菜品。

要求：
1. 判断这是什么菜，给出准确的中文菜名。
2. 估计你的判断把握程度，用 0 到 1 之间的小数表示。
3. 如果图片里不是菜品（比如是风景、人物、物品），
   把 name 填成"非菜品"，confidence 填 0。

只输出下面这个 JSON，不要输出任何其他文字：
{
  "name": "菜名",
  "confidence": 0.85,
  "cuisine": "所属菜系，不确定就留空",
  "description": "一句话描述这道菜的样子，20 字以内"
}"""


NUTRITION_PROMPT = """你是一位中餐厨师兼营养师。请给出「{dish}」这道菜的标准配方。

要求：
1. 按一份的量给出配料与克重。一份指普通餐馆的常规份量。
2. 配料用常见的说法，比如"猪里脊肉""鸡蛋""番茄""食用油""生抽""盐"。
3. 只计实际会被吃进去的部分。炸东西的油只算吸附量（一般 10-30 克），
   不要把整锅油算进去。
4. 给出这道菜成品的总重量（克）。
5. 调味料也要算进去，它们对钠含量影响很大。

只输出下面这个 JSON，不要输出任何其他文字：
{{
  "ingredients": [
    {{"name": "猪里脊肉", "grams": 200}},
    {{"name": "鸡蛋", "grams": 50}}
  ],
  "total_g": 500
}}"""


# ---------------------------------------------------------------- 客户端
class VisionClient:
    """
    OpenAI 兼容接口的视觉模型客户端。

    **没有用 openai 这个包，而是直接用 urllib 发 HTTP 请求。**

    为什么这么做：
      1. 这个接口的本质就是"POST 一个 JSON、拿回一个 JSON"，
         自己发请求不到 80 行，没必要为它引入一个依赖。
      2. 装 openai 包时踩了坑：PyPI 索引能通，但文件服务器
         files.pythonhosted.org 在这台机器上不通，pip 会一直卡在下载。
         换成清华镜像虽然能下（实测 1.5 MB/s），但既然能不依赖就不依赖。
      3. 错误处理可以自己控制得更贴合本项目 —— 比如把 401/404/429
         翻成用户能看懂的中文提示，而不用去猜 SDK 抛的异常类型。

    用法：
        c = VisionClient(api_key="sk-...", base_url=..., model=...)
        r = c.recognize(pil_image)
        print(r.name, r.confidence)
    """

    def __init__(self, api_key: str, base_url: str, model: str,
                 provider: str = "自定义", timeout: float = DEFAULT_TIMEOUT):
        if not api_key or not api_key.strip():
            raise VisionAPIError("还没有填写 API Key。")
        if not base_url or not base_url.strip():
            raise VisionAPIError("还没有填写接口地址（base_url）。")
        if not model or not model.strip():
            raise VisionAPIError("还没有填写模型名称。")

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        self.provider = provider
        self.timeout = timeout

    # ------------------------------------------------------------ 内部
    def _endpoint(self) -> str:
        """
        拼出对话补全的完整地址。

        各家的 base_url 写法不太一样：有的只有域名（https://api.deepseek.com），
        有的已经带了路径（.../compatible-mode/v1）。所以这里判断一下，
        没带路径的就补上 /v1。这样两种写法都能用。
        """
        base = self.base_url
        if not base.endswith("/v1") and "/v1/" not in base + "/":
            # 已经含 compatible-mode/v1 这类路径的就不再补
            if "/compatible-mode" not in base:
                base = base + "/v1"
        return base + "/chat/completions"

    def _post(self, payload: dict) -> dict:
        """发一次请求，返回解析后的 JSON。所有异常都转成中文提示。"""
        import urllib.error
        import urllib.request

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint(),
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % self.api_key,
                "Accept": "application/json",
                # 有的服务会看 User-Agent，带上更保险
                "User-Agent": "dish-nutrition-project/1.0",
            })

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            # HTTP 错误要读出响应体 —— 各家都会在里面写明原因，
            # 只看状态码很难判断到底哪儿错了
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            raise VisionAPIError(self._friendly_http_error(e.code, detail))
        except urllib.error.URLError as e:
            reason = str(getattr(e, "reason", e))
            if "timed out" in reason.lower():
                raise VisionAPIError(
                    "请求超时（%.0f 秒）。可能是网络慢或服务端繁忙，"
                    "稍后重试。" % self.timeout)
            raise VisionAPIError("连不上服务：%s。请检查网络连接，"
                                 "以及接口地址是否填写正确。" % reason[:120])
        except Exception as e:
            raise VisionAPIError("调用出错：%s" % str(e)[:150])

        try:
            return json.loads(body)
        except json.JSONDecodeError:
            raise VisionAPIError("服务返回的不是合法 JSON：%s" % body[:200])

    @staticmethod
    def _friendly_http_error(code: int, detail: str) -> str:
        """
        把 HTTP 错误翻成用户能看懂的话。

        401 / 403 / 404 / 429 这四种最常见，而且解决办法完全不同，
        所以分开说 —— 笼统地报"调用失败"对用户没有帮助。
        """
        low = detail.lower()
        if code == 401 or "invalid_api_key" in low or "incorrect api key" in low:
            return ("API Key 无效（401）。请检查：① Key 有没有填错或过期；"
                    "② Key 和接口地址是不是同一个地域的 —— "
                    "有些厂家不同地域的 Key 不能混用。")
        if code == 404 or "model_not_found" in low or "does not exist" in low:
            return ("找不到模型或接口（404）。多半是模型名或接口地址不对 —— "
                    "厂家经常更新模型名，请到官方文档核对当前可用名称，"
                    "然后在上面「模型名称」里改过来。")
        if code == 403 or "permission" in low:
            return "没有权限（403）。可能是这个 Key 没开通该模型，或账户欠费。"
        if code == 429 or "rate limit" in low or "quota" in low:
            return "请求太频繁或额度用完（429）。等一会儿再试，或检查账户余额。"
        if code >= 500:
            return "服务端出错（%d）。这是对方服务的问题，稍后重试。" % code
        return "调用失败（HTTP %d）：%s" % (code, detail[:200] or "无详细信息")

    def _chat(self, content) -> str:
        """发一次对话请求，返回文本内容。content 是 OpenAI 格式的块数组。"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            # 识别任务要稳定，不要发挥。温度调低。
            "temperature": 0.1,
        }
        obj = self._post(payload)

        # 有的服务出错时返回 200 但 body 里带 error
        if isinstance(obj, dict) and obj.get("error"):
            err = obj["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise VisionAPIError("服务返回错误：%s" % str(msg)[:200])

        try:
            return obj["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise VisionAPIError("服务返回的内容格式不符合预期：%s"
                                 % json.dumps(obj, ensure_ascii=False)[:200])

    # ------------------------------------------------------------ 对外
    def recognize(self, image, verbose: bool = False) -> DishResult:
        """识别一张菜品照片。"""
        import time
        t0 = time.time()

        data_url = image_to_data_url(image)
        content = [
            {"type": "text", "text": RECOGNIZE_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        text = self._chat(content)
        elapsed = time.time() - t0

        data = _extract_json(text)
        if not data or not data.get("name"):
            # 模型没按格式返回时，不要把原始输出直接丢给用户看，
            # 但也不能假装成功 —— 返回一个"识别失败"的结果，
            # 原始输出留在 raw 字段里供排查。
            return DishResult(
                name="", confidence=None, raw=text, elapsed=elapsed,
                provider=self.provider, model=self.model)

        conf = data.get("confidence")
        try:
            conf = float(conf) if conf is not None else None
            if conf is not None:
                conf = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            conf = None

        return DishResult(
            name=str(data.get("name", "")).strip(),
            confidence=conf,
            cuisine=str(data.get("cuisine") or "").strip(),
            description=str(data.get("description") or "").strip(),
            raw=text, elapsed=elapsed,
            provider=self.provider, model=self.model)

    def estimate_recipe(self, dish_name: str) -> Tuple[List[Dict], Optional[float], str]:
        """
        让模型给出这道菜的配料与克重。

        返回 (配料列表, 成品总重, 原始输出)。
        配料列表形如 [{"name": "猪里脊肉", "grams": 200}, ...]

        注意：这些克重是**模型估算的**，不是权威数据。
        界面和报告里必须标注清楚，不能和营养库里那些有出处的数据混为一谈。
        """
        content = [{"type": "text",
                    "text": NUTRITION_PROMPT.format(dish=dish_name)}]
        text = self._chat(content)

        data = _extract_json(text)
        if not data:
            return [], None, text

        ings = []
        for it in data.get("ingredients") or []:
            if not isinstance(it, dict):
                continue
            name = str(it.get("name") or "").strip()
            try:
                grams = float(it.get("grams"))
            except (TypeError, ValueError):
                continue
            if name and grams > 0:
                ings.append({"name": name, "grams": grams})

        total_g = data.get("total_g")
        try:
            total_g = float(total_g) if total_g is not None else None
        except (TypeError, ValueError):
            total_g = None

        return ings, total_g, text


def match_known_dish(ai_name: str, classes) -> Optional[str]:
    """
    把 AI 认出的菜名和本地营养库的 16 道菜对一下。

    为什么要这步：AI 可能报出"宫保鸡丁""宫爆鸡丁""Kung Pao Chicken"，
    而本地库里的 key 是 Kung_Pao_Chicken。如果对不上，
    明明库里有的菜却要走"AI 估算营养"那条路，既慢又不如库里的准。

    匹配策略从严到宽：完全相等 -> 中文名相等 -> 互相包含。
    宁可漏配（退回 AI 估算）也不要错配 —— 把"番茄炒蛋"配成"宫保鸡丁"
    比配不上严重得多。
    """
    if not ai_name:
        return None

    import config

    q = ai_name.strip()
    q_low = q.lower().replace(" ", "").replace("_", "").replace("-", "")

    # 一轮：英文 key 直接相等
    for c in classes:
        if q_low == c.lower().replace("_", "").replace("-", ""):
            return c

    # 二轮：中文名完全相等
    for c in classes:
        cn = config.CN_NAME.get(c, "")
        if cn and q == cn:
            return c

    # 三轮：互相包含（只认长度 >= 3 的，避免"鸡"这种字误配）
    for c in classes:
        cn = config.CN_NAME.get(c, "")
        if len(cn) >= 3 and (cn in q or q in cn):
            return c

    return None


def find_ingredient_key(name: str) -> Optional[str]:
    """
    把 AI 报出的配料名映射到本地配料营养表的 key。

    这是"AI 估配料 + 本地算法算营养"能成立的关键一步。
    AI 说的是"猪里脊肉""鸡蛋"，而配料表里的 key 是 pork_lean、egg。

    匹配不上就返回 None —— 调用方会跳过该项并在结果里注明，
    不会拿相近的东西顶替。这条原则在本项目里已经吃过两次亏
    （"藕"配到"藕粉"、"鸡汤"配到"鸡精"），不再犯。
    """
    if not name:
        return None

    from build_nutrition_db import ING_CN_NAME, load_ingredients

    ing = load_ingredients()
    if not ing:
        return None

    q = name.strip()

    # 一轮：配料表里的中文名，完全相等
    for key, item in ing.items():
        if item[0] == q:
            return key

    # 二轮：AI 的名字包含表里的名字，或反过来（取最长匹配，避免误配）
    best, best_len = None, 0
    for key, item in ing.items():
        cn = item[0]
        if len(cn) < 2:
            continue
        if cn in q or q in cn:
            if len(cn) > best_len:
                best, best_len = key, len(cn)

    return best
