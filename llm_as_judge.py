from dotenv import load_dotenv
import os
import json
import logging
from typing import Optional
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## --- 法律调查质量的 LLM-as-a-Judge 评分标准 ---
LEGAL_SURVEY_RUBRIC = """
您是一位专家法律调查方法学家和严格的法律审查员。您的任务是评估给定法律调查问题的质量。为整体质量提供 1 到 5 的分数，以及详细的理由和具体反馈。重点关注以下标准：

1.  **清晰性和精确性（分数 1-5）：**
    * 1：极度模糊、高度歧义或令人困惑。
    * 3：中等清晰，但可以更精确。
    * 5：完全清晰、无歧义，在法律术语（如适用）和意图上精确。
2.  **中立性和偏见（分数 1-5）：**
    * 1：高度引导性或有偏见，明确影响受访者偏向特定答案。
    * 3：略微暗示性或可能被解释为引导性。
    * 5：完全中立、客观，没有任何引导性语言或带有倾向性的术语。
3.  **相关性和焦点（分数 1-5）：**
    * 1：与声明的调查主题无关或超出范围。
    * 3：松散相关，但可以更集中。
    * 5：与调查目标直接相关，并且集中于单一概念。
4.  **完整性（分数 1-5）：**
    * 1：遗漏了准确回答所需的关键信息或提供的上下文不足。
    * 3：基本完整，但缺少次要细节。
    * 5：提供受访者彻底回答所需的所有必要上下文和信息。
5.  **受众适当性（分数 1-5）：**
    * 1：使用目标受众无法理解的术语或对专家来说过于简单。
    * 3：通常适当，但某些术语可能具有挑战性或过于简化。
    * 5：完全适合目标调查受众的假定法律知识和背景。

**输出格式：** 您的响应必须是纯 JSON 对象，不要用 markdown 代码块包裹：
{
  "overall_score": <1-5 整数>,
  "rationale": "<简短理由>",
  "detailed_feedback": ["<要点1>", "<要点2>", ...],
  "concerns": ["<关切1>", ...],
  "recommended_action": "<建议>"
}
"""


class LLMJudgeForLegalSurvey:
    """使用 OpenAI 兼容 API 评估法律调查问题的类。

    默认使用阿里云百炼 DashScope (qwen-max)，也可切换为任何 OpenAI 兼容的模型。
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
    ):
        """
        初始化 LLM Judge。

        Args:
            model_name: 模型名称，默认从 .env 读取 LLM_MODEL 或使用 qwen-max
            api_key: API key，默认从 .env 读取 API_KEY
            base_url: API 地址，默认从 .env 读取 BASE_URL 或使用 DashScope
            temperature: 生成温度。较低的温度更适合确定性评估。
        """
        self.model_name = model_name or os.getenv("LLM_MODEL", "qwen-max")
        api_key = api_key or os.getenv("API_KEY")
        base_url = base_url or os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

        if not api_key:
            raise ValueError("未找到 API_KEY，请在 .env 文件中配置")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature

    def _generate_prompt(self, survey_question: str) -> str:
        """为 LLM judge 构建完整提示词。"""
        return f"{LEGAL_SURVEY_RUBRIC}\n\n---\n**要评估的法律调查问题：**\n{survey_question}\n---"

    def judge_survey_question(self, survey_question: str) -> Optional[dict]:
        """
        使用 LLM 判断单个法律调查问题的质量。

        Args:
            survey_question: 要评估的法律调查问题。
        Returns:
            Optional[dict]: 包含 LLM 判断的字典，如果发生错误则返回 None。
        """
        full_prompt = self._generate_prompt(survey_question)
        try:
            logging.info(f"向 '{self.model_name}' 发送判断请求...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的法律调查评估专家。请严格按照 JSON 格式输出。"},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                logging.error("LLM 返回空响应")
                return None

            return json.loads(content)

        except json.JSONDecodeError:
            logging.error(f"无法将 LLM 响应解码为 JSON。原始响应：{content}")
            return None
        except Exception as e:
            logging.error(f"LLM 判断期间发生意外错误：{e}")
            return None


if __name__ == "__main__":
    judge = LLMJudgeForLegalSurvey()

    # --- 高质量的示例 ---
    good_legal_survey_question = """
    在多大程度上您同意或不同意瑞士当前的知识产权法充分保护新兴的 AI 生成内容，假设该内容满足联邦最高法院确立的原创性标准？
    （选择一项：强烈不同意、不同意、中立、同意、强烈同意）
    """
    print("\n--- 评估好的法律调查问题 ---")
    judgment_good = judge.judge_survey_question(good_legal_survey_question)
    if judgment_good:
        print(json.dumps(judgment_good, indent=2, ensure_ascii=False))

    # --- 有偏见/差的示例 ---
    biased_legal_survey_question = """
    难道您不同意像 FADP 这样过度限制性的数据隐私法正在阻碍瑞士的基本技术创新和经济增长吗？
    （选择一项：是、否）
    """
    print("\n--- 评估有偏见的法律调查问题 ---")
    judgment_biased = judge.judge_survey_question(biased_legal_survey_question)
    if judgment_biased:
        print(json.dumps(judgment_biased, indent=2, ensure_ascii=False))

    # --- 模糊/含糊的示例 ---
    vague_legal_survey_question = """
    您对法律科技有什么想法？
    """
    print("\n--- 评估含糊的法律调查问题 ---")
    judgment_vague = judge.judge_survey_question(vague_legal_survey_question)
    if judgment_vague:
        print(json.dumps(judgment_vague, indent=2, ensure_ascii=False))