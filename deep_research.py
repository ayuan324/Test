from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
# from duckduckgo_search import DDGS # 移除 DuckDuckGo
from tavily import TavilyClient # 导入 Tavily Client
import asyncio
from tqdm import tqdm
import json
import ast
import os
import time # 导入 time 模块

os.environ["SSL_CERT_FILE"] = ""
# # 移除硬编码的环境变量设置，这些应该在外部设置
# os.environ['TAVILY_API_KEY'] = "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"

# 加载 Tavily API 密钥
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    # 在无法获取密钥时可以选择抛出错误，更严格
    # raise ValueError("错误：请设置 TAVILY_API_KEY 环境变量。")
    print("[Error] 未找到 TAVILY_API_KEY 环境变量。Tavily 搜索将无法工作。")
    tavily_client = None # 保持 None，后续逻辑会处理
else:
    tavily_client = TavilyClient(api_key=tavily_api_key)

# --- 移除全局 LLM 初始化 ---
# model = ChatOpenAI(...)

# +++ 新增 LLM 创建函数 +++
def create_llm(model_name: str = "qwen-max", temperature: float = 0.7, max_tokens: int = 20480, **kwargs):
    """根据名称和参数创建并返回一个 LangChain Chat Model 实例"""
    print(f"[INFO] Creating LLM: {model_name}, Temp: {temperature}, Max Tokens: {max_tokens}")

    if model_name == "qwen-max":
        # api_key = os.getenv("DASHSCOPE_API_KEY", "sk-629fea8a62e94a9cbf4ef47827ce53e4") # 移除硬编码后备值
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
             raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量 (用于 Qwen)")
        return ChatOpenAI(
            model="qwen-max-latest",
            openai_api_key=api_key,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=False,
            **kwargs
        )
    elif model_name == "deepseek-chat":
        # api_key = os.getenv("DEEPSEEK_API_KEY", "sk-8d71a948abc44172ae9471247099b92b") # 移除硬编码后备值
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
             raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量 (用于 DeepSeek)")
        return ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=False,
            **kwargs
        )
    elif model_name == "azure-gpt-4o":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
        api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

        if not api_key:
            raise ValueError("请设置 AZURE_OPENAI_API_KEY 环境变量 (用于 Azure OpenAI)")
        if not azure_endpoint:
             raise ValueError("请设置 AZURE_OPENAI_ENDPOINT 环境变量 (用于 Azure OpenAI)")

        return AzureChatOpenAI(
            openai_api_version=api_version,
            deployment_name=deployment_name,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            temperature=temperature,
            max_tokens=min(max_tokens, 4096),
            streaming=False,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")

async def main():
    # 这个 main 函数在 UI 模式下不会被直接调用
    query = input("请输入：\n")
    # 简单的回调函数，用于命令行打印
    def print_callback(step_name, content, token_used=0, time_taken=0):
        print("="*50)
        print(f"状态: {step_name} (耗时: {time_taken:.2f}s | Tokens: {token_used})")
        if isinstance(content, (dict, list)):
            print(f"内容: \n{json.dumps(content, ensure_ascii=False, indent=2)}")
        else:
            print(f"内容: {content}")
        print("="*50)

    raw_reports = await cell(model, query, ui_callback=print_callback)
    print("="*50)
    print("最终报告:")
    print(raw_reports['report'])

def _get_token_usage(response):
    """尝试从 LangChain 响应中提取 token 使用量"""
    # LangChain 的 ChatResult 通常在 response.llm_output['token_usage'] 中包含使用信息
    # 对于 AzureChatOpenAI, 可能在 response.response_metadata['token_usage']
    usage = getattr(response, 'usage', None)
    if usage:
        # 尝试 Anthropic, OpenAI, Cohere 等常见格式
        if hasattr(usage, 'total_tokens'): return usage.total_tokens
        if hasattr(usage, 'input_tokens') and hasattr(usage, 'output_tokens'): return usage.input_tokens + usage.output_tokens
        if isinstance(usage, dict):
            if 'total_tokens' in usage: return usage['total_tokens']
            if 'input_tokens' in usage and 'output_tokens' in usage: return usage['input_tokens'] + usage['output_tokens']

    llm_output = getattr(response, 'llm_output', None)
    if llm_output and 'token_usage' in llm_output:
        return llm_output['token_usage'].get('total_tokens', 0)

    response_metadata = getattr(response, 'response_metadata', None)
    if response_metadata and 'token_usage' in response_metadata:
        return response_metadata['token_usage'].get('total_tokens', 0)

    # 尝试直接访问属性
    try:
        if hasattr(response, 'token_usage') and isinstance(response.token_usage, dict):
            return response.token_usage.get('total_tokens', 0)
    except AttributeError:
        pass

    # 如果都失败，返回0
    print("[Warning] 无法获取 token 使用量信息")
    return 0

async def assess_search_quality(model, original_query: str, subtask: str, keywords: list, tavily_response: dict, ui_callback=None) -> str:
    """评估Tavily搜索结果的质量。"""
    t0 = time.time()
    if ui_callback:
        ui_callback("评估搜索质量中...", f"正在评估针对子任务 '{str(subtask)[:30]}...' 的搜索结果", 0, 0)

    # 构建评估prompt
    prompt_messages = [
        SystemMessage(
            """你是一个专业的搜索结果质量评估员。你需要根据原始用户查询、当前子任务、使用的搜索关键词以及Tavily返回的搜索结果，来评估搜索结果的质量。\n"
            "Tavily的响应包含 `answer` (一个直接的综合答案) 和 `results` (一个包含多个网页摘要的列表)。\n"
            "请评估这些结果是否与子任务高度相关、信息是否丰富、是否能为回答子任务提供足够的材料。\n"
            "你的评估等级只能是以下四种之一：excellent, good, poor, failed.\n"
            "- excellent: 结果高度相关，信息丰富，看起来可以直接用于回答子任务。Tavily的 `answer` 字段质量很高。\n"
            "- good: 结果相关，包含有用的信息，但可能需要进一步筛选或补充。Tavily的 `answer` 字段有用但可能不完整。\n"
            "- poor: 结果相关性较低，信息量不足，或者Tavily的 `answer` 字段质量差或不相关。可能需要用不同的关键词重试。\n"
            "- failed: 结果完全不相关，或者搜索API返回错误/无结果。必须重试。\n"
            "请只返回评估等级的单词，例如：good"""
        ),
        HumanMessage(
            f"""原始用户查询: {original_query}\n"
            f"当前子任务/方面: {subtask}\n"
            f"使用的搜索关键词: {keywords}\n"
            f"Tavily搜索结果 (JSON):\n{json.dumps(tavily_response, ensure_ascii=False, indent=2)}\n\n"
            f"请评估以上搜索结果的质量 (excellent, good, poor, failed):"""
        )
    ]

    response = await model.ainvoke(prompt_messages)
    quality_grade = response.content.strip().lower()
    t1 = time.time()
    tokens_used = _get_token_usage(response)

    valid_grades = ["excellent", "good", "poor", "failed"]
    if quality_grade not in valid_grades:
        # 如果返回的不是标准答案，默认给 poor，并记录原始回答
        if ui_callback:
            ui_callback("搜索质量评估格式错误", f"LLM返回非标准评估: '{quality_grade}'. 默认为 poor.", tokens_used, t1-t0)
        quality_grade = "poor"
    elif ui_callback:
        ui_callback("搜索质量评估完成", f"子任务 '{str(subtask)[:30]}...' 的搜索结果质量: {quality_grade.upper()}", tokens_used, t1-t0)
    
    return quality_grade

async def cell(model, query, ui_callback=None):
    """修改后的思考单元，包含搜索结果质量评估和重试机制。"""
    aggregated_results = []
    max_search_retries = 3 # 每个子任务最多重试3次搜索

    # --- 1. 一次性任务细化 ---
    t0_refine = time.time()
    refine_message = [
        SystemMessage("""你是一个经验丰富的项目经理。你需要根据用户提出的研究任务，将其分解为一系列具体、可执行的子任务或需要研究的关键方面。分解的粒度应该适中，能够覆盖用户需求的核心，避免过于琐碎或过于宏大。请根据任务的复杂性决定分解出的子任务数量。输出的内容必须是一个 Python 列表，例如：["子任务1", "方面2", "需要调研的点3", ...]"""),
        HumanMessage(f"请将以下用户研究任务进行分解：\n\n{query}")
    ]
    if ui_callback: ui_callback("细化任务中...", f"正在为 '{query[:50]}...' 分解子任务", 0, 0)
    response_refine = await model.ainvoke(refine_message)
    t1_refine = time.time()
    tokens_refine = _get_token_usage(response_refine)
    refined_subtasks_str = response_refine.content
    try:
        refined_subtasks = ast.literal_eval(refined_subtasks_str)
        if not isinstance(refined_subtasks, list): refined_subtasks = [refined_subtasks_str]
    except (ValueError, SyntaxError):
        refined_subtasks = [refined_subtasks_str]
        if ui_callback: ui_callback("解析细化列表失败", f"无法解析LLM返回的列表: {refined_subtasks_str}", tokens_refine, t1_refine - t0_refine)
    if ui_callback: ui_callback("任务细化完成", {"原始任务": query, "分解出的子任务/方面": refined_subtasks}, tokens_refine, t1_refine - t0_refine)

    # --- 2. 顺序执行子任务搜索 ---
    for i, subtask in enumerate(refined_subtasks):
        current_task_label = f"子任务 {i+1}/{len(refined_subtasks)}: {str(subtask)[:30]}..."
        if ui_callback: ui_callback("处理子任务", f"开始处理: {current_task_label}", 0, 0)
        search_attempts = 0
        current_subtask_search_results = []
        keywords_history = []
        quality = "failed" # Default quality
        while search_attempts < max_search_retries:
            search_attempts += 1
            if ui_callback and search_attempts > 1: ui_callback(f"搜索重试 ({search_attempts}/{max_search_retries}) - {current_task_label}", f"尝试为子任务 '{str(subtask)[:30]}...' 重新生成关键词并搜索",0,0)
            t0_key = time.time()
            keyword_prompt_addition = "" 
            if keywords_history: keyword_prompt_addition = f"先前尝试过的关键词组合 {keywords_history} 未能得到满意的搜索结果。请尝试生成与之前显著不同的关键词。"
            message_key = [
                SystemMessage("""你是一名熟练的搜索引擎使用者。请根据用户的总体研究任务和当前正在处理的具体子任务/方面，生成3个最相关的搜索关键词。""" + keyword_prompt_addition + """输出格式必须是 Python 列表，例如：["关键词1", "关键词2", "关键词3"]"""),
                HumanMessage(f"总体研究任务：{query}\n当前子任务/方面：{subtask}\n请生成搜索关键词。")
            ]
            if ui_callback: ui_callback(f"为子任务生成关键词... ({current_task_label}, 尝试 {search_attempts})", f"正在为 '{str(subtask)[:30]}...' 生成关键词", 0, 0)
            response_key = await model.ainvoke(message_key)
            t1_key = time.time()
            tokens_key = _get_token_usage(response_key)
            content_key = response_key.content
            try:
                keywords_list = ast.literal_eval(content_key)
                if not isinstance(keywords_list, list): keywords_list = [content_key]
            except (ValueError, SyntaxError):
                keywords_list = [content_key]
                if ui_callback: ui_callback("解析关键词列表失败", {"子任务": subtask, "原始输出": content_key, "尝试次数": search_attempts}, tokens_key, t1_key - t0_key)
            keywords_history.append(keywords_list)
            if ui_callback: ui_callback(f"关键词生成完成 ({current_task_label}, 尝试 {search_attempts})", {"子任务": subtask, "关键词": keywords_list}, tokens_key, t1_key - t0_key)
            t0_search = time.time()
            search_info_for_attempt = []
            search_summary_ui = ""
            tavily_response_data = None
            if not tavily_client:
                error_msg = "Tavily Client 未初始化 (缺少 API 密钥)"
                if ui_callback: ui_callback("搜索错误", error_msg, 0, 0)
                search_info_for_attempt.append({"keyword": "N/A", "info": error_msg})
                search_summary_ui += error_msg + "\n"
                quality = "failed"
            else:
                combined_keywords = " ".join(keywords_list)
                if ui_callback: ui_callback(f"查找信息 (Tavily)... ({current_task_label}, 尝试 {search_attempts})", f"使用关键词 '{combined_keywords}' 进行搜索", 0, 0)
                try:
                    tavily_response_data = tavily_client.search(query=combined_keywords, search_depth="advanced", max_results=5, include_answer=True, include_raw_content=False, include_images=False)
                    search_info_for_attempt.append({"keyword": combined_keywords, "info": tavily_response_data})
                    summary_for_ui = f"Tavily 搜索 '{combined_keywords}' (尝试 {search_attempts}):\n"
                    if tavily_response_data.get("answer"): summary_for_ui += f"**综合答案:** {tavily_response_data['answer']}\n"
                    summary_for_ui += "**相关结果:**\n"
                    for idx, res in enumerate(tavily_response_data.get("results", [])[:3]): summary_for_ui += f"  {idx+1}. {res.get('title', 'N/A')}: {res.get('content', 'N/A')[:100]}...\n"
                    search_summary_ui += summary_for_ui
                except Exception as e:
                    error_info = f"Tavily 搜索 '{combined_keywords}' (尝试 {search_attempts}) 时发生错误: {e}"
                    if ui_callback: ui_callback("Tavily 搜索错误", error_info, 0, 0)
                    search_info_for_attempt.append({"keyword": combined_keywords, "info": error_info})
                    search_summary_ui += error_info + "\n"
                    tavily_response_data = {"error": str(e)}
            t1_search = time.time()
            if ui_callback: ui_callback(f"查找信息完成 ({current_task_label}, 尝试 {search_attempts})", search_summary_ui, 0, t1_search - t0_search)
            if tavily_response_data and not tavily_response_data.get("error"):
                quality = await assess_search_quality(model, query, subtask, keywords_list, tavily_response_data, ui_callback)
            else:
                quality = "failed"
                if ui_callback: ui_callback("搜索质量评估跳过", f"由于搜索错误，子任务 '{str(subtask)[:30]}...' 的搜索结果质量记为 failed", 0, 0)
            current_subtask_search_results = search_info_for_attempt
            if quality in ["excellent", "good"]:
                if ui_callback: ui_callback("搜索结果接受", f"子任务 '{str(subtask)[:30]}...' 的搜索结果质量达标 ({quality.upper()})",0,0)
                break
            elif search_attempts >= max_search_retries:
                if ui_callback: ui_callback("搜索重试达到上限", f"子任务 '{str(subtask)[:30]}...' 搜索重试次数已达上限，接受当前结果 (质量: {quality.upper()})",0,0)
                break
        aggregated_results.append({"subtask": subtask, "final_search_quality": quality, "search_attempts_taken": search_attempts, "search_results": current_subtask_search_results})

    # --- 3. 最终报告汇总 --- (强化 Prompt 对 Markdown 的要求)
    t0_final_report = time.time()
    final_report_system_prompt = ("""
你是一位顶级的分析报告撰写专家。你收到了用户的原始研究任务、该任务分解后的子任务列表，以及通过Tavily搜索获得的每个子任务的相关信息（包括搜索质量评估）。
你的目标是整合所有这些信息，撰写一份结构清晰、内容详尽、表达丰富、逻辑连贯的最终研究报告。
报告应回答用户的初始任务，并覆盖所有分解出的子任务/方面。

**报告撰写指南:**
1.  **最大限度地利用多样化的Markdown格式。** 包括但不限于：
    *   层级清晰的标题（例如：`# 主标题`, `## 子任务1：[子任务名称]`, `### 主要发现`, `#### 详细点`）。
    *   用于强调关键点或关键词的**粗体**、*斜体*和 `代码片段`。
    *   用于组织信息的无序列表（`-` 或 `*`）和有序列表。
    *   使用引用块 (`>`) 来突出显示搜索结果中的直接引述或重要论述。
    *   如果适用，使用Markdown表格来组织信息。
2.  **内容的深化与整合**: 针对每个子任务，基于提供的Tavily搜索结果（特别是`answer`字段和`results`中的各个摘要），不仅仅是罗列信息，而是要深入挖掘和分析，并整合来自不同信息源的洞见。如果可能，请指出子任务之间的关联性。
3.  **结构**: 整个报告应具有逻辑流程，包括清晰的引言（背景和目的）、每个子任务的详细分析（每个都像一个小章节一样对待），以及一个强有力的结论/总结（概述主要发现，如果可能，包括未来的展望或建议）。
4.  **考虑搜索质量**: 如果某些子任务的搜索质量被标记为"poor"或"failed"，请在报告中明确指出相关信息可能缺失、有限，或需要谨慎解读。
5.  **尝试视觉化表达**: 如果内容允许，尝试使用基于文本的图表（例如：用字符表示的简单条形图或树状结构）或Mermaid.js语法来生成图表，以更清晰地传达信息。Mermaid代码务必用 ```mermaid ... ``` 代码块包裹。例如：
    ```mermaid
    graph TD;
        A-->B;
        A-->C;
        B-->D;
        C-->D;
    ```
    这有助于展示复杂关系或流程。
6.  **满足指定的字数要求**（如果提供，例如：1500字以上），提供足够详细的信息，使内容充实。避免仅仅为了凑字数而进行冗余描述。

请直接输出最终报告的内容。
""")

    # 从 aggregated_results 中提取更易读的信息给 HumanMessage
    # 这有助于减少传递给LLM的原始JSON的复杂性，并突出重点
    readable_aggregated_info = []
    for res in aggregated_results:
        subtask_info = f"子任务: {res['subtask']}\n搜索质量: {res['final_search_quality']}\n试行次数: {res['search_attempts_taken']}\n"
        search_details = ""
        # 提取每个尝试的Tavily answer和部分results
        if res.get('search_results'):
            for attempt_idx, attempt in enumerate(res['search_results']): # search_results is a list of attempts (usually 1)
                if attempt.get('info') and isinstance(attempt['info'], dict):
                    tavily_data = attempt['info']
                    if tavily_data.get('answer'):
                        search_details += f"  Tavily的回答 (试行 {attempt_idx+1}): {tavily_data['answer']}\n"
                    if tavily_data.get('results'):
                        search_details += f"  相关结果 (试行 {attempt_idx+1}的Top3):\n"
                        for r_idx, r_item in enumerate(tavily_data['results'][:3]):
                            search_details += f"    {r_idx+1}. {r_item.get('title', 'N/A')}: {r_item.get('content', 'N/A')[:150]}...\n"
                elif attempt.get('info'): # 如果info是错误字符串
                     search_details += f"  搜索错误 (试行 {attempt_idx+1}): {attempt['info']}\n"
        readable_aggregated_info.append(subtask_info + search_details)
    
    aggregated_info_for_prompt = "\n\n".join(readable_aggregated_info)

    final_report_human_prompt = f"""用户最初的研究任务：{query}

分解后的子任务/方面列表：
{json.dumps(refined_subtasks, ensure_ascii=False, indent=2)}

各子任务/方面搜索到的信息摘要：
{aggregated_info_for_prompt}

以上所有信息基础上，撰写最终研究报告。使用Markdown格式，并遵循指定指示。"""

    final_report_message = [
        SystemMessage(final_report_system_prompt),
        HumanMessage(final_report_human_prompt)
    ]
    if ui_callback: ui_callback("总结所有信息...", "正在综合所有子任务的搜索结果生成最终报告", 0, 0)
    response_final_report = await model.ainvoke(final_report_message)
    t1_final_report = time.time()
    tokens_final_report = _get_token_usage(response_final_report)
    final_report = response_final_report.content
    if ui_callback: ui_callback("生成最终报告", final_report, tokens_final_report, t1_final_report - t0_final_report)

    return {'task': query, 'report': final_report}

# 仅在直接运行此脚本时执行
if __name__ == "__main__":
    # 定义一个简单的打印回调
    def print_callback(step_name, content, token_used=0, time_taken=0):
        print("="*50)
        print(f"状态: {step_name} (耗时: {time_taken:.2f}s | Tokens: {token_used})")
        if isinstance(content, (dict, list)):
            print(f"内容: \n{json.dumps(content, ensure_ascii=False, indent=2)}")
        else:
            print(f"内容: {content}")
        print("="*50)

    print("--- deep_research.py 作为主脚本运行 (仅用于本地测试) ---")

    if not tavily_client:
        print("错误：Tavily 客户端未初始化，请确保 TAVILY_API_KEY 已设置。")
    else:
        # --- 为命令行测试创建模型 ---
        try:
            test_model_name = "qwen-max" # 或从环境变量/参数获取
            test_temperature = 0.7
            print(f"命令行测试，尝试创建模型: {test_model_name}, 温度: {test_temperature}")
            cli_model = create_llm(model_name=test_model_name, temperature=test_temperature)
            print(f"模型 {test_model_name} 创建成功。")
        except ValueError as e:
            print(f"创建测试模型时出错: {e}")
            cli_model = None
        except Exception as e:
            print(f"创建测试模型时发生未知错误: {e}")
            cli_model = None

        if cli_model:
            # 移除 input()，改为使用一个固定的测试查询
            # test_query = input("请输入测试查询：\n")
            test_query = "分析一下最近关于人工智能的重大新闻"
            print(f"使用固定测试查询: '{test_query}'")
            print("开始执行 cell 函数...")
            try:
                 asyncio.run(cell(cli_model, test_query, ui_callback=print_callback))
                 print("cell 函数执行完毕。")
            except Exception as cell_exec_error:
                 print(f"执行 cell 函数时出错: {cell_exec_error}")
        else:
            print("无法创建测试模型，无法执行 cell 函数。")

    print("--- deep_research.py 主脚本测试结束 ---")
