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


async def cell(model, query, ui_callback=None):
    """
    修改后的思考单元：
    1. 对用户 query 进行一次性细化，分解为子任务列表。
    2. 顺序对每个子任务：生成关键词 -> Tavily 搜索。
    3. 收集所有子任务的搜索结果。
    4. 基于所有结果生成最终报告。
    """
    aggregated_results = [] # 存储所有子任务的搜索结果

    # --- 1. 一次性任务细化 --- (不再递归)
    t0 = time.time()
    refine_message = [
        SystemMessage("""
                    你是一个经验丰富的项目经理。你需要根据用户提出的研究任务，将其分解为一系列具体、可执行的子任务或需要研究的关键方面。
                    分解的粒度应该适中，能够覆盖用户需求的核心，避免过于琐碎或过于宏大。
                    请根据任务的复杂性决定分解出的子任务数量。
                    输出的内容必须是一个 Python 列表，例如：["子任务1", "方面2", "需要调研的点3", ...]
                    """),
        HumanMessage(f"请将以下用户研究任务进行分解：\n\n{query}")
    ]
    if ui_callback:
        ui_callback("细化任务中...", f"正在为 '{query[:50]}...' 分解子任务", 0, 0)
    response_refine = await model.ainvoke(refine_message)
    t1 = time.time()
    tokens_refine = _get_token_usage(response_refine)
    refined_subtasks_str = response_refine.content
    try:
        refined_subtasks = ast.literal_eval(refined_subtasks_str)
        if not isinstance(refined_subtasks, list):
            refined_subtasks = [refined_subtasks_str] # 如果解析出来不是列表，则整个作为一项
    except (ValueError, SyntaxError):
        refined_subtasks = [refined_subtasks_str] # 解析失败，整个字符串作为一个子任务
        if ui_callback:
             ui_callback("解析细化列表失败", f"无法解析LLM返回的列表: {refined_subtasks_str}", tokens_refine, t1 - t0)

    if ui_callback:
        ui_callback("任务细化完成", {"原始任务": query, "分解出的子任务/方面": refined_subtasks}, tokens_refine, t1 - t0)

    # --- 2. 顺序执行子任务搜索 --- (不再有评估和递归)
    for i, subtask in enumerate(refined_subtasks):
        current_task_label = f"子任务 {i+1}/{len(refined_subtasks)}: {str(subtask)[:30]}..."
        if ui_callback:
             ui_callback("处理子任务", f"开始处理: {current_task_label}", 0, 0)

        # --- 2a. 生成搜索关键词 ---
        t0_key = time.time()
        message_key = [
            SystemMessage("""
                        你是一名熟练的搜索引擎使用者。请根据用户的总体研究任务和当前正在处理的具体子任务/方面，生成3个最相关的搜索关键词。
                        输出格式必须是 Python 列表，例如：["关键词1", "关键词2", "关键词3"]
                        """),
            HumanMessage(f"总体研究任务：{query}\n当前子任务/方面：{subtask}\n请生成搜索关键词。")
        ]
        if ui_callback:
             ui_callback(f"为子任务生成关键词... ({current_task_label})", f"正在为 '{str(subtask)[:30]}...' 生成关键词", 0, 0)
        response_key = await model.ainvoke(message_key)
        t1_key = time.time()
        tokens_key = _get_token_usage(response_key)
        content_key = response_key.content
        try:
            keywords_list = ast.literal_eval(content_key)
            if not isinstance(keywords_list, list):
                keywords_list = [content_key]
        except (ValueError, SyntaxError):
            keywords_list = [content_key]
            if ui_callback:
                ui_callback("解析关键词列表失败", {"子任务": subtask, "原始输出": content_key}, tokens_key, t1_key - t0_key)

        if ui_callback:
            ui_callback(f"关键词生成完成 ({current_task_label})", {"子任务": subtask, "关键词": keywords_list}, tokens_key, t1_key - t0_key)

        # --- 2b. Tavily 搜索 ---
        t0_search = time.time()
        search_info_for_subtask = []
        search_summary_ui = ""

        if not tavily_client:
            error_msg = "Tavily Client 未初始化 (缺少 API 密钥)"
            if ui_callback:
                ui_callback("搜索错误", error_msg, 0, 0)
            search_info_for_subtask.append({"keyword": "N/A", "info": error_msg})
            search_summary_ui += error_msg + "\n"
        else:
            combined_keywords = " ".join(keywords_list)
            if ui_callback:
                 ui_callback(f"查找信息 (Tavily)... ({current_task_label})", f"使用关键词 '{combined_keywords}' 进行搜索", 0, 0)
            try:
                response = tavily_client.search(
                    query=combined_keywords,
                    search_depth="advanced",
                    max_results=5,
                    include_answer=True,
                    include_raw_content=False,
                    include_images=False
                )
                search_info_for_subtask.append({"keyword": combined_keywords, "info": response})

                # 准备 UI 摘要
                summary_for_ui = f"Tavily 搜索 '{combined_keywords}':\n"
                if response.get("answer"):
                     summary_for_ui += f"**综合答案:** {response['answer']}\n"
                summary_for_ui += "**相关结果:**\n"
                for idx, res in enumerate(response.get("results", [])[:3]):
                    summary_for_ui += f"  {idx+1}. {res.get('title', 'N/A')}: {res.get('content', 'N/A')[:100]}...\n"

                search_summary_ui += summary_for_ui

            except Exception as e:
                error_info = f"Tavily 搜索 '{combined_keywords}' 时发生错误: {e}"
                if ui_callback:
                     ui_callback("Tavily 搜索错误", error_info, 0, 0)
                search_info_for_subtask.append({"keyword": combined_keywords, "info": error_info})
                search_summary_ui += error_info + "\n"

        t1_search = time.time()
        if ui_callback:
            ui_callback(f"查找信息完成 ({current_task_label})", search_summary_ui, 0, t1_search - t0_search)

        # 存储这个子任务的搜索结果
        aggregated_results.append({
            "subtask": subtask,
            "search_results": search_info_for_subtask
        })

    # --- 3. 最终报告汇总 --- (移除质检和迭代)
    t0_final_report = time.time()
    final_report_message = [
        SystemMessage("""
                    你是顶级的分析报告撰写专家。你收到了用户最初的研究任务、该任务分解出的子任务列表、以及针对每个子任务通过 Tavily 搜索获得的信息（可能包含综合答案和多个结果条目）。
                    你的目标是综合所有这些信息，撰写一份结构清晰、内容详实、逻辑连贯的最终研究报告，确保报告回答了用户最初的任务，并覆盖了所有分解出的子任务/方面。
                    请直接输出最终的报告内容。
                    """),
        HumanMessage(f"""用户最初的研究任务：{query}

分解出的子任务/方面列表：
{json.dumps(refined_subtasks, ensure_ascii=False, indent=2)}

针对各子任务/方面搜索到的信息汇总：
{json.dumps(aggregated_results, ensure_ascii=False, indent=2)}

请基于以上所有信息，撰写最终的研究报告。""")
    ]
    if ui_callback:
        ui_callback("总结所有信息...", "正在综合所有子任务的搜索结果生成最终报告", 0, 0)
    response_final_report = await model.ainvoke(final_report_message)
    t1_final_report = time.time()
    tokens_final_report = _get_token_usage(response_final_report)
    final_report = response_final_report.content

    if ui_callback:
        # 显示完整报告或摘要给UI
        # report_summary = final_report[:1000] + "..." if len(final_report) > 1000 else final_report
        ui_callback("生成最终报告", final_report, tokens_final_report, t1_final_report - t0_final_report)

    return {'task': query, 'report': final_report} # 返回最终结果

# 仅在直接运行此脚本时执行
if __name__ == "__main__":
    # ... (print_callback remains the same) ...

    if not tavily_api_key:
        # 如果 Tavily client 是 None，这里直接退出或给出错误
        print("错误：Tavily 客户端未初始化，请确保 TAVILY_API_KEY 已设置。")
    else:
        # --- 为命令行测试创建模型 ---
        try:
            test_model_name = "qwen-max" # 或者从命令行参数获取
            test_temperature = 0.7
            print(f"命令行测试，使用模型: {test_model_name}, 温度: {test_temperature}")
            # create_llm 会检查所需的环境变量
            cli_model = create_llm(model_name=test_model_name, temperature=test_temperature)
        except ValueError as e:
            print(f"创建模型时出错: {e}")
            cli_model = None
        except Exception as e:
            print(f"创建模型时发生未知错误: {e}")
            cli_model = None

        if cli_model:
            test_query = input("请输入测试查询：\n")
            asyncio.run(cell(cli_model, test_query, ui_callback=print_callback))
        else:
            print("无法创建测试模型，退出。")
