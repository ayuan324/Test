import streamlit as st
import time
import asyncio
from deep_research import cell, create_llm # 导入 cell 和新的 create_llm
import json # 导入 json

# 设置页面配置
st.set_page_config(page_title="Deep Research 智能研究助手", layout="wide")
st.title("📊 Deep Research 智能研究助手")

# 初始化 session_state (增加 LLM 配置)
if 'start_time' not in st.session_state:
    st.session_state['start_time'] = 0
if 'token_total' not in st.session_state:
    st.session_state['token_total'] = 0
if 'steps' not in st.session_state:
    st.session_state['steps'] = [] # 存储 (step_name, content, token_used, time_taken)
if 'final_report' not in st.session_state:
    st.session_state['final_report'] = None
if 'running' not in st.session_state:
    st.session_state['running'] = False
# LLM 默认配置
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = "qwen-max"
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0.7

# UI 回调函数 (修改后)
def ui_callback(step_name, content, token_used=0, time_taken=0):
    """用于从后端逻辑接收更新并显示在UI上 (不再调用 rerun)"""
    st.session_state['steps'].append((step_name, content, token_used, time_taken))
    st.session_state['token_total'] += token_used
    # 注意：这里不再调用 st.rerun()。UI 更新将依赖于 Streamlit 的自然流程
    # 或通过更新下面的 placeholder 实现。

# --- UI 布局 ---
col1, col2 = st.columns([2, 1]) # 主内容区和侧边栏

with col1:
    st.header("研究过程")
    # 输入区域
    query = st.text_input("请输入你的研究问题：", key="query_input", disabled=st.session_state['running'])
    start_btn = st.button("🚀 开始分析", key="start_button", disabled=st.session_state['running'] or not query)

    # 过程展示区 (使用 placeholder)
    steps_placeholder = st.empty() # 创建一个空占位符

    # 最终报告区
    if st.session_state['final_report']:
        st.header("✅ 最终报告")
        st.markdown(st.session_state['final_report'])

with col2:
    st.header("配置")
    # --- LLM 配置 UI --- (新增)
    st.selectbox(
        "选择语言模型:",
        options=["qwen-max", "deepseek-chat", "azure-gpt-4o"], # 可选模型列表
        key="selected_model",
        disabled=st.session_state['running'],
        help="确保所选模型的 API Key 和 Endpoint 已在环境中正确配置。"
    )
    st.slider(
        "模型温度 (Temperature):",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="temperature",
        disabled=st.session_state['running'],
        help="值越高，输出越随机；值越低，输出越确定。"
    )
    st.markdown("---") # 分隔线
    # ---------------------

    st.header("统计信息")
    # 统计信息区 (使用 placeholder 保证实时更新)
    stats_placeholder = st.empty()

    st.info("Agent 会逐步分解任务、搜索信息并生成报告。")

# --- 后端逻辑调用 ---
if start_btn and query and not st.session_state['running']:
    # 重置状态
    st.session_state['start_time'] = time.time()
    st.session_state['token_total'] = 0
    st.session_state['steps'] = []
    st.session_state['final_report'] = None
    st.session_state['running'] = True
    st.rerun() # 仅在开始时 rerun 一次以禁用输入

elif st.session_state['running'] and not st.session_state['final_report']:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            raise ex

    query_to_run = st.session_state['query_input']
    with st.spinner("🤖 Agent 正在工作中，请稍候..."):
        # 在调用 cell 之前，更新一次统计信息
        with stats_placeholder.container():
            total_time = time.time() - st.session_state.get('start_time', time.time())
            st.metric("总耗时", f"{total_time:.2f} 秒")
            st.metric("总Token消耗", f"{st.session_state.get('token_total', 0)}")

        try:
            # --- 创建 LLM 实例 (新增) ---
            try:
                current_model = create_llm(
                    model_name=st.session_state['selected_model'],
                    temperature=st.session_state['temperature']
                    # 如需传递 max_tokens 等，可在此处添加
                )
            except Exception as model_creation_error:
                st.error(f"创建语言模型失败: {model_creation_error}")
                st.session_state['running'] = False
                st.rerun()
                st.stop() # 停止脚本执行
            # ---------------------------

            # 定义一个包装回调，用于更新 placeholder (修改内部逻辑)
            def update_ui_display(*args):
                ui_callback(*args) # 先用旧回调更新 state

                # -- Helper to get emoji based on step name --
                def get_step_emoji(step_name):
                    name_lower = step_name.lower()
                    if "失败" in name_lower or "错误" in name_lower:
                        return "⚠️"
                    elif "细化任务中" in name_lower:
                        return "🧠"
                    elif "任务细化完成" in name_lower:
                        return "🧩"
                    elif "处理子任务" in name_lower:
                        return "⚙️"
                    elif "关键词" in name_lower:
                        return "🔑"
                    elif "查找信息 (tavily)" in name_lower:
                        return "🔍"
                    elif "查找信息完成" in name_lower:
                        return "📄"
                    elif "总结所有信息" in name_lower:
                        return "✍️"
                    elif "生成最终报告" in name_lower:
                        return "✅"
                    else:
                        return "➡️" # Default

                # 更新步骤显示区域 (使用 expander)
                with steps_placeholder.container():
                    st.subheader("思考过程链") # 可以加个标题
                    if not st.session_state['steps']:
                        st.write("等待 Agent 开始...")
                    else:
                        num_steps = len(st.session_state['steps'])
                        # 反向遍历步骤列表
                        for i in range(num_steps - 1, -1, -1):
                            s_name, s_content, s_tokens, s_time = st.session_state['steps'][i]
                            # is_last_step 在反向遍历中变为 is_first_displayed_step
                            is_newest_step = (i == num_steps - 1)
                            emoji = get_step_emoji(s_name)
                            # 标签现在使用反向计数或保持正向计数？保持正向易于理解
                            expander_label = f"{emoji} {i + 1}. {s_name}" # 保持原有编号
                            with st.expander(expander_label, expanded=is_newest_step):
                                # 在展开内容中显示详细信息和耗时/token
                                st.markdown(f"_耗时: {s_time:.2f}s | Tokens: {s_tokens}_ ")
                                st.markdown("---") # 分隔符
                                if isinstance(s_content, (dict, list)):
                                    try:
                                        st.code(json.dumps(s_content, ensure_ascii=False, indent=2), language='json')
                                    except TypeError:
                                        st.write(s_content) # 回退显示原始格式
                                else:
                                    st.markdown(s_content)

                # 更新统计信息区域
                with stats_placeholder.container():
                    total_time = time.time() - st.session_state.get('start_time', time.time())
                    st.metric("总耗时", f"{total_time:.2f} 秒")
                    st.metric("总Token消耗", f"{st.session_state.get('token_total', 0)}")

            # 运行 cell，传入配置好的 LLM 实例
            result = loop.run_until_complete(cell(current_model, query_to_run, ui_callback=update_ui_display))
            st.session_state['final_report'] = result['report']
            st.session_state['running'] = False
            st.rerun() # 任务完成后 rerun 以显示最终报告并解锁输入
        except Exception as e:
            st.error(f"分析过程中出现错误: {e}")
            st.session_state['running'] = False
            st.rerun()
else:
     # 如果没有运行，也显示一次统计信息
     with stats_placeholder.container():
         total_time = time.time() - st.session_state.get('start_time', 0)
         st.metric("总耗时", f"{total_time:.2f} 秒" if st.session_state.get('start_time', 0) > 0 else "N/A")
         st.metric("总Token消耗", f"{st.session_state.get('token_total', 0)}")
     # 清空步骤区域如果不在运行中且没有最终报告
     if not st.session_state['running'] and not st.session_state['final_report']:
         steps_placeholder.empty() 