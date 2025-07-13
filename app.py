import streamlit as st
import requests
import os

# --- 配置 ---
# 从环境变量读取API地址，方便部署，如果未设置则使用默认值
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE_URL}/query"

st.set_page_config(page_title="RAG 知识库", layout="wide", page_icon="💡")
st.title("💡 智能知识库查询助手")
st.caption("由 LlamaIndex, FastAPI, 和 Streamlit 强力驱动")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！有什么可以帮您从知识库中查找的吗？"}]

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("在这里输入您的问题..."):
    # 显示用户消息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调用后端API
    with st.chat_message("assistant"):
        with st.spinner("正在思考与检索..."):
            try:
                response = requests.post(API_URL, json={"question": prompt}, timeout=180) # 延长超时时间
                response.raise_for_status()
                data = response.json()
                answer = data.get("answer")
                sources = data.get("sources", [])
                
                st.markdown(answer)
                with st.expander("查看引用来源"):
                    if sources:
                        for i, source in enumerate(sources):
                            st.info(f"来源 {i+1} (相似度: {source['score']:.4f})\n\n" + source['content'])
                    else:
                        st.write("本次回答没有直接引用具体的文本块。")
                
                # 将完整回答（包括来源）存入历史记录
                full_response_for_history = answer
                st.session_state.messages.append({"role": "assistant", "content": full_response_for_history})

            except requests.exceptions.RequestException as e:
                st.error(f"无法连接到后端API: {e}")