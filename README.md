# 💡 RAG-Gemini-KB: 本地知识库问答系统

这是一个基于 LlamaIndex、Gemini 1.5 Pro 和 FastAPI 构建的、拥有现代化 Web 界面的本地知识库问答（RAG）项目。你可以将自己的 PDF 文档加入知识库，并通过聊天界面进行智能问答。

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red?logo=streamlit)](https://streamlit.io/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10-blueviolet)](https://www.llamaindex.ai/)

## 📝 项目简介

本项目旨在提供一个完整的、开箱即用的 RAG 解决方案，其核心流程如下：

1.  **数据处理 :** 使用 `unstructured` 库高精度地解析 PDF 文档（包括表格），并通过 `LlamaIndex` 的 `SemanticSplitterNodeParser` 进行语义感知的智能文本分块。
2.  **向量化与存储 :** 使用 `BAAI/bge-large-zh-v1.5` 这个强大的中英双语模型将文本块转换为向量，并存储在本地的 `ChromaDB` 向量数据库中。
3.  **后端服务:** 基于 `FastAPI` 构建 API 服务，接收查询请求，通过 RAG 流程（检索、增强、生成）调用 `Google Gemini 1.5 Pro` 模型生成答案及来源。
4.  **前端界面:** 使用 `Streamlit` 创建一个交互式的聊天应用，用户可以方便地进行提问和查看结果。

## ✨ 主要特性

* **先进的 RAG 流程:** 结合了 LlamaIndex 的强大功能和 Gemini 1.5 Pro 的卓越理解生成能力。
* **高质量文档解析:** 支持 `hi_res` 高分辨率策略解析 PDF，能有效提取文本和表格。
* **智能语义分块:** 采用基于 LLM 的语义分块器，相比传统固定长度分块，能更好地保留上下文完整性。
* **中英双语优化:** 选用的 `bge-large-zh-v1.5` 嵌入模型在中英文上均有出色表现。
* **前后端分离:** API 服务与 UI 界面分离，易于独立部署和扩展。
* **本地持久化:** 所有知识库数据均存储在本地的 ChromaDB 中，方便管理和迁移。

## 🏛️ 项目结构

```
.
├── .gitignore          # Git 忽略配置
├── README.md           # 就是你正在看的这个文件
├── requirements.txt    # Python 依赖包列表
├── .env.example        # 环境变量模板
├── data/               # 存放你的源 PDF 文档
│   └── example.pdf
├── main.py             # FastAPI 后端服务
├── add_to_index.py     # 数据处理与入库脚本
└── app.py              # Streamlit 前端应用
```

## 🚀 快速开始

请按照以下步骤在你的本地环境中运行本项目。

### 1. 先决条件

* Python 3.9 或更高版本
* Git
* (可选，但推荐) 一个可以正常工作的网络代理，因为需要访问 Google 和 Hugging Face 的服务。
* **系统依赖 (用于 `unstructured`):**
    * **Debian/Ubuntu:** `sudo apt-get update && sudo apt-get install -y poppler-utils tesseract-ocr`
    * **macOS:** `brew install poppler tesseract`

### 2. 安装与配置

**1. 克隆仓库**
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
```

**2. 创建并激活虚拟环境**
```bash
python -m venv venv
# Windows
# venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

**3. 安装依赖**
```bash
pip install -r requirements.txt
```

**4. 配置环境变量**
复制模板文件，并填入你自己的 API 密钥。
```bash
cp .env.example .env
```
然后编辑 `.env` 文件:
```dotenv
# .env
# 替换成你自己的 Google API Key
GOOGLE_API_KEY="AIzaSy...your...key" 
```
> **注意:** 脚本中硬编码了代理地址 `http://127.0.0.1:7897`。如果你的代理地址不同，请在 `main.py` 和 `add_to_index.py` 中修改 `proxy_url` 变量。

### 3. 运行流程

**步骤一: 添加并处理你的文档**

1.  将你想要查询的 PDF 文件放入 `data/` 目录下。
2.  运行数据处理脚本，它会读取 `data` 目录下的所有 PDF，处理后存入本地的向量数据库中 (`./chroma_db/` 目录会自动创建)。

```bash
python add_to_index.py --dir data
```
你会看到文件被逐一处理，并显示入库后的文档块总数。

**步骤二: 启动后端 API 服务**

打开一个新的终端窗口，运行 FastAPI 服务。
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
服务启动后，你可以访问 http://127.0.0.1:8000/docs 查看自动生成的 API 文档。

**步骤三: 启动前端 Web 应用**

再打开一个新的终端窗口，运行 Streamlit 应用。
```bash
streamlit run app.py
```
应用启动后，会自动在浏览器中打开一个新页面 (通常是 http://localhost:8501)。

**步骤四: 开始查询！**

在 Streamlit 聊天界面中，输入你的问题，开始与你的本地知识库对话吧！

## 🔧 API 端点

本项目提供了一个核心 API 端点供程序化调用。

* **Endpoint:** `/query`
* **Method:** `POST`
* **Request Body:**
    ```json
    {
      "question": "你的问题是什么？"
    }
    ```
* **Success Response (200 OK):**
    ```json
    {
      "answer": "这是由 Gemini 生成的答案...",
      "sources": [
        {
          "content": "这是引用来源的文本块内容...",
          "score": 0.85
        }
      ]
    }
    ```

## 后续拓展内容

* [X] **支持更多文件类型:** 扩展 `add_to_index.py` 以支持如 `.docx`, `.md`, `.txt` 等更多格式。
* [ ] **优化会话历史:** 将聊天历史记忆融入 RAG 检索，实现更连贯的多轮对话。
* [ ] **前端功能增强:** 增加清除历史、重新生成答案等功能。

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源。
