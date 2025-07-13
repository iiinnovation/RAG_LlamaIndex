import os
import chromadb
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# --- LlamaIndex and other imports ---
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai.base import GoogleGenAI


# 1. 初始化和配置 (服务启动时执行一次)
# -- 加载环境变量 (.env 文件) --
load_dotenv()
print("[*] 正在加载环境变量...")

# -- 代理配置 --
# 注意: 在生产环境中，最好通过K8s/Docker等方式统一配置环境变量
proxy_url = "http://127.0.0.1:7897"
os.environ['HTTPS_PROXY'] = proxy_url
print(f"[*] 已配置网络代理: {proxy_url}")

# -- 模型和持久化路径配置 --
EMBED_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base_collection"

# -- 全局模型配置 --
print("[*] 正在配置LlamaIndex全局设置...")
try:
    # 配置嵌入模型
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    
    # 配置LLM (Gemini)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("错误: 未在.env文件中找到 GOOGLE_API_KEY")
        
    Settings.llm = GoogleGenAI(
        model_name="models/gemini-1.5-pro-latest",
        api_key=google_api_key
    )
    print("[*] 全局模型配置成功！")

except Exception as e:
    print(f"[!] 模型加载失败: {e}")
    exit(1)


# -- 加载索引和创建查询引擎 --
print("[*] 正在加载索引和创建查询引擎...")
try:
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine(similarity_top_k=5)
    print(f"[*] 查询引擎已就绪！知识库中现有 {chroma_collection.count()} 个文档块。")
except Exception as e:
    print(f"[!] 加载索引失败: {e}")
    exit(1)

# 2. 定义 FastAPI 应用 和 数据模型

app = FastAPI(
    title="RAG 知识库查询 API",
    description="一个用于查询本地知识库的API服务",
    version="1.0.0"
)

# Pydantic模型用于请求体校验
class QueryRequest(BaseModel):
    question: str

# Pydantic模型用于定义响应体结构
class SourceNode(BaseModel):
    content: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceNode]] = None

# 3. 创建 API 端点 (Endpoint)

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    接收一个问题，通过RAG流程返回答案和引用来源。
    """
    print(f"[*] 收到查询: {request.question}")
    response = query_engine.query(request.question)
    
    source_nodes = [
        SourceNode(content=node.get_content(), score=node.score)
        for node in response.source_nodes
    ]
    
    return QueryResponse(answer=str(response), sources=source_nodes)

@app.get("/")
def read_root():
    return {"message": "欢迎使用RAG知识库API，请向 /query 端点发送POST请求进行查询。"}