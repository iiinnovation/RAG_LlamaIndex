import os
import argparse
import chromadb
from dotenv import load_dotenv

# --- 导入所需模块 ---
from unstructured.partition.pdf import partition_pdf
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai.base import GoogleGenAI

# --- 配置部分 (与主服务保持一致) ---
load_dotenv()
proxy_url = "http://127.0.0.1:7897"
os.environ['HTTPS_PROXY'] = proxy_url

EMBED_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
UNSTRUCTURED_STRATEGY = "hi_res"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base_collection"

def main():
    parser = argparse.ArgumentParser(description="向RAG知识库中添加新的PDF文档。")
    parser.add_argument("--dir", type=str, required=True, help="包含要处理的PDF文件的目录路径。")
    args = parser.parse_args()

    # --- 模型配置 (此脚本需要嵌入和LLM用于分块) ---
    print("[*] 正在加载所需模型...")
    try:
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("错误: 未在.env文件中找到 GOOGLE_API_KEY")
        Settings.llm = GoogleGenAI(model_name="models/gemini-1.5-pro-latest", api_key=google_api_key)
        print("[*] 模型加载成功。")
    except Exception as e:
        print(f"[!] 模型加载失败: {e}")
        return

    # --- 连接数据库并加载索引 ---
    print("[*] 连接到ChromaDB...")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # --- 开始处理文件 ---
    directory_path = args.dir
    print(f"[*] 开始批量处理目录: {directory_path}")
    pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]

    for pdf_path in pdf_files:
        print(f"\n--- 正在处理文件: {os.path.basename(pdf_path)} ---")
        try:
            # 步骤 1: 解析PDF
            elements = partition_pdf(filename=pdf_path, strategy=UNSTRUCTURED_STRATEGY, infer_table_structure=True, languages=['chi_sim', 'eng'])
            
            # 步骤 2: 组合文本
            text_blocks = []
            for el in elements:
                if type(el).__name__ in ["Title", "NarrativeText", "ListItem"]:
                    text_blocks.append(el.text)
                elif type(el).__name__ == "Table" and el.text_as_html:
                    text_blocks.append(f"<table>{el.text_as_html}</table>")
            full_text = "\n\n".join(text_blocks)
            
            # 步骤 3: 语义分块 (使用最终正确的构造函数)
            splitter = SemanticSplitterNodeParser(
                buffer_size=1, 
                breakpoint_percentile_threshold=95,
                embed_model=Settings.embed_model,
                llm=Settings.llm
            )
            nodes = splitter.get_nodes_from_documents([Document(text=full_text)])
            
            # 步骤 4: 插入索引
            index.insert_nodes(nodes)
            print(f"[*] 成功处理并插入! 集合中文档总数现在为: {chroma_collection.count()}")

        except Exception as e:
            print(f"[!] 处理文件 {os.path.basename(pdf_path)} 时出错: {e}")

if __name__ == "__main__":
    main()