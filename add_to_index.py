import os
import argparse
import chromadb
from dotenv import load_dotenv

# --- 导入所需模块 ---
# 修改: 从 unstructured 中导入更多的解析器
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.md import partition_md
from unstructured.partition.text import partition_text

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

# 新增: 创建文件扩展名到解析函数的映射
# 这样可以轻松地添加更多文件类型
FILE_PARSERS = {
    ".pdf": partition_pdf,
    ".docx": partition_docx,
    ".md": partition_md,
    ".txt": partition_text,
    # 如果需要，可以继续添加，例如:
    # ".html": partition_html,
    # ".pptx": partition_pptx,
}

def main():
    parser = argparse.ArgumentParser(description="向RAG知识库中添加新的文档。")
    parser.add_argument("--dir", type=str, required=True, help="包含要处理的文档文件的目录路径。")
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
    
    # 修改: 获取所有支持的文件类型
    supported_extensions = tuple(FILE_PARSERS.keys())
    files_to_process = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(supported_extensions)]

    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        print(f"\n--- 正在处理文件: {file_name} ---")
        try:
            # 步骤 1: 动态选择解析器并解析文档
            file_ext = os.path.splitext(file_name)[1].lower()
            parser_func = FILE_PARSERS.get(file_ext)
            
            if not parser_func:
                print(f"[!] 跳过不支持的文件类型: {file_name}")
                continue
            
            # 注意：并非所有解析器都接受所有参数。
            # 'unstructured'库会自动忽略不适用的参数。
            # 例如，'strategy' 主要用于PDF，但在这里传递是安全的。
            elements = parser_func(
                filename=file_path, 
                strategy=UNSTRUCTURED_STRATEGY, 
                infer_table_structure=True, 
                languages=['chi_sim', 'eng']
            )
            
            # 步骤 2: 组合文本 (此逻辑保持不变，因为它适用于所有格式)
            text_blocks = []
            for el in elements:
                if type(el).__name__ in ["Title", "NarrativeText", "ListItem"]:
                    text_blocks.append(el.text)
                elif type(el).__name__ == "Table" and hasattr(el.metadata, 'text_as_html') and el.metadata.text_as_html:
                    text_blocks.append(f"<table>{el.metadata.text_as_html}</table>")
            full_text = "\n\n".join(text_blocks)
            
            # 步骤 3: 语义分块 (此逻辑保持不变)
            splitter = SemanticSplitterNodeParser(
                buffer_size=1, 
                breakpoint_percentile_threshold=95,
                embed_model=Settings.embed_model,
                llm=Settings.llm
            )
            nodes = splitter.get_nodes_from_documents([Document(text=full_text, metadata={"file_name": file_name})])
            
            # 步骤 4: 插入索引 (此逻辑保持不变)
            index.insert_nodes(nodes)
            print(f"[*] 成功处理并插入! 集合中文档总数现在为: {chroma_collection.count()}")

        except Exception as e:
            print(f"[!] 处理文件 {file_name} 时出错: {e}")

if __name__ == "__main__":
    main()