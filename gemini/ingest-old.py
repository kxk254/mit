import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

load_dotenv()

import os
import glob




GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "./docs")
# CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", r"H:\dev\mit\gemini\docs")  # Use absolute path for reliability
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", r"H:\dev\mit\gemini\chroma_db")

# DOCUMENT_PATH = os.path.abspath(DOCUMENT_PATH)  # Convert to absolute path
print("DOC PATH===", DOCUMENT_PATH)


# 対応するドキュメント形式とローダーのマッピング

LOADER_MAPPING = {
    ".pdf": PDFMinerLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
}

def load_documents(directory_path):
    """指定されたディレクトリから対応する形式のドキュメントをロードする"""
    all_docs = []
    for ext, loader_class in LOADER_MAPPING.items():
        glob_pattern = f"**/*{ext}"  # Relative pattern
        print(f"Loading {ext} files from {directory_path}...")
        files = glob.glob(os.path.join(directory_path, glob_pattern), recursive=True)
        print(f"Found {len(files)} {ext} files: {files}")
        try:
            loader = DirectoryLoader(
                directory_path,
                glob=glob_pattern,
                loader_cls=loader_class,
                recursive=True,
                show_progress=True
            )
            loaded_docs = loader.load()
            print(f"Loaded {len(loaded_docs)} {ext} files.")
            all_docs.extend(loaded_docs)
        except Exception as e:
            print(f"Error loading {ext} files: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping {ext} files.")
    return all_docs

def split_documents(documents):
    """ドキュメントをチャンクに分割する"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def embed_documents(documents):
    """ドキュメントチャンクをベクトル化する"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings

def create_or_update_vector_store(documents, embeddings, persist_directory):
    """ベクトルデータベースを作成または更新する"""
    if not documents:
        print("No documents to process. Skipping vector store creation/update.")
        return
    print(f"Creating/updating vector store in {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print("Vector store creation/update complete.")

def main():
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        return
    if not os.path.exists(DOCUMENT_PATH):
        print(f"Error: Document directory not found at {DOCUMENT_PATH}.")
        return

    print(f"Loading documents from: {DOCUMENT_PATH}")
    documents = load_documents(DOCUMENT_PATH)
    print(f"Splitting {len(documents)} documents...")
    split_docs = split_documents(documents)
    print(f"Total {len(split_docs)} document chunks created.")
    print("Creating embedding function...")
    embeddings = embed_documents(split_docs)
    print("Creating or updating vector store...")
    create_or_update_vector_store(split_docs, embeddings, CHROMA_DB_PATH)
    print("Ingestion process finished.")

if __name__ == "__main__":
    main()