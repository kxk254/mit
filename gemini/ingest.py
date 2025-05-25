import os
import glob
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,  # 追加: URLからの読み込み用
)
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders.pdf import PDFMinerLoader # PDFMinerLoader を明示的にインポート

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings # 新しいインポートパス

import traceback # エラー詳細表示用

# .env ファイルから環境変数を読み込む
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "./docs") # 元のコメントアウト
# CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db") # 元のコメントアウト

# ドキュメントディレクトリとChromaDBのパス（絶対パス推奨）
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", r"H:\dev\mit\gemini\docs") # 必要に応じてパスを変更
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", r"H:\dev\mit\gemini\chroma_db") # 必要に応じてパスを変更

print("DOC PATH===", DOCUMENT_PATH)
print("CHROMA DB PATH===", CHROMA_DB_PATH)


# 対応するローカルドキュメント形式とローダーのマッピング
# WebBaseLoaderはDirectoryLoaderとは異なる方法で扱うため、ここには含めません。
LOADER_MAPPING = {
    ".pdf": PDFMinerLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
}


URLS_TO_INGEST = [
    "https://www.fsa.go.jp/status/s_jirei/s_jirei.xlsx",
    "https://www.fsa.go.jp/news/30/wp/supervisory_approaches_revised.pdf",
    "https://www.fsa.go.jp/news/30/dp/compliance_report.pdf",
    "https://www.fsa.go.jp/manual/manualj/manual_yokin/07.pdf",
    "https://www.fsa.go.jp/news/18/ginkou/20070216-4/04.pdf",
    "https://www.smfg.co.jp/investor/financial/disclosure/h2607_c_disc_pdf/h2607c_17.pdf",
    "https://www.shugiin.go.jp/internet/itdb_kaigirokua.nsf/html/kaigirokua/009516120041119011.htm",
    "https://dl.ndl.go.jp/view/prepareDownload?itemId=info%3Andljp%2Fpid%2F9217242&contentNo=1",
    # 必要に応じてURLを追加してください
]


# ここにデータベースに含めたいURLのリストを定義します
# 例：
# URLS_TO_INGEST = [
#     "https://ja.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%9F%A5%E8%83%BD",
#     "https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r05/html/nd111120.html",
#     # 必要に応じてURLを追加してください
# ]
# 初期状態では空のリストにしておきます。必要に応じて追加してください。


def load_documents(directory_path, urls_to_ingest):
    """
    指定されたディレクトリ内のローカルドキュメントと、指定されたURLからコンテンツを読み込みます。

    Args:
        directory_path (str): ローカルドキュメントのディレクトリパス。
        urls_to_ingest (list): 読み込むURLのリスト。

    Returns:
        list: 読み込まれた全てのDocumentオブジェクトのリスト。
    """
    all_docs = []

    # 1. ローカルドキュメントの読み込み
    if directory_path and os.path.exists(directory_path):
        print(f"Loading local documents from: {directory_path}...")
        for ext, loader_class in LOADER_MAPPING.items():
            glob_pattern = f"**/*{ext}"  # Relative pattern
            print(f"  Loading {ext} files...")
            files = glob.glob(os.path.join(directory_path, glob_pattern), recursive=True)
            print(f"  Found {len(files)} {ext} files.")
            if files:
                try:
                    loader = DirectoryLoader(
                        directory_path,
                        glob=glob_pattern,
                        loader_cls=loader_class,
                        recursive=True,
                        show_progress=True
                    )
                    loaded_docs = loader.load()
                    print(f"  Loaded {len(loaded_docs)} {ext} documents.")
                    all_docs.extend(loaded_docs)
                except Exception as e:
                    print(f"  Error loading {ext} files: {e}")
                    # traceback.print_exc() # デバッグが必要な場合にコメント解除
                    print(f"  Skipping {ext} files.")
            else:
                 print(f"  No {ext} files found.")
    elif directory_path:
         print(f"Local document directory not found at {directory_path}. Skipping local document loading.")

    # 2. URLからのコンテンツ読み込み
    if urls_to_ingest:
        print(f"Loading documents from URLs: {urls_to_ingest}...")
        try:
            # WebBaseLoader はURLのリストを直接受け取れます
            loader = WebBaseLoader(urls_to_ingest)
            url_docs = loader.load()
            print(f"Loaded {len(url_docs)} documents from URLs.")
            all_docs.extend(url_docs)
        except Exception as e:
            print(f"Error loading documents from URLs: {e}")
            # traceback.print_exc() # デバッグが必要な場合にコメント解除
            print("Skipping URL loading.")
    else:
        print("No URLs specified for ingestion.")

    print(f"Finished loading. Total documents loaded: {len(all_docs)}")
    return all_docs

def split_documents(documents):
    """
    ドキュメントをチャンクに分割します。
    """
    if not documents:
        print("No documents to split.")
        return []
    print(f"Splitting {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    print(f"Total {len(split_docs)} document chunks created.")
    return split_docs

def embed_documents():
    """
    ドキュメント埋め込み用のエンベディング関数を作成します。
    """
    print("Creating embedding function...")
    # GoogleGenerativeAIEmbeddings は初期化時に API キーを使用
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    print("Embedding function created.")
    return embeddings

def create_or_update_vector_store(documents, embeddings, persist_directory):
    """
    ドキュメントチャンクとエンベディングを使用して、ChromaDBにベクトルストアを作成または更新します。
    """
    if not documents:
        print("No document chunks to process. Skipping vector store creation/update.")
        return
    print(f"Creating/updating vector store in {persist_directory}...")

    # 既存のストアがあればロードし、なければ新規作成
    # ただし、今回の実装では常にfrom_documentsで上書きまたは新規作成します。
    # 追記する場合は別のロジックが必要です。今回はシンプルに全てを読み込み直して作成します。
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # persist() は from_documents 内で自動的に呼ばれる場合もありますが、明示的に呼んでおくと安全です。
    # vectorstore.persist() # from_documents で自動的に呼ばれるためコメントアウト
    print("Vector store creation/update complete.")

def main():
    """
    メインの実行フロー。ドキュメント読み込み、分割、埋め込み、ベクトルストア作成を行います。
    """
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        print("Please create a .env file in the same directory and add GOOGLE_API_KEY='YOUR_API_KEY'")
        return

    # ドキュメントとURLをまとめて読み込む
    # load_documents 関数にディレクトリパスとURLリストの両方を渡します
    documents = load_documents(DOCUMENT_PATH, URLS_TO_INGEST)

    if not documents:
        print("No documents or URLs were successfully loaded. Exiting.")
        return

    # ドキュメントをチャンクに分割
    split_docs = split_documents(documents)

    if not split_docs:
         print("No document chunks were created after splitting. Exiting.")
         return

    # エンベディング関数を作成
    embeddings = embed_documents()

    # ベクトルストアを作成または更新
    create_or_update_vector_store(split_docs, embeddings, CHROMA_DB_PATH)

    print("Ingestion process finished.")

if __name__ == "__main__":
    main()