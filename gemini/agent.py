import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.agents import tool, AgentExecutor, create_react_agent, initialize_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import StructuredTool

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") # Web検索用

# 埋め込みモデルのパス
EMBEDDING_MODEL = "models/embedding-001"
# LLMモデルのパス (例: gemini-pro または gemini-1.5-pro-latest)
LLM_MODEL = "gemini-2.0-flash"

def load_vector_store():
    """既存のベクトルデータベースをロードする"""
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Error: Vector database not found at {CHROMA_DB_PATH}.")
        print("Please run ingest.py first to create the database.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    return vectorstore

def create_qa_chain(vectorstore, llm):
    """RAGベースのQAチェーンを作成する"""

    # プロンプトテンプレートの定義
    # context にはベクトルDBから取得した関連ドキュメントの内容が入る
    # question にはユーザーの質問が入る
    # 日本語で、コンプライアンスの専門家のように振る舞うよう指示
    template = """あなたは優秀なAIアシスタントであり、特に社内コンプライアンスを含む様々な関する質問に対応します。
以下の「提供されたコンテキスト」を基に、ユーザーの質問に正確かつ丁寧に回答してください。
もし、提供されたコンテキストだけでは質問に答えられない場合は、「提供されたコンテキストには回答に必要な情報がありませんでした。」と正直に答えるか、またはウェブ検索の結果（利用可能な場合）も参照して回答を補完してください。
回答は専門的でありながらも分かりやすく、社内規定や法規に基づいた情報を提供してください。
憶測で回答したり、存在しない情報を作り出したりしないでください。

提供されたコンテキスト:
{context}

ユーザーの質問:
{question}

回答:"""

    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    # RetrievalQAチェーンを作成
    # retriever: ベクトルDBから関連ドキュメントを検索するオブジェクト
    # chain_type: "stuff" は取得したドキュメントを全てプロンプトに詰め込む
    # llm: 使用する言語モデル
    # prompt: 使用するプロンプトテンプレート
    # return_source_documents=True: 回答の根拠となったドキュメントを返す（オプション）
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # 質問に似たドキュメントをk個取得
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True # どの資料を参照したか確認したい場合にTrueにする
    )
    return qa_chain

# Web検索ツールを定義 (LangChainのツール形式)
@tool
def tavily_search(query: str) -> str:
    """指定されたクエリでWeb検索を実行し、結果を返します。
    特に社内資料に情報がない場合や、最新の情報が必要な場合に有用です。
    例: '日本の最新の個人情報保護法改正について' """
    if not TAVILY_API_KEY:
        return "Tavily APIキーが設定されていません。Web検索は利用できません。"
    search = TavilySearchResults(max_results=5) # 上位5件の結果を取得
    results = search.run(query)
    # 結果を整形して返す
    formatted_results = []
    for i, result in enumerate(results):
        formatted_results.append(f"Result {i+1}: Title: {result['title']}, URL: {result['url']}, Snippet: {result['content']}")
    return "\n".join(formatted_results)


# def create_agent(llm, qa_chain):
#     """RAGとWeb検索ツールを組み合わせたエージェントを作成する"""

#     # エージェントが利用できるツールリスト
#     tools = [
#         # RAG (内部資料検索) をツールとして定義
#         tool(func=qa_chain.invoke,
#              name="internal_document_search",
#              description="社内コンプライアンス資料を検索し、質問への回答を見つけます。主に社内規定、マニュアル、ガイドラインに関する質問に使用します。質問の前にこのツールを最初に試してください。例: '当社の情報セキュリティポリシーについて教えてください'"),
#         tavily_search # 上記で定義したWeb検索ツール
#     ]

#     # エージェントのプロンプト
#     # tools_prompt には利用可能なツールの情報が自動的に追加される
#     agent_prompt = PromptTemplate.from_template("""あなたは社内コンプライアンス専門家AIです。
# ユーザーからのコンプライアンスに関する質問に対し、最も適切と思われるツール（内部資料検索またはWeb検索）を選択して回答を生成してください。
# 最終的な回答は、分かりやすく、根拠に基づいた情報を提供してください。
# ユーザーからの質問: {input}
# 利用可能なツール:
# {tools}
# {agent_scratchpad}
# 回答までの思考プロセスと、使用するツール、その引数を思考の後に示してください。
# """)


#     # エージェントを作成
#     # create_react_agent は ReAct (Reasoning and Acting) フレームワークに基づいたエージェントを作成
#     # thinking process (思考) -> select tool (ツール選択) -> run tool (ツール実行) -> observe result (結果観察) -> thinking process (思考) ... という流れで動作
#     agent = create_react_agent(llm, tools, agent_prompt)

#     # エージェントの実行器
#     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

#     return agent_executor

def create_agent(llm, qa_chain):
    def qa_tool(query: str) -> str:
        """Run a QA query on the vector store."""
        return qa_chain.invoke({"query": query})["result"]

    tools = [
        StructuredTool.from_function(
            func=qa_tool,
            name="QA_Tool",
            description="様々な資料を検索し、質問への回答を見つけます。WEB検索も積極的に活用します。例: '当社の情報セキュリティポリシーについて教えてください'"
        ),
        tavily_search
    ]
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="zero-shot-react-description",
        verbose=True
    )
    return agent_executor

def main():
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        print("Please set up your .env file correctly.")
        return

    print("Loading vector database...")
    vectorstore = load_vector_store()
    if vectorstore is None:
        return

    print("Initializing Google Gemini model...")
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0) # temperature=0 でより決定的な回答に

    print("Creating QA chain...")
    # RAG単体で使う場合のQAチェーンも作成しておくとデバッグなどに便利
    qa_chain_direct = create_qa_chain(vectorstore, llm)

    print("Creating agent with tools...")
    # エージェントを作成 (RAGとWeb検索を使い分ける)
    agent_executor = create_agent(llm, qa_chain_direct)


    print("\n--- コンプライアンス質問応答エージェント ---")
    print("社内コンプライアンスに関する質問を入力してください。")
    print("終了するには '終了' または 'exit' と入力してください。")
    print("-----------------------------------------------")

    while True:
        question = input("\n質問: ")
        if question.lower() in ["終了", "exit"]:
            print("エージェントを終了します。")
            break

        if not question.strip():
            print("質問を入力してください。")
            continue

        try:
            print("回答を生成中です...")
            # エージェントを実行
            # agent_executor.invoke({"input": question}) の代わりに stream も利用可能
            response = agent_executor.invoke({"input": question})

            print("\n--- 回答 ---")
            print(response.get("output", "回答を生成できませんでした。"))

            # if response.get("source_documents"):
            #     print("\n--- 参照ドキュメント ---")
            #     for i, doc in enumerate(response["source_documents"]):
            #         print(f"ドキュメント {i+1}:")
            #         print(f"  Source: {doc.metadata.get('source', 'N/A')}")
            #         print(f"  Content: {doc.page_content[:200]}...") # 内容の一部を表示
            #     print("---------------------")

        except Exception as e:
            print(f"エラーが発生しました: {e}")
            print("もう一度お試しいただくか、システム管理者にお問い合わせください。")


if __name__ == "__main__":
    main()