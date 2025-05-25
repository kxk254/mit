
# pip install google-generativeai python-dotenv
# pip install -q google-generativeai langchain langchain-chroma langchain-google-genai python-dotenv pypdf unstructured
# pip install -q --upgrade typing-extensions  # 依存関係の更新
# pip install -q pillow  # unstructuredが必要とする場合がある
# pip install langchain-community
# Web検索機能も利用する場合は、以下のいずれかを追加インストールします（例: Tavily）。
# https://aistudio.google.com/app/prompts/new_chat
# https://app.tavily.com/home
# pip install -q tavily-python
# pip install pdfminer.six

import google.generativeai as genai
import os
from dotenv import load_dotenv # .env ファイルから環境変数を読み込む

# .env ファイルから環境変数を読み込む   .env  GOOGLE_API_KEY='YOUR_API_KEY'
load_dotenv()

# 1. Gemini APIキーの設定
# 環境変数からAPIキーを読み込みます。
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("エラー: GOOGLE_API_KEY 環境変数が設定されていません。")
    print(".env ファイルに GOOGLE_API_KEY='YOUR_API_KEY' を記述するか、環境変数に設定してください。")
    exit()

genai.configure(api_key=API_KEY)

# 2. Geminiモデルの初期化
# 使用するモデルを指定します。例: 'gemini-pro'
# 利用可能なモデルはAPIリファレンスを確認してください。
try:
    model = genai.GenerativeModel('gemini-2.0-flash')
    print("Geminiモデルをロードしました。")
except Exception as e:
    print(f"エラー: Geminiモデルのロードに失敗しました。APIキーが正しいか、またはモデル名を確認してください。詳細: {e}")
    exit()

# 3. コンプライアンス質問応答のためのプロンプトベース
# AIに与える役割と指示を定義します。この部分はAIの応答の質に大きく影響します。
SYSTEM_INSTRUCTIONS = """
あなたは企業のコンプライアンスに関する社員からの質問に答えるAIアシスタントです。
以下の質問に対して、一般的な企業のコンプライアンスの観点から、分かりやすく丁寧に回答してください。
回答は具体的かつ実践的な内容を心がけてください。

**重要な注意点:**
- 提供する情報は一般的な情報であり、特定の状況における法的助言や社内規定の公式な解釈ではありません。
- 個別の事案に関する正確な判断や、社内規定の詳細については、必ず所属部署の責任者やコンプライアンス担当部署にご確認ください。
- 私はAIであり、公式な決定や法的な拘束力を持つ回答はできません。
- 不明な点や曖昧な質問については、追加情報の提供をお願いするか、コンプライアンス担当部署への相談を促してください。

ユーザーからの質問:
"""

# 4. ユーザーからの質問を受け付け、応答を生成する対話ループ
print("\n================================================")
print(" コンプライアンス質問応答AIアシスタント")
print("================================================")
print("コンプライアンスに関する質問を入力してください。")
print("終了するには '終了' または 'exit' と入力してください。")
print("-" * 50)

# 会話履歴を持たせることで、前後の文脈を理解した応答が可能になります。
chat_session = model.start_chat(history=[])

while True:
    user_question = input("あなた: ")

    # 終了コマンドのチェック
    if user_question.lower() in ['終了', 'exit', 'quit']:
        print("AIアシスタントを終了します。ご利用ありがとうございました。")
        break

    # 空の入力はスキップ
    if not user_question.strip():
        print("質問を入力してください。")
        continue

    try:
        # プロンプトとユーザー質問を組み合わせてGeminiに送信
        # start_chatを使っているため、send_messageの引数はユーザーの入力のみでOKですが、
        # ここではより役割を明確にするためにSYSTEM_INSTRUCTIONSを含めたプロンプトを送信します。
        # chat_sessionは履歴を管理するため、同じセッション内で連続して質問すると前回の会話も考慮されます。
        
        # send_messageに直接プロンプトとユーザー質問を渡す方法:
        # response = chat_session.send_message(f"{SYSTEM_INSTRUCTIONS}\n{user_question}")

        # シンプルにユーザー質問だけを渡す方法（start_chatで設定した履歴とシステム指示が使われます）:
        # ここでは、ユーザーの質問に回答するという主目的のために、簡潔にユーザー入力のみを渡します。
        # より複雑な指示が必要な場合は、start_chatのsystem_instructionパラメータ（現在はベータ機能）や、
        # 各メッセージに指示を含めることを検討します。
        response = chat_session.send_message(user_question)


        # 応答を表示
        # 応答が複数の部分に分かれる場合があるため、partsを結合して表示します。
        ai_response = "".join(part.text for part in response.parts)
        print("\nAIアシスタント:")
        print(ai_response)
        print("-" * 50)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("もう一度試すか、システム管理者に連絡してください。")
        print("-" * 50)