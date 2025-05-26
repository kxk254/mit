使用方法:

.env ファイル、ingest.py、agent.py を同じフォルダに置きます。
.env を編集し、必要なAPIキーとパスを設定します。
ingest.py を実行してベクトルデータベースを構築します。
python ingest.py
成功すると、.envで指定したパス（デフォルトは ./chroma_db）にベクトルデータベースが作成されます。
agent.py を実行してエージェントを起動します。
python agent.py
エージェントが起動し、質問の入力待ちになります。コンプライアンスに関する質問を入力してエンターキーを押してください。
Geminiモデルが内部資料とWeb検索（設定されていれば）を基に回答を生成します。
終了するには「終了」または「exit」と入力してください。
VS Codeでの実行

VS Codeで ingest.py または agent.py ファイルを開き、エディタ右上にある緑色の三角ボタン（実行ボタン）をクリックすると、ファイルが実行されます。
デバッグを行う場合は、ブレークポイントを設定し、虫のアイコンのデバッグボタンをクリックします。
カスタマイズと拡張

対応ドキュメント形式の追加: ingest.py の LOADER_MAPPING に新しいファイル形式と対応するLangChainローダーを追加します。必要に応じて pip install で追加ライブラリをインストールしてください（例: unstructured は様々な形式に対応できますが、依存関係が多いです）。
RAGパラメータの調整: create_qa_chain 内の search_kwargs={"k": 3} の k の値を変更することで、検索するドキュメントチャンクの数を調整できます。
プロンプトの調整: agent.py 内の PROMPT テンプレートや agent_prompt テンプレートを変更することで、AIの応答スタイルや思考プロセスを調整できます。
Web検索ツールの変更: tavily_search 関数の代わりに、Google Custom Search API や他の検索APIを利用するツールを実装して追加できます。
UIの追加: コマンドラインではなく、StreamlitやGradioなどを使って簡単なWeb UIを作成することも可能です。