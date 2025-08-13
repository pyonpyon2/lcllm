# file: lmstudio_rag_pdf.py
# PDF -> split -> (LM Studio embeddings) -> Chroma -> Answer
import argparse
import os

from typing import List
from dataclasses import dataclass

from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========= 0) 引数処理 =========
parser = argparse.ArgumentParser(
    description="PDF → 分割 → 埋め込み → Chroma → 質問応答"
)
parser.add_argument("pdf_path", help="入力PDFファイルのパス")
parser.add_argument(
    "--query",
    default="このPDFの要点を日本語で箇条書きでまとめてください。",
    help="質問内容（指定しない場合は要点要約）"
)
args = parser.parse_args()

if not os.path.exists(args.pdf_path):
    raise FileNotFoundError(f"PDFが見つかりません: {args.pdf_path}")

PDF_PATH = args.pdf_path
USER_QUERY = args.query

# ========= 0) 設定 =========
BASE_URL = "http://localhost:1234/v1"       # LM Studio Local Server
API_KEY  = "not-needed"                      # LM StudioならダミーでOK
EMBEDDING_MODEL_ID = "text-embedding-nomic-embed-text-1.5"  # /v1/models の id
CHAT_MODEL_ID      = "openai/gpt-oss-20b"                # 同上（例）
# PDF_PATH           = "docs.pdf"              # 読み込むPDFのパス
PERSIST_DIR        = None                    # 永続化したい場合は "./.chroma-pdf-demo" など

# チャンク設定（必要に応じて調整）
CHUNK_SIZE = 700
CHUNK_OVERLAP = 80
TOP_K = 4   # 取得する関連チャンク数


# ========= 1) OpenAI互換クライアント =========
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# ========= 2) 埋め込み：LangChain互換ラッパー =========
class LMStudioEmbedding:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [
            client.embeddings.create(model=EMBEDDING_MODEL_ID, input=t).data[0].embedding
            for t in texts
        ]
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

embedding_fn = LMStudioEmbedding()


# ========= 3) PDF読み込み & 分割 =========
def load_and_split_pdf(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # 各ページが Document（metadata に page が入る）
    # ページ番号を source として埋める（後で出典を見せるため）
    for d in docs:
        page = d.metadata.get("page", None)
        d.metadata = {**d.metadata, "source": f"page-{page}"}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


# ========= 4) ベクトルDB構築（Chroma） =========
def build_vectordb(chunks: List[Document]) -> Chroma:
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name="demo-lms-pdf",
        persist_directory=PERSIST_DIR,
    )
    return vectordb


# ========= 5) 検索 & 回答 =========
SYSTEM_PROMPT = (
    "あなたは日本語で簡潔かつ正確に回答するアシスタントです。"
    "参照文書に基づいて回答し、推測は避け、箇条書きを活用してください。"
)

def build_user_prompt(query: str, ctx_docs: List[Document]) -> str:
    header = "■ 質問:\n" + query.strip() + "\n\n"
    ctx = "■ 参考文書（抜粋）:\n"
    for i, d in enumerate(ctx_docs, 1):
        src = d.metadata.get("source", f"doc-{i}")
        # 長すぎるとプロンプトが増えるので適度に切る
        content = d.page_content.strip().replace("\n", " ")
        if len(content) > 500:
            content = content[:500] + "…"
        ctx += f"[{i}] ({src}) {content}\n"
    instr = (
        "\n■ 指示:\n"
        "- 上の参考文書の内容に基づいて答えてください。\n"
        "- 根拠が弱い場合は「分かりません」と答えてください。\n"
        "- 必要なら箇条書きを使ってください。\n"
    )
    return header + ctx + instr

def retrieve(vectordb: Chroma, query: str, k: int = TOP_K) -> List[Document]:
    return vectordb.similarity_search(query, k=k)

def answer(vectordb: Chroma, query: str) -> str:
    ctx_docs = retrieve(vectordb, query, k=TOP_K)
    user_prompt = build_user_prompt(query, ctx_docs)
    resp = client.chat.completions.create(
        model=CHAT_MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()
    refs = "参照: " + ", ".join(d.metadata.get("source", f"doc-{i+1}") for i, d in enumerate(ctx_docs))
    return content + "\n\n" + refs


# ========= 6) 実行 =========
if __name__ == "__main__":
    print(">>> Loading & splitting PDF ...")
    chunks = load_and_split_pdf(PDF_PATH)
    print(f" - chunks: {len(chunks)}")
    print(chunks)

    print(">>> Building vectordb (this may take a bit for large PDFs) ...")
    vectordb = build_vectordb(chunks)
    print(" - collection size:", vectordb._collection.count())

    # 動作確認用の質問例（適宜変えてください）
#    query = "このPDFの要点を日本語で箇条書きでまとめてください。"
    query = USER_QUERY
    print("\n--- 質問:", query)
    print("\n--- 回答 ---")
    print(answer(vectordb, query))
