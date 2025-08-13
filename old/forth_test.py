# file: forth_test.py
# TXT(複数) -> split -> (LM Studio embeddings) -> Chroma -> Answer
# 使い方:
#   python forth_test.py file1.txt file2.txt ... --query "要約して" \
#          --persist_dir ./.chroma-txt-demo
#
# メモ:
# - LM StudioのOpenAI互換APIを使用します。LM Studioでサーバ起動後に実行してください。
# - 埋め込みはバッチ投入に対応（OpenAI互換: embeddings.create(input=list[str])）。
# - 参照表示はファイル名とチャンク番号を出します。

import argparse
import os
from typing import List
from pathlib import Path

from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========= 0) 引数処理 =========
parser = argparse.ArgumentParser(
    description="TXT(複数) → 分割 → 埋め込み → Chroma → 質問応答"
)
parser.add_argument(
    "files",
    nargs="+",
    help="入力テキストファイルのパス（複数可）"
)
parser.add_argument(
    "--query",
    default="これらのテキストの要点を日本語で箇条書きでまとめてください。",
    help="質問内容（指定しない場合は要点要約）"
)
parser.add_argument(
    "--encoding",
    default="utf-8",
    help="テキストファイルの文字エンコーディング（既定: utf-8）"
)
parser.add_argument(
    "--persist_dir",
    default=None,
    help="Chromaの永続化ディレクトリ（例: ./.chroma-txt-demo）。未指定なら非永続"
)
parser.add_argument("--chunk_size", type=int, default=700)
parser.add_argument("--chunk_overlap", type=int, default=80)
parser.add_argument("--top_k", type=int, default=4)

# LM Studio接続設定
parser.add_argument("--base_url", default="http://localhost:1234/v1")
parser.add_argument("--api_key", default="not-needed")
parser.add_argument(
    "--embedding_model_id",
    default="text-embedding-nomic-embed-text-1.5",
    help="LM Studioの埋め込みモデルID (/v1/modelsのid)"
)
parser.add_argument(
    "--chat_model_id",
    default="openai/gpt-oss-20b",
    help="LM StudioのチャットモデルID (/v1/modelsのid)"
)

args = parser.parse_args()

# 入力ファイル存在チェック
file_paths: List[Path] = [Path(p) for p in args.files]
missing = [str(p) for p in file_paths if not p.exists()]
if missing:
    raise FileNotFoundError(f"見つからないファイルがあります: {missing}")

USER_QUERY = args.query
PERSIST_DIR = args.persist_dir
CHUNK_SIZE = args.chunk_size
CHUNK_OVERLAP = args.chunk_overlap
TOP_K = args.top_k
BASE_URL = args.base_url
API_KEY = args.api_key
EMBEDDING_MODEL_ID = args.embedding_model_id
CHAT_MODEL_ID = args.chat_model_id
ENCODING = args.encoding

# ========= 1) OpenAI互換クライアント =========
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ========= 2) 埋め込み：LangChain互換ラッパー =========
class LMStudioEmbedding:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # まとめて投入（OpenAI互換APIは配列投入OK）
        resp = client.embeddings.create(model=EMBEDDING_MODEL_ID, input=texts)
        # resp.data は入力順に戻る
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

embedding_fn = LMStudioEmbedding()

# ========= 3) TXT読み込み & 分割 =========
def load_and_split_texts(paths: List[Path]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        with open(p, "r", encoding=ENCODING, errors="ignore") as f:
            content = f.read()
        # 1ファイル＝1 Document（後で分割）
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": p.name,
                    "path": str(p.resolve()),
                },
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # チャンク番号を付与（参照表示で使う）
    per_file_counter = {}
    for d in chunks:
        src = d.metadata.get("source", "unknown")
        per_file_counter[src] = per_file_counter.get(src, 0) + 1
        d.metadata["chunk_id"] = per_file_counter[src]
    return chunks

# ========= 4) ベクトルDB構築（Chroma） =========
def build_vectordb(chunks: List[Document]) -> Chroma:
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name="demo-lms-texts",
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
        chunk_id = d.metadata.get("chunk_id", i)
        # 長すぎるとプロンプトが膨らむので適度に切る
        content = d.page_content.strip().replace("\n", " ")
        if len(content) > 500:
            content = content[:500] + "…"
        ctx += f"[{i}] ({src}#chunk-{chunk_id}) {content}\n"
    instr = (
        "\n■ 指示:\n"
        "- 上の参考文書の内容に基づいて答えてください。\n"
        "- 根拠が弱い場合は『分かりません』と答えてください。\n"
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
    refs = "参照: " + ", ".join(
        f"{d.metadata.get('source','?')}#chunk-{d.metadata.get('chunk_id','?')}" for d in ctx_docs
    )
    return content + "\n\n" + refs


# ========= 6) 実行 =========
if __name__ == "__main__":
    print(">>> Loading & splitting TEXTs ...")
    chunks = load_and_split_texts(file_paths)
    print(f" - chunks: {len(chunks)}")

    print(">>> Building vectordb (this may take a bit for large corpora) ...")
    vectordb = build_vectordb(chunks)
    try:
        count = vectordb._collection.count()  # type: ignore[attr-defined]
    except Exception:
        count = -1
    print(" - collection size:", count)

    query = USER_QUERY
    print("\n--- 質問:", query)
    print("\n--- 回答 ---")
    print(answer(vectordb, query))
