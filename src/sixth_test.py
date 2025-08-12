# file: fifth_test.py
# TXT(複数) -> split -> (LM Studio embeddings) -> Chroma -> MMR検索 -> Answer
# 使い方:
#   インデックス作成+質問:
#     python fifth_test.py a.txt b.txt --persist_dir ./.chroma-txt-demo --query "要約して"
#   既存DBから質問のみ:
#     python fifth_test.py --ask_only --persist_dir ./.chroma-txt-demo --query "定義は？"

import argparse
from pathlib import Path
from typing import List

from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========= 0) 引数処理 =========
parser = argparse.ArgumentParser(
    description="TXT(複数) → 分割 → 埋め込み → Chroma → MMR検索 → 質問応答"
)
parser.add_argument("files", nargs="*", help="入力テキストファイルのパス（複数可）")
parser.add_argument("--query", default="これらのテキストの要点を日本語で箇条書きでまとめてください。")
parser.add_argument("--encoding", default="utf-8")
parser.add_argument("--persist_dir", default=None, help="Chroma永続ディレクトリ（未指定は非永続）")
parser.add_argument("--chunk_size", type=int, default=900)
parser.add_argument("--chunk_overlap", type=int, default=150)
parser.add_argument("--top_k", type=int, default=6, help="最終的に渡す文書数（MMRのk）")
parser.add_argument("--ask_only", action="store_true", help="インデックス作成をスキップ。既存Chromaから検索のみ")
parser.add_argument("--collection_name", default="demo-lms-texts")

# LM Studio接続設定
parser.add_argument("--base_url", default="http://localhost:1234/v1")
parser.add_argument("--api_key", default="not-needed")
parser.add_argument("--embedding_model_id", default="text-embedding-nomic-embed-text-1.5")
parser.add_argument("--chat_model_id", default="openai/gpt-oss-20b")
#parser.add_argument("--chat_model_id", default="llama-3.2-1b-instruct")

# MMR設定
parser.add_argument("--fetch_k", type=int, default=50, help="MMRで内部的にまず集める候補数")
parser.add_argument("--mmr_lambda", type=float, default=0.3, help="MMRの多様性重み(0〜1、低いほど多様性重視)")

args = parser.parse_args()

USER_QUERY = args.query

# ========= 1) OpenAI互換クライアント =========
client = OpenAI(base_url=args.base_url, api_key=args.api_key)

# ========= 2) 埋め込み：LangChain互換ラッパー =========
class LMStudioEmbedding:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = client.embeddings.create(model=args.embedding_model_id, input=texts)
        return [d.embedding for d in resp.data]
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

embedding_fn = LMStudioEmbedding()

# ========= 3) TXT読み込み & 分割 =========
def load_and_split_texts(paths: List[Path]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        with open(p, "r", encoding=args.encoding, errors="ignore") as f:
            content = f.read()
        docs.append(Document(page_content=content, metadata={"source": p.name, "path": str(p.resolve())}))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
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
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
    )

# ========= 5) 検索 & 回答 =========
SYSTEM_PROMPT = (
    "あなたは日本語で簡潔かつ正確に回答するアシスタントです。"
    "参照文書に基づいて回答し、推測は避け、箇条書きを活用してください。"
    "コンテキストに無い場合は『分かりません』と答えてください。"
)

def build_user_prompt(query: str, ctx_docs: List[Document]) -> str:
    header = "■ 質問:\n" + query.strip() + "\n\n"
    ctx = "■ 参考文書（抜粋）:\n"
    for i, d in enumerate(ctx_docs, 1):
        src = d.metadata.get("source", f"doc-{i}")
        chunk_id = d.metadata.get("chunk_id", i)
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

def retrieve_mmr(vectordb: Chroma, query: str, k: int, fetch_k: int, mmr_lambda: float) -> List[Document]:
    """MMR（最大限界関連性）で多様性を確保しつつk件取得"""
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": mmr_lambda},
    )
    return retriever.get_relevant_documents(query)

def answer(vectordb: Chroma, query: str) -> str:
    # MMRで直接 top_k 件を取得
    ctx_docs = retrieve_mmr(vectordb, query, k=args.top_k, fetch_k=args.fetch_k, mmr_lambda=args.mmr_lambda)
    user_prompt = build_user_prompt(query, ctx_docs)
    resp = client.chat.completions.create(
        model=args.chat_model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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
    file_paths: List[Path] = [Path(p) for p in args.files]
    if not args.ask_only and not file_paths:
        raise SystemExit("ファイルが指定されていません。--ask_only を使う場合を除き、少なくとも1つのテキストファイルを指定してください。")
    missing = [str(p) for p in file_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"見つからないファイルがあります: {missing}")

    if args.ask_only:
        if not args.persist_dir:
            raise SystemExit("--ask_only を使う場合は --persist_dir を指定してください。")
        print(">>> Loading existing vectordb from persist directory ...")
        vectordb = Chroma(
            collection_name=args.collection_name,
            embedding_function=embedding_fn,
            persist_directory=args.persist_dir,
        )
        try:
            count = vectordb._collection.count()  # type: ignore[attr-defined]
        except Exception:
            count = -1
        print(" - loaded collection size:", count)
    else:
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

    print("\n--- 質問:", USER_QUERY)
    print("\n--- 回答 ---")
    print(answer(vectordb, USER_QUERY))
