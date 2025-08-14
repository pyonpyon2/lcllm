import sys
import time
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass

from openai import OpenAI
from langchain_chroma import Chroma  # ← 新パッケージ
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import argparse

# =====================
# ロギング設定
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "あなたは日本語で簡潔かつ正確に回答するアシスタントです。"
    "参照文書に基づいて回答し、推測は避け、箇条書きを活用してください。"
    "コンテキストに無い場合は『分かりません』と答えてください。"
)

@dataclass
class Config:
    files: List[Path]
    query: str
    encoding: str
    persist_dir: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    ask_only: bool
    collection_name: str
    base_url: str
    api_key: str
    embedding_model_id: str
    chat_model_id: str
    fetch_k: int
    mmr_lambda: float

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*")
    p.add_argument("--query", default="これらのテキストの要点を日本語で箇条書きでまとめてください。")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--persist_dir", default=None)
    p.add_argument("--chunk_size", type=int, default=900)
    p.add_argument("--chunk_overlap", type=int, default=150)
    p.add_argument("--top_k", type=int, default=6)
    p.add_argument("--ask_only", action="store_true")
    p.add_argument("--collection_name", default="demo-lms-texts")
    p.add_argument("--base_url", default="http://localhost:1234/v1")
    p.add_argument("--api_key", default="not-needed")
    p.add_argument("--embedding_model_id", default="text-embedding-nomic-embed-text-1.5")
    p.add_argument("--chat_model_id", default="openai/gpt-oss-20b")
    p.add_argument("--fetch_k", type=int, default=50)
    p.add_argument("--mmr_lambda", type=float, default=0.3)
    args = p.parse_args()
    return Config(
        files=[Path(f) for f in args.files],
        query=args.query,
        encoding=args.encoding,
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        ask_only=args.ask_only,
        collection_name=args.collection_name,
        base_url=args.base_url,
        api_key=args.api_key,
        embedding_model_id=args.embedding_model_id,
        chat_model_id=args.chat_model_id,
        fetch_k=args.fetch_k,
        mmr_lambda=args.mmr_lambda,
    )

class LMStudioEmbedding:
    def __init__(self, client: OpenAI, model_id: str):
        self.client = client
        self.model_id = model_id
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model_id, input=texts)
        return [d.embedding for d in resp.data]
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def load_and_split_texts(paths: List[Path], encoding: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    start = time.perf_counter()
    docs = []
    for p in paths:
        with open(p, "r", encoding=encoding, errors="ignore") as f:
            content = f.read()
        docs.append(Document(page_content=content, metadata={"source": p.name, "path": str(p.resolve())}))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for idx, d in enumerate(chunks, 1):
        d.metadata["chunk_id"] = idx
    logger.info(f"Loaded and split into {len(chunks)} chunks in {time.perf_counter() - start:.2f}s")
    return chunks

class QAService:
    def __init__(self, cfg: Config):
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self.embedding_fn = LMStudioEmbedding(self.client, cfg.embedding_model_id)
        self.cfg = cfg

    def build_vectordb(self, chunks: List[Document]) -> Chroma:
        start = time.perf_counter()
        db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_fn,
            collection_name=self.cfg.collection_name,
            persist_directory=self.cfg.persist_dir,
        )
        logger.info(f"Built vectordb in {time.perf_counter() - start:.2f}s")
        return db

    def retrieve_mmr(self, vectordb: Chroma, query: str) -> List[Document]:
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.cfg.top_k, "fetch_k": self.cfg.fetch_k, "lambda_mult": self.cfg.mmr_lambda},
        )
        return retriever.invoke(query)  # ← 新API

    def build_user_prompt(self, query: str, ctx_docs: List[Document]) -> str:
        header = f"■ 質問:\n{query.strip()}\n\n■ 参考文書（抜粋）:\n"
        for i, d in enumerate(ctx_docs, 1):
            src = d.metadata.get("source", f"doc-{i}")
            chunk_id = d.metadata.get("chunk_id", i)
            content = d.page_content.strip().replace("\n", " ")
            if len(content) > 500:
                content = content[:500] + "…"
            header += f"[{i}] ({src}#chunk-{chunk_id}) {content}\n"
        header += "\n■ 指示:\n- 上の参考文書に基づいて答えてください。\n- 必要なら箇条書きを使ってください。\n"
        return header

    def answer(self, vectordb: Chroma, query: str) -> str:
        start = time.perf_counter()
        ctx_docs = self.retrieve_mmr(vectordb, query)
        user_prompt = self.build_user_prompt(query, ctx_docs)
        resp = self.client.chat.completions.create(
            model=self.cfg.chat_model_id,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_prompt}],
            temperature=0.2,
        )
        logger.info(f"Generated answer in {time.perf_counter() - start:.2f}s")
        content = resp.choices[0].message.content.strip()
        refs = "参照: " + ", ".join(f"{d.metadata.get('source','?')}#chunk-{d.metadata.get('chunk_id','?')}" for d in ctx_docs)
        return content + "\n\n" + refs

def main():
    total_start = time.perf_counter()
    cfg = parse_args()
    service = QAService(cfg)

    if cfg.ask_only:
        if not cfg.persist_dir:
            sys.exit("--ask_only を使う場合は --persist_dir を指定してください。")
        logger.info("Loading existing vectordb...")
        vectordb = Chroma(
            collection_name=cfg.collection_name,
            embedding_function=service.embedding_fn,
            persist_directory=cfg.persist_dir,
        )
    else:
        chunks = load_and_split_texts(cfg.files, cfg.encoding, cfg.chunk_size, cfg.chunk_overlap)
        vectordb = service.build_vectordb(chunks)

    logger.info(f"Question: {cfg.query}")
    answer_text = service.answer(vectordb, cfg.query)
    print("\n--- 回答 ---\n" + answer_text)
    logger.info(f"Total execution time: {time.perf_counter() - total_start:.2f}s")

if __name__ == "__main__":
    main()
