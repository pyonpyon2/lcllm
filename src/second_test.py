from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# LM Studio APIクライアント
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
model_name = "text-embedding-nomic-embed-text-1.5"

# 埋め込みラッパークラス
class LMStudioEmbedding:
    def embed_documents(self, texts):
        return [client.embeddings.create(model=model_name, input=t).data[0].embedding for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

embedding_fn = LMStudioEmbedding()

# データ
docs = [
    Document(page_content="横浜は港町で、みなとみらいには観覧車がある。"),
    Document(page_content="AWSのIAMロールは、人が一時的に引き受ける（Assume）仕組み。"),
    Document(page_content="CIDRは可変長サブネットマスクでアドレスを表す方式。"),
]

# Chromaに投入
vectordb = Chroma.from_documents(docs, embedding=embedding_fn, collection_name="demo-lms")

print("✅ Vector store built.")

# 類似検索
hits = vectordb.similarity_search("IAMって何？", k=2)
for i, h in enumerate(hits, 1):
    print(f"[{i}] {h.page_content}")
