from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

response = client.embeddings.create(
    model="text-embedding-nomic-embed-text-1.5",  # curlで確認したID
    input="こんにちは、これは埋め込みのテストです。"
)

vec = response.data[0].embedding
print("✅ 次元数:", len(vec))
print("✅ 先頭5値:", vec[:5])
