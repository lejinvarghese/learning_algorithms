from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "models/nomic-embed-text-esci/checkpoint-7000", trust_remote_code=True
)
model.push_to_hub(
    repo_id="lv12/esci-nomic-embed-text-v1.5", private=False, exist_ok=True
)