from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "models/nomic-embed-text-esci/checkpoint-50000",
    trust_remote_code=True,
)
model.push_to_hub(
    repo_id="lv12/esci-nomic-embed-text-v1_5_3",
    private=False,
    exist_ok=True,
    replace_model_card=True,
    train_datasets="tasksource/esci",
    commit_message="full set multi loss ESCI triplets",
)
