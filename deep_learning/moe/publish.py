from huggingface_hub import upload_folder, create_repo
from transformers import AutoModel, AutoConfig
import warnings
warnings.filterwarnings("ignore", message=".*untrained layers.*")  


## Upload model to the Hub
repo_id = "lv12/bert_base_uncased_embedding_moe"

try:
    create_repo(repo_id=repo_id, repo_type="model")
except:
    pass

upload_folder(
    folder_path="models/bert_base_uncased_embedding_moe/20250312232018",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Uploading model.pt",
)


## Load model from the Hub
config = AutoConfig.from_pretrained("lv12/bert_base_uncased_embedding_moe", trust_remote_code=True)
model = AutoModel.from_pretrained("lv12/bert_base_uncased_embedding_moe", config=config, trust_remote_code=True, ignore_mismatched_sizes=True)