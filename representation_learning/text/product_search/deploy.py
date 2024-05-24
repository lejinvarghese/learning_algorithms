
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from constants import DIRECTORY

model_path = f"{DIRECTORY}/models/cross-encoder/ms-marco-MiniLM-L-12-v2"
dest_model_name = "esci-ms-marco-MiniLM-L-12-v2"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.push_to_hub(dest_model_name)
tokenizer.push_to_hub(dest_model_name)
    
    