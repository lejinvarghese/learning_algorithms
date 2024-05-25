from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import no_grad
from sentence_transformers import CrossEncoder


model_name = "lv12/esci-ms-marco-MiniLM-L-12-v2"

queries = [
    "adidas shoes",
    "adidas shoes",
    "girls sandals",
    "backpacks",
    "shoes", 
    "mustard sleeveless gown"
]
documents =  [
    '{"title": "Nike Air Max", "description": "The best shoes you can get, with air cushion", "brand": "Nike", "color": "black"}',
    '{"title": "Adidas Ultraboost", "description": "The shoes that represent the world", "brand": "Adidas", "color": "white"}',
    '{"title": "Womens sandals", "description": "Sandals:  wide width 9", "brand": "Chacos", "color": "blue"}',
    '{"title": "Girls surf backpack", "description": "The best backpack in town", "brand": "Roxy", "color": "pink"}',
    '{"title": "Fresh watermelon", "description": "The best fruit in town, all you can eat", "brand": "Fruitsellers Inc.", "color": "green"}',
    '{"title": "Floral yellow dress with frills and lace", "description": "Brighten up your summers with a gorgeous dress", "brand": "Dressmakers Inc.", "color": "bright yellow"}'
    ]

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(
    queries,
    documents,
    padding=True,
    truncation=True,
    return_tensors="pt",
)

model.eval()
with no_grad():
    scores = model(**inputs).logits.cpu().detach().numpy()
    print(scores)


model = CrossEncoder(model_name, max_length=512)
scores = model.predict([(q, d) for q, d in zip(queries, documents)])
print(scores)