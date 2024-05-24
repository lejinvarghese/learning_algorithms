from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "lv12/esci-ms-marco-MiniLM-L-12-v2"

queries = [
    "adidas shoes",
    "adidas sambas",
    "girls sandals",
    "backpacks",
    "shoes", 
    "mustard blouse"
]
documents =  [
        "Nike Air Max, with air cushion",
        "Adidas Ultraboost, the best boost you can get",
        "Women's sandals wide width 9",
        "Girl's surf backpack",
        "Fresh watermelon, all you can eat",
        "Floral yellow dress with frills and lace"
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
scores = model(**inputs).logits.cpu().detach().numpy()
print(scores)