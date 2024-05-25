from typing import List, Dict
from tqdm import tqdm

from pandas import DataFrame
from sentence_transformers import  InputExample

def get_document(row: Dict) -> str:
    return str({"title": row.get("product_title"), 
                        "description": row.get("product_description"), 
                        "brand": row.get("product_brand"), 
                        "color": row.get("product_color")})

def get_pairs(data: DataFrame, split: str="training") -> List[InputExample]:
    examples = []
    for _, row in tqdm(
        data.iterrows(),
        total=len(data),
        desc=f"Generating sentence pairs for {split} set: ",
        colour="green",
    ):
        examples.append(
            InputExample(
                texts=[row.get("query"), get_document(row)],
                label=float(row.get("gain")),
            )
        )
    return examples

def get_triplets(data: DataFrame, split: str="training") -> Dict[int, Dict]:
    triplets, anchors = {}, {}
    for _, row in tqdm(
        data.iterrows(),
        total=len(data),
        desc=f"Generating triplets for {split} set: ",
        colour="green",
    ):
        qid = anchors.get(row.get("query"), len(anchors))
        if qid == len(anchors):
            anchors[row.get("query")] = qid
        if qid not in triplets:
            triplets[qid] = {
                "query": row.get("query"),
                "positive": set(),
                "negative": set(),
            }
        if row.get("gain") > 0.0:
            triplets[qid]["positive"].add(get_document(row))
        elif row.get("gain") == 0.0:
            triplets[qid]["negative"].add(get_document(row))
    return triplets