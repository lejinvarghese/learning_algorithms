import click
import pandas as pd
from sentence_transformers import SentenceTransformer

prompts = {"query": "search_query: ", "document": "search_document: "}


def main():

    models = [
        "nomic-ai/nomic-embed-text-v1.5",
        "lv12/esci-nomic-embed-text-v1_5_1",
        "models/nomic-embed-text-train-esci/checkpoint-20000",
    ]
    sentences = [
        "search_query: shoes",
        "search_query: nike",
        "search_query: adidas",
        "search_query: nike shoes",
        "search_query: adidas shoes",
        "search_query: guitar",
        "search_document: Nike Air Max Dn, Nike, White",
        "search_document: Nike Sportswear Tech Fleece Windrunner, Nike, Black",
        "search_document: Gazelle Indoor Shoes, Adidas, Green",
        "search_document: Adicolor Crew Sweatshirt Set, Adidas, Green",
        "search_document: Electric Guitar, Green",
        "search_document: Guitar strings, Green",
    ]
    results = {}
    for m in models:
        model = SentenceTransformer(
            m,
            trust_remote_code=True,
            device="cpu",
        )
        embeddings = model.encode(sentences)
        results[m] = model.similarity(embeddings, embeddings)
        click.secho(f"Model: {m}", fg="green")
        click.secho(results[m], fg="yellow")
    df = pd.concat(
        {
            m: pd.DataFrame(results[m].tolist(), columns=sentences, index=sentences)
            for m in results
        },
        axis=1,
    )
    df = df.reset_index().melt(
        id_vars="index", var_name=["model", "sentence2"], value_name="similarity"
    )

    df.rename(columns={"index": "sentence1"}, inplace=True)
    df.sort_values(by=["model", "similarity"], ascending=False, inplace=True)
    df = df[df["sentence1"] != df["sentence2"]]
    df["similarity"] = round(df["similarity"], 2)
    click.secho(df, fg="magenta")
    df[["model", "sentence1", "sentence2", "similarity"]].to_csv(
        "models/nomic-embed-text-esci/examples.csv", index=False
    )


if __name__ == "__main__":
    main()
