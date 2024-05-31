from datasets import load_dataset, Dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction, TripletEvaluator
from sentence_transformers import SentenceTransformer
from train import build_dataset
from multiprocessing import cpu_count

RANDOM_STATE = 42
DATASET_NAME = "tasksource/esci"
N_PROCS = cpu_count() - 2
cols = [
    "product_title",
    "product_description",
    "product_brand",
    "product_color",
    "esci_label",
    "query",
]
esci = load_dataset(DATASET_NAME, num_proc=N_PROCS, columns=cols, split="test")
esci_valid_df = esci.to_pandas()
print(esci_valid_df.shape)

prompts = {"query": "search_query: ", "document": "search_document:"}


def main():
    eval_dataset = build_triplets(esci_valid_df, prompts=prompts, sample_frac=0.01)
    print(f"Finished sampling triplets:{eval_dataset.shape}")

    models = [
        "models/nomic-embed-text-esci/checkpoint-5300",
        "models/nomic-embed-text-esci/checkpoint-7000",
        "models/nomic-embed-text-esci/checkpoint-11000",
        "nomic-ai/nomic-embed-text-v1.5",
    ]
    results = {}
    for m in models:
        model = SentenceTransformer(m, trust_remote_code=True)
        # Initialize the evaluator
        # dev_evaluator = EmbeddingSimilarityEvaluator(
        #     sentences1=eval_dataset["anchor"],
        #     sentences2=eval_dataset["positive"],
        #     scores=eval_dataset["score"],
        #     main_similarity=SimilarityFunction.COSINE,
        #     name="sts-dev",
        # )
        dev_evaluator = TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            main_distance_function=SimilarityFunction.COSINE,
            name="all-nli-dev",
        )
        # Run evaluation manually:
        results[m] = dev_evaluator(model)
    print(results)

    """
    'models/nomic-embed-text-esci/checkpoint-11000': {'all-nli-dev_cosine_accuracy': 0.48403282897039945, 'all-nli-dev_dot_accuracy': 0.2857365125053483, 'all-nli-dev_manhattan_accuracy': 0.47391963903691314, 'all-nli-dev_euclidean_accuracy': 0.4743475047648683, 'all-nli-dev_max_accuracy': 0.48403282897039945
    }, 'nomic-ai/nomic-embed-text-v1.5': {'all-nli-dev_cosine_accuracy': 0.454004434244817, 'all-nli-dev_dot_accuracy': 0.2972499902757789, 'all-nli-dev_manhattan_accuracy': 0.4517873118363219, 'all-nli-dev_euclidean_accuracy': 0.4529153214827492, 'all-nli-dev_max_accuracy': 0.454004434244817
    }
    """


if __name__ == "__main__":
    main()
