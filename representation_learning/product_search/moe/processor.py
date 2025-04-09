import click
from tqdm import trange
from adapters import DatasetAggregator

@click.command()
@click.option("--sample_size", default=None, type=int, help="Number of samples to generate.")
@click.option("--subset_name", default=None, type=str, help="Subset name to push to the hub.")
@click.option("--chunk_size", default=None, type=int, help="Maximum number of queries to process in each chunk.")
@click.option("--chunk_suffix", default=None, type=str, help="Suffix to add to the chunk name.")
def main(sample_size, subset_name, chunk_size, chunk_suffix):
    ds = DatasetAggregator(sample_size=sample_size, chunk_size=chunk_size)
    max_chunks = ds.identify_max_chunks()

    if subset_name is None:
        subsets = ["pairs", "triplets"]
    else:
        subsets = [subset_name]

    for subset in subsets:
        if subset == "pairs":
            ds.generate_pairs()
            ds.push_to_hub(subset_name=subset, chunk_suffix=chunk_suffix)
        elif subset == "triplets":
            for i in trange(max_chunks, desc="Processing chunks", colour="blue"):
                click.secho(f"Generating triplets for chunk {i+1}/{max_chunks}", fg="blue")
                ds.generate_triplets(chunk_index=i)
                ds.push_to_hub(subset_name=subset, chunk_index=i, chunk_suffix=chunk_suffix)


if __name__ == "__main__":
    main()
