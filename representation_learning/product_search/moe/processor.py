import click
from adapters import DatasetAggregator


@click.command()
@click.option("--sample_size", default=None, type=int, help="Number of samples to generate.")
@click.option("--subset_name", default=None, type=str, help="Subset name to push to the hub.")
def main(sample_size, subset_name):
    ds = DatasetAggregator(sample_size=sample_size)

    if subset_name is None:
        ds.generate_pairs()
        ds.generate_triplets()
        ds.push_to_hub()
    elif subset_name == "pairs":
        ds.generate_pairs()
        ds.push_to_hub(subset_name=subset_name)
    elif subset_name == "triplets":
        ds.generate_triplets()
        ds.push_to_hub(subset_name=subset_name)


if __name__ == "__main__":
    main()
