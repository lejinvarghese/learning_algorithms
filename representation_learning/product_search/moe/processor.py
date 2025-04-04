import click
from adapters import DatasetAggregator, HomeDepotDataset


@click.command()
@click.option("--sample_size", default=None, type=int, help="Number of samples to generate.")
def main(sample_size):
    ds = DatasetAggregator(sample_size=sample_size)
    samples = ds.generate_pairs()
    samples = ds.generate_triplets()
    ds.push_to_hub()


if __name__ == "__main__":
    main()
