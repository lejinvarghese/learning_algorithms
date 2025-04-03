import click
from adapters import HomeDepotDataset, AmazonDataset


@click.command()
@click.option("--sample_size", default=None, type=int, help="Number of samples to generate.")
def main(sample_size):
    ds = AmazonDataset(sample_size=sample_size)
    samples = ds.generate_pairs()
    samples = ds.generate_triplets()


if __name__ == "__main__":
    main()
