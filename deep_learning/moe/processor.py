import click
from adapters.core import HomeDepotDataset


def main():
    ds = HomeDepotDataset(sample_size=10)
    samples = ds.load()
    click.secho(f"Dataset loaded with {len(samples)} samples.", fg="green")
    click.secho(f"First sample: {samples[0]}", fg="blue")


if __name__ == "__main__":
    main()
