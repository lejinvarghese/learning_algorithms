import click
from adapters.core import HomeDepotDataset


def main():
    ds = HomeDepotDataset(sample_size=None)
    samples = ds.generate_pairs()
    samples = ds.generate_triplets()


if __name__ == "__main__":
    main()
