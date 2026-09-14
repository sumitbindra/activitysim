"""`asim` command line entry point."""

import click


@click.group()
def main():
    """Run and inspect ActivitySim prototype_mtc runs."""


if __name__ == "__main__":
    main()
