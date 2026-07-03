import click
from .data.split import split
from .data.convert import bids_to_nnunet


@click.group(no_args_is_help=True)
def cli():
    pass

cli.add_command(split)
cli.add_command(bids_to_nnunet)