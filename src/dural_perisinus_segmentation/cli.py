import click

from .data.convert import bids_to_nnunet, nnunet_outputs_to_bids
from .data.split import split
from .evaluation.evaluate import evaluate
from .evaluation.inter_rater import compute_inter_rater


@click.group(no_args_is_help=True)
def cli():
    pass


cli.add_command(split)
cli.add_command(bids_to_nnunet)
cli.add_command(nnunet_outputs_to_bids)
cli.add_command(evaluate)
cli.add_command(compute_inter_rater)
