import click
from pathlib import Path
from clinicadl.split import make_split


@click.command(no_args_is_help=True)
@click.argument(
    "df",
    type=Path,
)
@click.option(
    "--n_test",
    type=int,
    help="Number of participants in the test set.",
    default=15,
    show_default=True,
)
@click.option(
    "--output_dir",
    type=Path,
    help="The directory where to store the splits.",
    default=Path(".") / "data",
    show_default=True,
)
@click.option(
    "--p_med_cond_threshold",
    type=click.FloatRange(min=0, max_open=1),
    help="The p-value threshold for stratification on medical condition. The higher, the more similar training and test distributions must be.",
    default=0.9,
    show_default=True,
)
@click.option(
    "--p_age_threshold",
    type=click.FloatRange(min=0, max_open=1),
    help="The p-value threshold for stratification on age. The higher, the more similar training and test distributions must be.",
    default=0.95,
    show_default=True,
)
@click.option("--seed", type=int, help="A seed for reproducibility.")
def split(
    df: Path,
    n_test: int,
    output_dir: Path,
    p_med_cond_threshold: float,
    p_age_threshold: float,
    seed: int | None = None,
) -> None:
    """
    Split participants into training and test sets.

    Stratification is performed on age and medical condition.
    A random split is drawn and statistical tests are performed to see if age and medical condition distributions are
    similar in the training and test sets (see clinicadl.split.make_split); if they are not, another split is drawn.

    Args:\n
        df (Path) : The path to the TSV DataFrame containing the participants in a column named 'participant_id'. The DataFrame must also contain columns 'age' and 'medical_condition'.
    """
    split_dir = make_split(
        df,
        n_test=n_test,
        output_dir=output_dir,
        stratification=["age", "medical_condition"],
        p_categorical_threshold=p_med_cond_threshold,
        p_continuous_threshold=p_age_threshold,
        seed=seed,
    )
    click.echo(f"split saved in {split_dir}")
