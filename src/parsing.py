from src.presets import configuration_list

from argparse import ArgumentParser, RawTextHelpFormatter

epilog = "Press R to reset, SPACE to pause/unpause the simulation!"
parser = ArgumentParser(prog="main.py", epilog=epilog, formatter_class=RawTextHelpFormatter)

configuration_help = (
    f"Available Configurations:\n{'\n'.join([f'[{i}] -> {c.name}' for i, c in enumerate(configuration_list)])}"
)
parser.add_argument(
    "-c",
    "--configuration",
    default=0,
    nargs="?",
    help=configuration_help,
    type=int,
)

quality_help = "Choose a quality multiplicator for the simulation (higher is better)."
parser.add_argument(
    "-q",
    "--quality",
    default=1,
    nargs="?",
    help=quality_help,
    type=int,
)

solver_type_help = "Choose the Taichi architecture to run on."
parser.add_argument(
    "-a",
    "--arch",
    default="CPU",
    nargs="?",
    choices=["CPU", "CUDA"],
    help=solver_type_help,
)

solver_type_help = "Choose the grid type (collocated or staggered)"
parser.add_argument(
    "-g",
    "--grid",
    default="Staggered",
    nargs="?",
    choices=["Staggered", "Collocated"],
    help=solver_type_help,
)

arguments = parser.parse_args()

# Parsed constants:
should_use_cuda_backend = arguments.arch.lower() == "cuda"
should_use_collocated = arguments.grid.lower() == "collocated"
