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

solver_type_help = "Choose whether to use a direct or iterative solver for solving the pressure equation."
parser.add_argument(
    "-s",
    "--solverType",
    default="Iterative",
    nargs="?",
    choices=["Direct", "Iterative"],
    help=solver_type_help,
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
    choices=["CPU", "GPU", "CUDA"],
    help=solver_type_help,
)

arguments = parser.parse_args()

# Parsed constants:
should_use_direct_solver = arguments.solverType.lower() == "direct"
should_use_cuda_backend = arguments.architecture.lower() == "cuda"
