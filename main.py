from src.parsing import arguments, should_use_cuda_backend
from src.presets import configuration_list
from src.simulation import Simulation

import taichi as ti


def main():
    # Initialize Taichi on the chosen architecture:
    ti.init(arch=ti.cuda if should_use_cuda_backend else ti.cpu, debug=False)

    initial_configuration = arguments.configuration % len(configuration_list)
    simulation_name = "APIC"
    radius = 0.0018

    simulation = Simulation(
        initial_configuration=initial_configuration,
        configurations=configuration_list,
        quality=arguments.quality,
        name=simulation_name,
        res=(720, 720),
        radius=radius,
    )

    simulation.run()

    print("\n", "#" * 100, sep="")
    print("###", simulation_name)
    print("#" * 100)
    print(">>> R        -> [R]eset the simulation.")
    print(">>> P|SPACE  -> [P]ause/Un[P]ause the simulation.")
    print()


if __name__ == "__main__":
    main()
