from src.parsing import arguments, should_use_cuda_backend
from src.poisson_disk import PoissonDiskSampler
from src.presets import configuration_list
from src.simulation import Simulation
from src.apic_solver import APIC

import taichi as ti


def main():
    # Initialize Taichi on the chosen architecture:
    if should_use_cuda_backend:
        ti.init(arch=ti.cuda, debug=False)
    else:
        ti.init(arch=ti.cpu, debug=True)
        # ti.init(arch=ti.gpu, debug=False)

    apic_solver = APIC(quality=arguments.quality, max_particles=100_000)
    poisson_disk_sampler = PoissonDiskSampler(apic_solver=apic_solver)

    simulation_name = "APIC"
    initial_configuration = arguments.configuration % len(configuration_list)
    simulation = Simulation(
        initial_configuration=initial_configuration,
        poisson_disk_sampler=poisson_disk_sampler,
        configurations=configuration_list,
        name=simulation_name,
        apic_solver=apic_solver,
        res=(720, 720),
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
