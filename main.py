from src.parsing import arguments, should_use_cuda_backend, should_use_collocated
from src.apic_solvers import CollocatedAPIC, StaggeredAPIC
from src.poisson_disk import PoissonDiskSampler
from src.presets import configuration_list
from src.simulation import Simulation

import taichi as ti


def main():
    # Initialize Taichi on the chosen architecture:
    ti.init(arch=ti.cuda if should_use_cuda_backend else ti.cpu, debug=False)

    initial_configuration = arguments.configuration % len(configuration_list)
    simulation_name = f"{'Collocated' if should_use_collocated else 'Staggered'} Affine Particle-In-Cell Method"

    # The radius for the particles and the Poisson-Disk Sampler:
    radius = 0.0012
    quality = 1
    max_particles = 500_000

    solver = CollocatedAPIC(quality, max_particles) if should_use_collocated else StaggeredAPIC(quality, max_particles)
    sampler = PoissonDiskSampler(apic_solver=solver, r=radius * 1.1, k=100)

    simulation = Simulation(
        initial_configuration=initial_configuration,
        configurations=configuration_list,
        poisson_disk_sampler=sampler,
        apic_solver=solver,
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
