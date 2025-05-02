import utils  # import first to append parent directory to path

from src.configurations import Configuration, Rectangle, Circle
from src.poisson_disk import PoissonDiskSampler

from src.presets import configuration_list
from src.constants import Classification
from src.simulation import Simulation
from src.apic_solver import APIC
from src.parsing import arguments

import taichi as ti
import numpy as np

offset = 0.0234375

class TestSimulation(Simulation):
    def __init__(self, configurations: list[Configuration], initial_configuration: int = 0) -> None:
        # State.
        self.is_paused = True
        self.should_write_to_disk = False
        self.is_showing_settings = not self.is_paused

        # Sampler and solver.
        self.solver = APIC(quality=arguments.quality, max_particles=100_000)
        self.sampler = PoissonDiskSampler(apic_solver=self.solver, r=0.0018, k=300)

        # Load the initial configuration and reset the solver to this configuration.
        self.current_frame = 0
        self.configurations = configurations
        self.configuration_id = initial_configuration
        self.load_configuration(configurations[self.configuration_id])

        # Fields for testing:
        self.divergence = ti.ndarray(ti.f32, shape=(self.solver.n_grid, self.solver.n_grid))
        self.we_succeeded = True

    @ti.kernel
    def compute_divergence(self, div: ti.types.ndarray()):  # pyright: ignore
        for i, j in self.solver.mass_c:
            div[i, j] = 0
            if self.solver.is_interior(i, j):
                if not self.solver.is_colliding(i + 1, j):
                    div[i, j] += self.solver.velocity_x[i + 1, j]
                if not self.solver.is_colliding(i - 1, j):
                    div[i, j] -= self.solver.velocity_x[i, j]
                if not self.solver.is_colliding(i, j + 1):
                    div[i, j] += self.solver.velocity_y[i, j + 1]
                if not self.solver.is_colliding(i, j - 1):
                    div[i, j] -= self.solver.velocity_y[i, j]

    def run(self) -> None:
        for i in range(1, 301):
            self.substep()
            prev_min = np.round(np.abs(np.min(self.divergence.to_numpy())))
            prev_max = np.round(np.abs(np.max(self.divergence.to_numpy())))
            self.compute_divergence(self.divergence)
            curr_min = np.round(np.abs(np.min(self.divergence.to_numpy())))
            curr_max = np.round(np.abs(np.max(self.divergence.to_numpy())))

            print(".", end=("\n" if i % 10 == 0 else " "), flush=True)

            if curr_min > prev_min or curr_max > prev_max:
                # The solver actually increased the divergence :(
                print("\n\nDivergence increased :(")
                print(f"prev_min = {prev_min}, prev_max = {prev_max}")
                print(f"curr_min = {curr_min}, curr_max = {curr_max}")
                self.we_succeeded = False
                break

            if np.any(np.round(self.divergence.to_numpy(), 2) != 0):
                print(f"\n\nDivergence too big :(")
                self.we_succeeded = False
                break

            if not self.we_succeeded:
                break


def main() -> None:
    # Initialize Taichi on the chosen architecture:
    if arguments.arch.lower() == "cpu":
        ti.init(arch=ti.cpu, debug=True, verbose=False)
    elif arguments.arch.lower() == "gpu":
        ti.init(arch=ti.gpu, debug=True)
    else:
        ti.init(arch=ti.cuda, debug=True)

    test_simulation = TestSimulation(
        initial_configuration=arguments.configuration,
        configurations=configuration_list,
    )

    for configuration in configuration_list:
        print(f"NOW RUNNING: {configuration.name}")
        test_simulation.load_configuration(configuration)
        test_simulation.run()
        if not test_simulation.we_succeeded:
            break

    print("\n")
    print("Divergence, min ->", np.min(test_simulation.divergence.to_numpy()))
    print("Divergence, max ->", np.max(test_simulation.divergence.to_numpy()))
    print()

    print("\033[92m:)))))))))\033[0m" if test_simulation.we_succeeded else "\033[91m:(\033[0m")


if __name__ == "__main__":
    main()
