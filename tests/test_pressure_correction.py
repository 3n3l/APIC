import utils  # import first to append parent directory to path

from src.poisson_disk import PoissonDiskSampler
from src.configurations import Configuration
from src.presets import configuration_list
from src.simulation import Simulation
from src.parsing import arguments
from src.apic_solver import APIC

import taichi as ti
import numpy as np

MAX_ITERATIONS = 300
LOWER_BOUND = -1e-6
UPPER_BOUND = 1e-6


def print_wrt_bound(value: float) -> str:
    if value < LOWER_BOUND or value > UPPER_BOUND:
        return utils.print_red(str(value))
    else:
        return utils.print_green(str(value))


class TestRenderer(Simulation):
    def __init__(self, configurations: list[Configuration]) -> None:
        self.solver = APIC(quality=arguments.quality, max_particles=300_000)
        self.sampler = PoissonDiskSampler(apic_solver=self.solver, r=0.002, k=100)

        # Load the initial configuration and reset the solver to this configuration:
        self.current_frame = 0
        self.configuration_id = 0
        self.configurations = configurations
        self.load_configuration(configurations[self.configuration_id])

        # Fields for testing:
        self.total_divergence = ti.ndarray(ti.f32, shape=(self.solver.n_grid, self.solver.n_grid))
        self.curr_divergence = ti.ndarray(ti.f32, shape=(self.solver.n_grid, self.solver.n_grid))
        self.max_divergence = 0
        self.min_divergence = 0

    @ti.kernel
    def compute_divergence(self, div: ti.types.ndarray(), avg: ti.types.ndarray()):  # pyright: ignore
        for i, j in self.solver.mass_c:
            div[i, j] = 0
            if self.solver.is_interior(i, j):
                div[i, j] += self.solver.velocity_x[i + 1, j]
                div[i, j] -= self.solver.velocity_x[i, j]
                div[i, j] += self.solver.velocity_y[i, j + 1]
                div[i, j] -= self.solver.velocity_y[i, j]
                avg[i, j] += div[i, j]

    def run(self) -> None:
        self.total_divergence.fill(0)
        self.max_divergence = 0
        self.min_divergence = 0

        for i in range(MAX_ITERATIONS):
            self.substep()
            self.compute_divergence(self.curr_divergence, self.total_divergence)

            print(".", end=("\n" if i % 10 == 0 else " "), flush=True)

            divergence = self.curr_divergence.to_numpy()
            abs_curr_min = np.min(divergence)
            if abs_curr_min < self.min_divergence:
                self.min_divergence = np.min(divergence)
            abs_curr_max = np.abs(np.max(divergence))
            if abs_curr_max > np.abs(self.max_divergence):
                self.max_divergence = np.max(divergence)


def main() -> None:
    # Initialize Taichi on the chosen architecture:
    if arguments.arch.lower() == "cpu":
        ti.init(arch=ti.cpu, debug=False, verbose=False, log_level=ti.INFO)
    elif arguments.arch.lower() == "gpu":
        ti.init(arch=ti.gpu, debug=arguments.debug)
    else:
        ti.init(arch=ti.cuda, debug=arguments.debug)

    test_renderer = TestRenderer(configurations=configuration_list)

    results = []
    all_tests_succeeded = True
    for configuration in configuration_list:
        print(f"NOW RUNNING: {configuration.name}")
        test_renderer.load_configuration(configuration)
        test_renderer.run()

        average_divergence = test_renderer.total_divergence.to_numpy() / MAX_ITERATIONS
        min_average, max_average = np.min(average_divergence), np.max(average_divergence)
        min_spiking, max_spiking = test_renderer.min_divergence, test_renderer.max_divergence
        test_succeeded = min_average > LOWER_BOUND and max_average < UPPER_BOUND
        all_tests_succeeded &= test_succeeded
        result = (
            f"{configuration.name}\n"
            f"-> average min, max = {print_wrt_bound(min_average)}, {print_wrt_bound(max_average)}\n"
            f"-> spiking min, max = {print_wrt_bound(min_spiking)}, {print_wrt_bound(max_spiking)}\n"
            f"-> {utils.print_green("PASSED!") if test_succeeded else utils.print_red("DID NOT PASS!")}\n"
        )
        results.append(result)

    print(f"\n\niterations = {MAX_ITERATIONS}, lower bound = {LOWER_BOUND}, upper bound = {UPPER_BOUND}\n")
    print(*results, sep="\n", end="\n\n")
    print("\033[92m:)))))))))\033[0m" if all_tests_succeeded else "\033[91m:(\033[0m")


if __name__ == "__main__":
    main()
