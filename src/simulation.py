from src.configurations import Configuration
from src.poisson_disk import PoissonDiskSampler
from src.constants import State, Color
from src.apic_solver import APIC

from abc import abstractmethod
from datetime import datetime

import taichi as ti
import os


@ti.data_oriented
class Simulation:
    def __init__(
        self,
        poisson_disk_sampler: PoissonDiskSampler,
        configurations: list[Configuration],
        initial_configuration: int,
        res: tuple[int, int],
        apic_solver: APIC,
        name: str,
    ) -> None:
        # State.
        self.is_paused = True
        self.should_write_to_disk = False
        self.is_showing_settings = not self.is_paused

        # Sampler and solver.
        self.poisson_disk_sampler = poisson_disk_sampler
        self.apic_solver = apic_solver

        # Create a parent directory, more directories will be created inside this
        # directory that contain newly created frames, videos and GIFs.
        self.parent_dir = ".output"
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)

        # Load the initial configuration and reset the solver to this configuration.
        self.current_frame = 0
        self.configurations = configurations
        self.configuration_id = initial_configuration
        self.load_configuration(configurations[self.configuration_id])

        # GGUI.
        self.window = ti.ui.Window(name, res)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    def substep(self) -> None:
        self.current_frame += 1

        # Load all remaining geometries with a satisfied frame threshold:
        if len(self.subsequent_geometries) > 0:
            if self.current_frame == self.subsequent_geometries[0].frame_threshold:
                geometry = self.subsequent_geometries.pop(0)
                self.add_geometry(geometry)

        for _ in range(int(2e-3 // self.apic_solver.dt)):
            self.apic_solver.reset_grids()
            self.apic_solver.particle_to_grid()
            self.apic_solver.momentum_to_velocity()
            self.apic_solver.classify_cells()
            self.apic_solver.compute_volumes()
            self.apic_solver.pressure_solver.solve()
            # self.apic_solver.heat_solver.solve()
            self.apic_solver.grid_to_particle()

    @ti.func
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        # Seed from the geometry and given position:
        self.apic_solver.conductivity_p[index] = geometry.conductivity
        self.apic_solver.temperature_p[index] = geometry.temperature
        self.apic_solver.capacity_p[index] = geometry.capacity
        self.apic_solver.velocity_p[index] = geometry.velocity
        # self.apic_solver.lambda_0_p[index] = geometry.lambda_0
        self.apic_solver.color_p[index] = geometry.color
        self.apic_solver.phase_p[index] = geometry.phase
        self.apic_solver.heat_p[index] = geometry.heat
        # self.apic_solver.mu_0_p[index] = geometry.mu_0
        self.apic_solver.position_p[index] = position

        # TODO: 5e9 would be a good test
        # self.apic_solver.lambda_0_p[index] = 5e9
        self.apic_solver.lambda_0_p[index] = 5e5
        self.apic_solver.mu_0_p[index] = 0

        # Set properties to default values:
        self.apic_solver.mass_p[index] = self.apic_solver.particle_vol * self.apic_solver.rho_0
        self.apic_solver.inv_lambda_p[index] = 1 / self.apic_solver.lambda_0[None]
        self.apic_solver.F_p[index] = ti.Matrix([[1, 0], [0, 1]])
        self.apic_solver.C_p[index] = ti.Matrix.zero(float, 2, 2)
        self.apic_solver.state_p[index] = State.Active
        self.apic_solver.JE_p[index] = 1.0
        self.apic_solver.JP_p[index] = 1.0

    @ti.kernel
    def add_geometry(self, geometry: ti.template()):  # pyright: ignore
        # Initialize background grid to the current positions:
        self.poisson_disk_sampler.initialize_grid(self.apic_solver.n_particles[None], self.apic_solver.position_p)

        # Update pointers, for a fresh sample this will be (0, 1), in the running simulation
        # this will reset this to where we left of, allowing to add more particles:
        self.poisson_disk_sampler.initialize_pointers(self.apic_solver.n_particles[None])

        # Find a good initial point for this sample run:
        initial_point = self.poisson_disk_sampler.generate_initial_point(geometry)
        self.add_particle(self.poisson_disk_sampler.tail(), initial_point, geometry)
        self.poisson_disk_sampler.increment_head()
        self.poisson_disk_sampler.increment_tail()

        while self.poisson_disk_sampler.can_sample_more_points():
            prev_position = self.apic_solver.position_p[self.poisson_disk_sampler._head[None]]
            self.poisson_disk_sampler.increment_head()  # Increment on each iteration

            for _ in range(self.poisson_disk_sampler.k):
                next_position = self.poisson_disk_sampler.generate_point_around(prev_position)
                next_index = self.poisson_disk_sampler.point_to_index(next_position)
                if self.poisson_disk_sampler.point_fits(next_position, geometry):
                    self.poisson_disk_sampler.background_grid[next_index] = self.poisson_disk_sampler.tail()
                    self.add_particle(self.poisson_disk_sampler.tail(), next_position, geometry)
                    self.poisson_disk_sampler.increment_tail()  # Increment when point is found

        # The head points to the last found position, this is the updated number of particles:
        self.apic_solver.n_particles[None] = self.poisson_disk_sampler._head[None]

    def load_configuration(self, configuration: Configuration) -> None:
        """
        Loads the chosen configuration into the MLS-MPM solver.
        ---
        Parameters:
            configuration: Configuration
        """
        self.apic_solver.ambient_temperature[None] = configuration.ambient_temperature
        self.apic_solver.lambda_0[None] = configuration.lambda_0
        self.apic_solver.theta_c[None] = configuration.theta_c
        self.apic_solver.theta_s[None] = configuration.theta_s
        self.apic_solver.zeta[None] = configuration.zeta
        self.apic_solver.mu_0[None] = configuration.mu_0
        self.apic_solver.nu[None] = configuration.nu
        self.apic_solver.E[None] = configuration.E
        self.configuration = configuration
        self.reset()

    def reset(self) -> None:
        """Reset the simulation."""

        # Reset the simulation:
        self.apic_solver.state_p.fill(State.Hidden)
        self.apic_solver.position_p.fill([42, 42])
        self.apic_solver.n_particles[None] = 0
        self.current_frame = 0

        # We copy this, so we can pop from this list and check the length:
        self.subsequent_geometries = self.configuration.subsequent_geometries.copy()

        # Load all the initial geometries into the solver:
        for geometry in self.configuration.initial_geometries:
            self.add_geometry(geometry)

    def dump_frames(self) -> None:
        """Creates an output directory, a VideoManager in this directory and then dumps frames to this directory."""
        date = datetime.now().strftime("%d%m%Y_%H%M%S")
        output_dir = f"{self.parent_dir}/{date}"
        os.makedirs(output_dir)
        self.video_manager = ti.tools.VideoManager(
            output_dir=output_dir,
            framerate=60,
            automatic_build=False,
        )

    def create_video(self) -> None:
        """Converts stored frames in the before created output directory to a video."""
        self.video_manager.make_video(gif=True, mp4=True)

    def show_configurations(self, subwindow) -> None:
        """
        Show all possible configurations in the subwindow, choosing one will
        load that configuration and reset the solver.
        ---
        Parameters:
            subwindow: GGUI subwindow
        """
        prev_configuration_id = self.configuration_id
        for i in range(len(self.configurations)):
            name = self.configurations[i].name
            if subwindow.checkbox(name, self.configuration_id == i):
                self.configuration_id = i
        if self.configuration_id != prev_configuration_id:
            _id = self.configuration_id
            configuration = self.configurations[_id]
            self.load_configuration(configuration)
            self.is_paused = True

    def show_parameters(self, subwindow) -> None:
        """
        Show all parameters in the subwindow, the user can then adjust these values
        with sliders which will update the correspoding value in the solver.
        ---
        Parameters:
            subwindow: GGUI subwindow
        """
        # TODO: Implement back stickiness + friction or remove them entirely
        # self.solver.stickiness[None] = subwindow.slider_float("stickiness", self.solver.stickiness[None], 1.0, 5.0)
        # self.solver.friction[None] = subwindow.slider_float("friction", self.solver.friction[None], 1.0, 5.0)
        self.apic_solver.theta_c[None] = subwindow.slider_float("theta_c", self.apic_solver.theta_c[None], 1e-2, 10e-2)
        self.apic_solver.theta_s[None] = subwindow.slider_float("theta_s", self.apic_solver.theta_s[None], 1e-3, 10e-3)
        self.apic_solver.zeta[None] = subwindow.slider_int("zeta", self.apic_solver.zeta[None], 3, 20)
        self.apic_solver.nu[None] = subwindow.slider_float("nu", self.apic_solver.nu[None], 0.1, 0.4)
        self.apic_solver.E[None] = subwindow.slider_float("E", self.apic_solver.E[None], 4.8e4, 5.5e5)
        E = self.apic_solver.E[None]
        nu = self.apic_solver.nu[None]
        self.apic_solver.lambda_0[None] = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.apic_solver.mu_0[None] = E / (2 * (1 + nu))

    def show_buttons(self, subwindow) -> None:
        """
        Show a set of buttons in the subwindow, this mainly holds functions to control the simulation.
        ---
        Parameters:
            subwindow: GGUI subwindow
        """
        if subwindow.button(" Stop recording  " if self.should_write_to_disk else " Start recording "):
            # This button toggles between saving frames and not saving frames.
            self.should_write_to_disk = not self.should_write_to_disk
            if self.should_write_to_disk:
                self.dump_frames()
            else:
                self.create_video()
        if subwindow.button(" Reset Particles "):
            self.reset()
        if subwindow.button(" Start Simulation"):
            self.is_paused = False

    def show_settings(self) -> None:
        """
        Show settings in a GGUI subwindow, this should be called once per generated frames
        and will only show these settings if the simulation is paused at the moment.
        """
        if not self.is_paused:
            self.is_showing_settings = False
            return  # don't bother
        self.is_showing_settings = True
        with self.gui.sub_window("Settings", 0.01, 0.01, 0.98, 0.98) as subwindow:
            self.show_parameters(subwindow)
            self.show_configurations(subwindow)
            self.show_buttons(subwindow)

    def handle_events(self) -> None:
        """Handle key presses arising from window events."""
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.reset()
            elif self.window.event.key in [ti.GUI.BACKSPACE, "s"]:
                self.should_write_to_disk = not self.should_write_to_disk
            elif self.window.event.key in [ti.GUI.SPACE, "p"]:
                self.is_paused = not self.is_paused
            elif self.window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                self.window.running = False  # Stop the simulation

    def render(self) -> None:
        """Renders the simulation with the data from the MLS-MPM solver."""
        self.canvas.set_background_color(Color.Background)
        self.canvas.circles(
            centers=self.apic_solver.position_p,
            color=Color.Water,
            radius=0.0015,
        )
        if self.should_write_to_disk and not self.is_paused and not self.is_showing_settings:
            self.video_manager.write_frame(self.window.get_image_buffer_as_numpy())
        self.window.show()

    def run(self) -> None:
        """Runs this simulation."""
        while self.window.running:
            self.handle_events()
            self.show_settings()
            if not self.is_paused:
                self.substep()
            self.render()
