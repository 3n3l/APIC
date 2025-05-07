from src.poisson_disk import PoissonDiskSampler
from src.configurations import Configuration
from src.constants import State, Color
from src.apic_solvers import APIC

from datetime import datetime

import taichi as ti
import os


@ti.data_oriented
class Simulation:
    def __init__(
        self,
        configurations: list[Configuration],
        initial_configuration: int,
        poisson_disk_sampler: PoissonDiskSampler,
        apic_solver: APIC,
        res: tuple[int, int],
        radius: float,
        name: str,
    ) -> None:
        # State.
        self.is_paused = True
        self.should_write_to_disk = False
        self.is_showing_settings = not self.is_paused

        # Sampler and solver.
        self.solver = apic_solver
        self.sampler = poisson_disk_sampler

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
        self.window = ti.ui.Window(name=name, res=res, fps_limit=60)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.radius = radius

    def substep(self) -> None:
        self.current_frame += 1

        # Load all remaining geometries with a satisfied frame threshold:
        if len(self.subsequent_geometries) > 0:
            if self.current_frame == self.subsequent_geometries[0].frame_threshold:
                geometry = self.subsequent_geometries.pop(0)
                self.add_geometry(geometry)

        self.solver.substep()

    @ti.kernel
    def add_geometry(self, geometry: ti.template()):  # pyright: ignore
        # Initialize background grid to the current positions:
        self.sampler.initialize_grid(self.solver.n_particles[None], self.solver.position_p)

        # Update pointers, for a fresh sample this will be (0, 1), in the running simulation
        # this will reset this to where we left of, allowing to add more particles:
        self.sampler.initialize_pointers(self.solver.n_particles[None])

        # Find a good initial point for this sample run:
        initial_point = self.sampler.generate_initial_point(geometry)
        self.solver.add_particle(self.sampler.tail(), initial_point, geometry)
        self.sampler.increment_head()
        self.sampler.increment_tail()

        while self.sampler.can_sample_more_points():
            prev_position = self.solver.position_p[self.sampler._head[None]]
            self.sampler.increment_head()  # Increment on each iteration

            for _ in range(self.sampler.k):
                next_position = self.sampler.generate_point_around(prev_position)
                next_index = self.sampler.point_to_index(next_position)
                if self.sampler.point_fits(next_position, geometry):
                    self.sampler.background_grid[next_index] = self.sampler.tail()
                    self.solver.add_particle(self.sampler.tail(), next_position, geometry)
                    self.sampler.increment_tail()  # Increment when point is found

        # The head points to the last found position, this is the updated number of particles:
        self.solver.n_particles[None] = self.sampler._head[None]

    def load_configuration(self, configuration: Configuration) -> None:
        """
        Loads the chosen configuration into the MLS-MPM solver.
        ---
        Parameters:
            configuration: Configuration
        """
        self.configuration = configuration
        self.reset()

    def reset(self) -> None:
        """Reset the simulation."""

        # Reset the simulation:
        self.solver.state_p.fill(State.Hidden)
        self.solver.position_p.fill([42, 42])
        self.solver.n_particles[None] = 0
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
        # TODO: the videomanager takes a postprocessing field,
        #       which just manipulates an ndarray and could be used to
        #       make the background transparent.
        self.video_manager = ti.tools.VideoManager(
            video_filename=f"APIC_{date}",
            output_dir=output_dir,
            automatic_build=False,
            framerate=60,
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
            centers=self.solver.position_p,
            radius=self.radius,
            color=Color.Water,
        )
        if self.should_write_to_disk and not self.is_paused and not self.is_showing_settings:
            self.video_manager.write_frame(self.window.get_image_buffer_as_numpy())
        self.window.show()

    def run(self) -> None:
        """Runs this simulation."""
        # TODO: move the iteration stuff to own method,
        #       let the user control how many frames should be recorded
        #       (framerate, gif-creation, video-creation could also all be controlled)
        # iteration = 0
        while self.window.running:
            # if iteration == 900:
            #     self.create_video()
            #     self.window.running = False
            self.handle_events()
            self.show_settings()
            if not self.is_paused:
                self.substep()
                # iteration += 1
            self.render()
