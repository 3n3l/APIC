from src.constants import Classification, State
from abc import abstractmethod

import taichi as ti


@ti.data_oriented
class APIC:
    def __init__(self, quality: int, max_particles: int):
        self.max_particles = max_particles
        self.n_grid = 128 * quality
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = 2e-3 / quality
        self.inv_dt = 1 / self.dt
        self.vol_p = (self.dx * 0.5) ** 2

        # Variable properties, must be stored in fields:
        self.n_particles = ti.field(dtype=ti.int32, shape=())

        # Properties on MAC-cells:
        self.classification_c = ti.field(dtype=ti.int8, shape=(self.n_grid, self.n_grid))

        # Properties on particles:
        self.position_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.velocity_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.state_p = ti.field(dtype=ti.f32, shape=max_particles)

        # The width of the simulation boundary in grid nodes and offsets to
        # guarantee that seeded particles always lie within the boundary:
        self.boundary_width = 3
        self.lower = self.boundary_width * self.dx
        self.upper = 1 - self.lower

        # Now we can initialize the colliding boundary (or bounding box) around the domain:
        self.initialize_boundary()

    @abstractmethod
    def reset_grids(self):
        pass

    @abstractmethod
    def particle_to_grid(self):
        pass

    @abstractmethod
    def momentum_to_velocity(self):
        pass

    @abstractmethod
    def correct_pressure(self):
        pass

    @abstractmethod
    def grid_to_particle(self):
        pass

    @abstractmethod
    def substep(self):
        pass

    @ti.func
    def in_bounds(self, x: float, y: float) -> bool:
        return self.lower < x < self.upper and self.lower < y < self.upper

    @ti.func
    def is_valid(self, i: int, j: int) -> bool:
        return i >= 0 and i <= self.n_grid - 1 and j >= 0 and j <= self.n_grid - 1

    @ti.func
    def is_colliding(self, i: int, j: int) -> bool:
        return self.is_valid(i, j) and self.classification_c[i, j] == Classification.Colliding

    @ti.func
    def is_interior(self, i: int, j: int) -> bool:
        return self.is_valid(i, j) and self.classification_c[i, j] == Classification.Interior

    @ti.func
    def is_empty(self, i: int, j: int) -> bool:
        return self.is_valid(i, j) and self.classification_c[i, j] == Classification.Empty

    @abstractmethod
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        pass

    @ti.kernel
    def initialize_boundary(self):
        for i, j in self.classification_c:
            is_colliding = not (self.boundary_width <= i < self.n_grid - self.boundary_width)
            is_colliding |= not (self.boundary_width <= j < self.n_grid - self.boundary_width)
            if is_colliding:
                self.classification_c[i, j] = Classification.Colliding
            else:
                self.classification_c[i, j] = Classification.Empty


    @ti.kernel
    def classify_cells(self):
        for i, j in self.classification_c:
            # Reset all the cells that don't belong to the colliding boundary:
            if not self.is_colliding(i, j):
                self.classification_c[i, j] = Classification.Empty

        for p in self.velocity_p:
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Find the nearest cell and set it to interior:
            i, j = ti.cast(self.position_p[p] * self.inv_dx, int)  # pyright: ignore
            if not self.is_colliding(i, j):  # pyright: ignore
                self.classification_c[i, j] = Classification.Interior
