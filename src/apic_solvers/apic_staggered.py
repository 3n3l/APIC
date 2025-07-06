from taichi.linalg import SparseMatrixBuilder, SparseCG
from src.constants import State, GRAVITY
from src.apic_solvers import APIC

import taichi as ti


@ti.data_oriented
class StaggeredAPIC(APIC):
    def __init__(self, max_particles: int, n_grid: int, dt: float):
        super().__init__(max_particles, n_grid, dt)

        # Properties on MAC-faces:
        self.classification_x = ti.field(dtype=ti.int8, shape=(self.n_grid + 1, self.n_grid))
        self.classification_y = ti.field(dtype=ti.int8, shape=(self.n_grid, self.n_grid + 1))
        self.velocity_x = ti.field(dtype=ti.f32, shape=(self.n_grid + 1, self.n_grid))
        self.velocity_y = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid + 1))
        self.volume_x = ti.field(dtype=ti.f32, shape=(self.n_grid + 1, self.n_grid))
        self.volume_y = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid + 1))
        self.mass_x = ti.field(dtype=ti.f32, shape=(self.n_grid + 1, self.n_grid))
        self.mass_y = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid + 1))

        # Properties on particles:
        self.cx_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.cy_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)

        # Offsets for weight computations:
        self.base_offset_x = ti.Vector([0.5, 1.0])
        self.base_offset_y = ti.Vector([1.0, 0.5])
        self.dist_offset_x = ti.Vector([0.0, 0.5])
        self.dist_offset_y = ti.Vector([0.5, 0.0])

    @ti.func
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        # Seed from the geometry and given position:
        self.velocity_p[index] = geometry.velocity
        self.position_p[index] = position

        # Set properties to default values:
        self.state_p[index] = State.Active
        self.cx_p[index] = 0
        self.cy_p[index] = 0

    @ti.kernel
    def reset_grids(self):
        for i, j in self.velocity_x:
            self.velocity_x[i, j] = 0
            self.volume_x[i, j] = 0
            self.mass_x[i, j] = 0

        for i, j in self.velocity_y:
            self.velocity_y[i, j] = 0
            self.volume_y[i, j] = 0
            self.mass_y[i, j] = 0

    @ti.kernel
    def particle_to_grid(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Lower left corner of the interpolation grid:
            base_x = ti.floor((self.position_p[p] * self.inv_dx - self.base_offset_x), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - self.base_offset_y), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - self.dist_offset_x
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - self.dist_offset_y

            # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2):
            # Based on https://www.bilibili.com/opus/662560355423092789
            w_x = [0.5 * (1.5 - dist_x) ** 2, 0.75 - (dist_x - 1) ** 2, 0.5 * (dist_x - 0.5) ** 2]
            w_y = [0.5 * (1.5 - dist_y) ** 2, 0.75 - (dist_y - 1) ** 2, 0.5 * (dist_y - 0.5) ** 2]

            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                weight_x = w_x[i][0] * w_x[j][1]
                weight_y = w_y[i][0] * w_y[j][1]
                self.mass_x[base_x + offset] += weight_x
                self.mass_y[base_y + offset] += weight_y
                dpos_x = ti.cast(offset - dist_x, ti.f32) * self.dx
                dpos_y = ti.cast(offset - dist_y, ti.f32) * self.dx
                velocity_x = self.velocity_p[p][0] + (self.cx_p[p] @ dpos_x)
                velocity_y = self.velocity_p[p][1] + (self.cy_p[p] @ dpos_y)
                self.velocity_x[base_x + offset] += weight_x * velocity_x
                self.velocity_y[base_y + offset] += weight_y * velocity_y

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.velocity_x:
            if (mass := self.mass_x[i, j]) > 0:
                self.velocity_x[i, j] /= mass
                collision_right = i >= (self.n_grid - self.boundary_width) and self.velocity_x[i, j] > 0
                collision_left = i <= self.boundary_width and self.velocity_x[i, j] < 0
                if collision_left or collision_right:
                    self.velocity_x[i, j] = 0

        for i, j in self.velocity_y:
            if (mass := self.mass_y[i, j]) > 0:
                self.velocity_y[i, j] /= mass
                self.velocity_y[i, j] += GRAVITY * self.dt
                collision_top = j >= (self.n_grid - self.boundary_width) and self.velocity_y[i, j] > 0
                collision_bottom = j <= self.boundary_width and self.velocity_y[i, j] < 0
                if collision_top or collision_bottom:
                    self.velocity_y[i, j] = 0

    @ti.kernel
    def compute_volumes(self):
        # FIXME: this control volume doesn't help with the density correction
        control_volume = 0.5 * self.dx * self.dx
        for i, j in self.classification_c:
            if self.is_interior(i, j):
                self.volume_x[i + 1, j] += control_volume
                self.volume_x[i, j] += control_volume
                self.volume_y[i, j + 1] += control_volume
                self.volume_y[i, j] += control_volume

    @ti.kernel
    def fill_pressure_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
        coefficient = self.dt * self.inv_dx * self.inv_dx
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            center = 0.0  # to keep max_num_triplets as low as possible
            idx = (i * self.n_grid) + j  # raveled index
            if self.is_interior(i, j):
                # Build the right-hand side of the linear system.
                # This uses a modified divergence, where the velocities of faces
                # bordering colliding (solid) cells are considered to be zero.
                if not self.is_colliding(i + 1, j):
                    b[idx] += self.inv_dx * self.velocity_x[i + 1, j]
                if not self.is_colliding(i - 1, j):
                    b[idx] -= self.inv_dx * self.velocity_x[i, j]
                if not self.is_colliding(i, j + 1):
                    b[idx] += self.inv_dx * self.velocity_y[i, j + 1]
                if not self.is_colliding(i, j - 1):
                    b[idx] -= self.inv_dx * self.velocity_y[i, j]

                # We will apply a Neumann boundary condition on the colliding faces,
                # to guarantee zero flux into colliding cells, by just not adding these
                # face values in the Laplacian for the off-diagonal values.
                # NOTE: we can use the raveled index to quickly access adjacent cells with:
                # idx(i, j) = (i * n) + j
                #   => idx(i - 1, j) = ((i - 1) * n) + j = (i * n) + j - n = idx(i, j) - n
                #   => idx(i, j - 1) = (i * n) + j - 1 = idx(i, j) - 1, etc.
                if not self.is_colliding(i - 1, j):
                    inv_rho = self.volume_x[i, j] / self.mass_x[i, j]
                    center -= coefficient * inv_rho
                    if self.is_interior(i - 1, j):
                        A[idx, idx - self.n_grid] += coefficient * inv_rho

                if not self.is_colliding(i + 1, j):
                    inv_rho = self.volume_x[i + 1, j] / self.mass_x[i + 1, j]
                    center -= coefficient * inv_rho
                    if self.is_interior(i + 1, j):
                        A[idx, idx + self.n_grid] += coefficient * inv_rho

                if not self.is_colliding(i, j - 1):
                    inv_rho = self.volume_y[i, j] / self.mass_y[i, j]
                    center -= coefficient * inv_rho
                    if self.is_interior(i, j - 1):
                        A[idx, idx - 1] += coefficient * inv_rho

                if not self.is_colliding(i, j + 1):
                    inv_rho = self.volume_y[i, j + 1] / self.mass_y[i, j + 1]
                    center -= coefficient * inv_rho
                    if self.is_interior(i, j + 1):
                        A[idx, idx + 1] += coefficient * inv_rho

                A[idx, idx] += center

            else:  # Homogeneous Dirichlet boundary condition.
                A[idx, idx] += 1.0
                b[idx] = 0.0

    @ti.kernel
    def apply_pressure(self, pressure: ti.types.ndarray()):  # pyright: ignore
        coefficient = self.dt * self.inv_dx
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            idx = i * self.n_grid + j
            if self.is_interior(i - 1, j) or self.is_interior(i, j):
                if not (self.is_colliding(i - 1, j) or self.is_colliding(i, j)):
                    pressure_gradient = pressure[idx] - pressure[idx - self.n_grid]
                    inv_rho = self.volume_x[i, j] / self.mass_x[i, j]
                    self.velocity_x[i, j] -= inv_rho * coefficient * pressure_gradient
                else:
                    self.velocity_x[i, j] = 0
            if self.is_interior(i, j - 1) or self.is_interior(i, j):
                if not (self.is_colliding(i, j - 1) or self.is_colliding(i, j)):
                    pressure_gradient = pressure[idx] - pressure[idx - 1]
                    inv_rho = self.volume_y[i, j] / self.mass_y[i, j]
                    self.velocity_y[i, j] -= inv_rho * coefficient * pressure_gradient
                else:
                    self.velocity_y[i, j] = 0

    def correct_pressure(self):
        n_cells = self.n_grid * self.n_grid
        A = SparseMatrixBuilder(max_num_triplets=(5 * n_cells), num_rows=n_cells, num_cols=n_cells, dtype=ti.f32)
        b = ti.ndarray(ti.f32, shape=n_cells)
        self.fill_pressure_system(A, b)

        # Solve the linear system, apply the resulting pressure:
        solver = SparseCG(A.build(), b, atol=1e-5, max_iter=500)
        self.apply_pressure(solver.solve()[0])

    @ti.kernel
    def grid_to_particle(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Lower left corner of the interpolation grid:
            base_x = ti.floor((self.position_p[p] * self.inv_dx - self.base_offset_x), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - self.base_offset_y), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - self.dist_offset_x
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - self.dist_offset_y

            # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2)
            # Based on https://www.bilibili.com/opus/662560355423092789
            w_x = [0.5 * (1.5 - dist_x) ** 2, 0.75 - (dist_x - 1) ** 2, 0.5 * (dist_x - 0.5) ** 2]
            w_y = [0.5 * (1.5 - dist_y) ** 2, 0.75 - (dist_y - 1) ** 2, 0.5 * (dist_y - 0.5) ** 2]

            next_velocity = ti.Vector.zero(ti.f32, 2)
            b_x = ti.Vector.zero(ti.f32, 2)
            b_y = ti.Vector.zero(ti.f32, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                x_weight = w_x[i][0] * w_x[j][1]
                y_weight = w_y[i][0] * w_y[j][1]
                offset = ti.Vector([i, j])
                dpos_x = ti.cast(offset, ti.f32) - dist_x
                dpos_y = ti.cast(offset, ti.f32) - dist_y
                x_velocity = x_weight * self.velocity_x[base_x + offset]
                y_velocity = y_weight * self.velocity_y[base_y + offset]
                next_velocity += [x_velocity, y_velocity]
                b_x += x_velocity * dpos_x
                b_y += y_velocity * dpos_y

            # We compute c_x, c_y from b_x, b_y as in https://doi.org/10.1016/j.jcp.2020.109311,
            # this avoids computing the weight gradients and results in less dissipation.
            # C = B @ (D^(-1)), NOTE: one inv_dx is cancelled with one dx in dpos.
            self.cx_p[p] = b_x * 4 * self.inv_dx
            self.cy_p[p] = b_y * 4 * self.inv_dx
            self.velocity_p[p] = next_velocity
            self.position_p[p] += self.dt * next_velocity

    def substep(self) -> None:
        for _ in range(4 * int(2e-3 // self.dt)):
            self.reset_grids()
            self.particle_to_grid()
            self.classify_cells()
            self.momentum_to_velocity()
            self.compute_volumes()
            self.correct_pressure()
            self.grid_to_particle()
