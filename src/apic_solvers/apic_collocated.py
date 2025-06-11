from taichi.linalg import SparseMatrixBuilder, SparseCG
from src.constants import State, GRAVITY
from src.apic_solvers import APIC

import taichi as ti


@ti.data_oriented
class CollocatedAPIC(APIC):
    def __init__(self, quality: int, max_particles: int):
        super().__init__(quality, max_particles)

        self.rho_0 = 1000 # TODO: implement density correction

        # Properties on MAC-cells:
        self.velocity_c = ti.Vector.field(2, dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.volume_c = ti.Vector.field(2, dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.mass_c = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))

        # Properties on particles:
        self.C_p = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)

    @ti.func
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        # Seed from the geometry and given position:
        self.velocity_p[index] = geometry.velocity
        self.position_p[index] = position

        # Set properties to default values:
        self.state_p[index] = State.Active
        self.C_p[index] = 0

    @ti.kernel
    def reset_grids(self):
        for i, j in self.velocity_c:
            self.velocity_c[i, j] = 0
            self.mass_c[i, j] = 0

    @ti.kernel
    def particle_to_grid(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Lower left corner of the interpolation grid:
            base_c = ti.floor((self.position_p[p] * self.inv_dx - 0.5), ti.i32)

            # Distance between lower left corner and particle position:
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32)

            # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2)
            # Based on https://www.bilibili.com/opus/662560355423092789
            w_c = [0.5 * (1.5 - dist_c) ** 2, 0.75 - (dist_c - 1) ** 2, 0.5 * (dist_c - 0.5) ** 2]

            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                weight_c = w_c[i][0] * w_c[j][1]
                self.mass_c[base_c + offset] += weight_c
                dpos_c = ti.cast(offset - dist_c, ti.f32) * self.dx
                velocity_c = self.velocity_p[p] + (self.C_p[p] @ dpos_c)
                self.velocity_c[base_c + offset] += weight_c * velocity_c

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.velocity_c:
            if (mass := self.mass_c[i, j]) > 0:
                # Normalize velocity:
                self.velocity_c[i, j] /= mass

                # Apply gravity:
                self.velocity_c[i, j] += [0, GRAVITY * self.dt]

                # Slip boundary condition:
                collision_right = i >= (self.n_grid - self.boundary_width) and self.velocity_c[i, j][0] > 0
                collision_left = i <= self.boundary_width and self.velocity_c[i, j][0] < 0
                if collision_left or collision_right:
                    self.velocity_c[i, j][0] = 0
                collision_top = j >= (self.n_grid - self.boundary_width) and self.velocity_c[i, j][1] > 0
                collision_bottom = j <= self.boundary_width and self.velocity_c[i, j][1] < 0
                if collision_top or collision_bottom:
                    self.velocity_c[i, j][1] = 0

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
                    b[idx] += self.inv_dx * self.velocity_c[i + 1, j][0]
                if not self.is_colliding(i - 1, j):
                    b[idx] -= self.inv_dx * self.velocity_c[i, j][0]
                if not self.is_colliding(i, j + 1):
                    b[idx] += self.inv_dx * self.velocity_c[i, j + 1][1]
                if not self.is_colliding(i, j - 1):
                    b[idx] -= self.inv_dx * self.velocity_c[i, j][1]

                # We will apply a Neumann boundary condition on the colliding faces,
                # to guarantee zero flux into colliding cells, by just not adding these
                # face values in the Laplacian for the off-diagonal values.
                # NOTE: we can use the raveled index to quickly access adjacent cells with:
                # idx(i, j) = (i * n) + j
                #   => idx(i - 1, j) = ((i - 1) * n) + j = (i * n) + j - n = idx(i, j) - n
                #   => idx(i, j - 1) = (i * n) + j - 1 = idx(i, j) - 1, etc.
                if not self.is_colliding(i - 1, j):
                    inv_rho = 1 / self.rho_0
                    center -= coefficient * inv_rho
                    if self.is_interior(i - 1, j):
                        A[idx, idx - self.n_grid] += coefficient * inv_rho

                if not self.is_colliding(i + 1, j):
                    inv_rho = 1 / self.rho_0
                    center -= coefficient * inv_rho
                    if self.is_interior(i + 1, j):
                        A[idx, idx + self.n_grid] += coefficient * inv_rho

                if not self.is_colliding(i, j - 1):
                    inv_rho = 1 / self.rho_0
                    center -= coefficient * inv_rho
                    if self.is_interior(i, j - 1):
                        A[idx, idx - 1] += coefficient * inv_rho

                if not self.is_colliding(i, j + 1):
                    inv_rho = 1 / self.rho_0
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
                    inv_rho = 1 / self.rho_0
                    self.velocity_c[i, j][0] -= inv_rho * coefficient * pressure_gradient
                else:
                    self.velocity_c[i, j][0] = 0
            if self.is_interior(i, j - 1) or self.is_interior(i, j):
                if not (self.is_colliding(i, j - 1) or self.is_colliding(i, j)):
                    pressure_gradient = pressure[idx] - pressure[idx - 1]
                    inv_rho = 1 / self.rho_0
                    self.velocity_c[i, j][1] -= inv_rho * coefficient * pressure_gradient
                else:
                    self.velocity_c[i, j][1] = 0

    def correct_pressure(self):
        n_cells = self.n_grid * self.n_grid
        A = SparseMatrixBuilder(max_num_triplets=(5 * n_cells), num_rows=n_cells, num_cols=n_cells, dtype=ti.f32)
        b = ti.ndarray(ti.f32, shape=n_cells)
        self.fill_pressure_system(A, b)

        # Solve the linear system, apply the resulting pressure:
        solver = SparseCG(A.build(), b, atol=1e-6, max_iter=500)
        self.apply_pressure(solver.solve()[0])

    @ti.kernel
    def grid_to_particle(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Lower left corner of the interpolation grid:
            base_c = ti.floor((self.position_p[p] * self.inv_dx - 0.5), ti.i32)

            # Distance between lower left corner and particle position:
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32)

            # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2)
            # Based on https://www.bilibili.com/opus/662560355423092789
            w_c = [0.5 * (1.5 - dist_c) ** 2, 0.75 - (dist_c - 1) ** 2, 0.5 * (dist_c - 0.5) ** 2]

            velocity = ti.Vector.zero(ti.f32, 2)
            B = ti.Matrix.zero(ti.f32, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                weight_c = w_c[i][0] * w_c[j][1]
                velocity_c = self.velocity_c[base_c + offset]
                velocity += weight_c * velocity_c
                dpos_c = ti.cast(offset, ti.f32) - dist_c
                B += weight_c * velocity_c.outer_product(dpos_c)

            # We compute c_x, c_y from b_x, b_y as in https://doi.org/10.1016/j.jcp.2020.109311,
            # this avoids computing the weight gradients and results in less dissipation.
            # C = B @ (D^(-1)), NOTE: one inv_dx is cancelled with one dx in dpos.
            self.C_p[p] = 4 * self.inv_dx * B
            self.position_p[p] += self.dt * velocity
            self.velocity_p[p] = velocity

    def substep(self) -> None:
        # TODO: find good ratio of timestep and iterations per timestep
        for _ in range(4):
            self.reset_grids()
            self.particle_to_grid()
            self.classify_cells()
            self.momentum_to_velocity()
            self.correct_pressure()
            self.grid_to_particle()
