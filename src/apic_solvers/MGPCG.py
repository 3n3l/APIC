from src.constants import Classification
import taichi as ti


@ti.data_oriented
class Pressure_MGPCGSolver:
    def __init__(self, apic_solver):
        # TODO: only one variable needed here
        self.solver = apic_solver
        self.n_grid = apic_solver.n_grid

        self.velocity_x = apic_solver.velocity_x
        self.velocity_y = apic_solver.velocity_y
        self.volume_x = apic_solver.volume_x
        self.volume_y = apic_solver.volume_y
        self.mass_x = apic_solver.mass_x
        self.mass_y = apic_solver.mass_y

        self.inv_dx = apic_solver.inv_dx
        self.dx = apic_solver.dx
        self.dt = apic_solver.dt

        # self.inv_lambda_c = apic_solver.cell_inv_lambda

        # self.JP_c = apic_solver.cell_JP
        # self.JE_c = apic_solver.cell_JE
        # self.dt = apic_solver.dt

        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 5
        self.n_grid_levels = 3

        # rhs of linear system
        self.b = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))

        # TODO: make lambda
        def grid_shape(l):
            return (self.n_grid // 2**l, self.n_grid // 2**l)

        # lhs of linear system and its corresponding form in coarse grids
        self.Adiag = [ti.field(dtype=ti.f32, shape=grid_shape(l)) for l in range(self.n_grid_levels)]
        self.Ax = [ti.field(dtype=ti.f32, shape=grid_shape(l)) for l in range(self.n_grid_levels)]
        self.Ay = [ti.field(dtype=ti.f32, shape=grid_shape(l)) for l in range(self.n_grid_levels)]

        # grid type
        self.classification_c = apic_solver.classification_c
        self.classifications = [ti.field(dtype=ti.i32, shape=grid_shape(l)) for l in range(self.n_grid_levels)]

        # pcg var
        self.r = [ti.field(dtype=ti.f32, shape=grid_shape(l)) for l in range(self.n_grid_levels)]
        self.z = [ti.field(dtype=ti.f32, shape=grid_shape(l)) for l in range(self.n_grid_levels)]

        self.p = apic_solver.pressure_c

        self.s = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.As = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.reduction = ti.field(dtype=ti.f32, shape=())
        self.alpha = ti.field(dtype=ti.f32, shape=())
        self.beta = ti.field(dtype=ti.f32, shape=())

        # self.N_ext = self.n_grid // 2  # number of ext cells set so that that total grid size is still power of 2
        # self.N_tot = 2 * self.n_grid

        # self.grid = ti.root.pointer(ti.ij, [self.N_tot // 4]).dense(ti.ij, 4).place(self.pressure_c, self.p, self.Ap)
        # for l in range(self.n_grid_levels):
        #     self.grid = ti.root.pointer(ti.ij, [self.N_tot // (4 * 2**l)]).dense(ti.ij, 4).place(self.r[l], self.z[l])

        # ti.root.place(self.alpha, self.beta, self.sum)

    @ti.kernel
    def _initialize(self):
        coefficient = self.dt * self.inv_dx * self.inv_dx
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            if self.solver.is_interior(i, j):
                # Build the right-hand side of the linear system.
                # This uses a modified divergence, where the velocities of faces
                # bordering colliding (solid) cells are considered to be zero.
                if not self.solver.is_colliding(i + 1, j):
                    self.b[i, j] += self.inv_dx * self.velocity_x[i + 1, j]
                if not self.solver.is_colliding(i - 1, j):
                    self.b[i, j] -= self.inv_dx * self.velocity_x[i, j]
                if not self.solver.is_colliding(i, j + 1):
                    self.b[i, j] += self.inv_dx * self.velocity_y[i, j + 1]
                if not self.solver.is_colliding(i, j - 1):
                    self.b[i, j] -= self.inv_dx * self.velocity_y[i, j]

                # We will apply a Neumann boundary condition on the colliding faces,
                # to guarantee zero flux into colliding cells, by just not adding these
                # face values in the Laplacian for the off-diagonal values.
                if not self.solver.is_colliding(i - 1, j):
                    inv_rho = self.volume_x[i, j] / self.mass_x[i, j]
                    self.Adiag[0][i, j] -= coefficient * inv_rho
                    if self.solver.is_interior(i - 1, j):
                        # self.Ax[0][i, j] += coefficient * inv_rho
                        self.Ax[0][i, j] = coefficient * inv_rho

                if not self.solver.is_colliding(i + 1, j):
                    inv_rho = self.volume_x[i + 1, j] / self.mass_x[i + 1, j]
                    self.Adiag[0][i, j] -= coefficient * inv_rho
                    if self.solver.is_interior(i + 1, j):
                        # self.Ax[0][i, j] += coefficient * inv_rho
                        self.Ax[0][i, j] = coefficient * inv_rho

                if not self.solver.is_colliding(i, j - 1):
                    inv_rho = self.volume_y[i, j] / self.mass_y[i, j]
                    self.Adiag[0][i, j] -= coefficient * inv_rho
                    if self.solver.is_interior(i, j - 1):
                        # self.Ay[0][i, j] += coefficient * inv_rho
                        self.Ay[0][i, j] = coefficient * inv_rho

                if not self.solver.is_colliding(i, j + 1):
                    inv_rho = self.volume_y[i, j + 1] / self.mass_y[i, j + 1]
                    self.Adiag[0][i, j] -= coefficient * inv_rho
                    if self.solver.is_interior(i, j + 1):
                        # self.Ay[0][i, j] += coefficient * inv_rho
                        self.Ay[0][i, j] = coefficient * inv_rho

            else:  # Homogeneous Dirichlet boundary condition.
                self.Adiag[0][i, j] += 1.0
                self.b[i, j] = 0.0

    @ti.kernel
    def gridtype_init(self, l: ti.template()):  # pyright: ignore
        for i, j in self.classifications[l]:
            # if i == 0 or i == self.n_grid // (2**l) - 1 or j == 0 or j == self.n_grid // (2 ** l) - 1:
            #     self.grid_type[l][i, j] = Classification.Colliding
            #     continue

            i2 = i * 2
            j2 = j * 2

            is_empty = self.classifications[l - 1][i2, j2] == Classification.Empty
            is_empty |= self.classifications[l - 1][i2, j2 + 1] == Classification.Empty
            is_empty |= self.classifications[l - 1][i2 + 1, j2] == Classification.Empty
            is_empty |= self.classifications[l - 1][i2 + 1, j2 + 1] == Classification.Empty
            if is_empty:
                self.classifications[l][i, j] = Classification.Empty
                continue

            is_interior = self.classifications[l - 1][i2, j2] == Classification.Interior
            is_interior |= self.classifications[l - 1][i2, j2 + 1] == Classification.Interior
            is_interior |= self.classifications[l - 1][i2 + 1, j2] == Classification.Interior
            is_interior |= self.classifications[l - 1][i2 + 1, j2 + 1] == Classification.Interior
            if is_interior:
                self.classifications[l][i, j] = Classification.Interior
                continue

            self.classifications[l][i, j] = Classification.Colliding

    @ti.kernel
    def preconditioner_init(self, l: ti.template()):  # pyright: ignore
        coefficient = self.dt * self.inv_dx * self.inv_dx / (2**l * 2**l)
        for i, j in self.classifications[l]:
            if self.classifications[l][i, j] == Classification.Interior:
                if self.classifications[l][i - 1, j] == Classification.Interior:
                    inv_rho = 1.0 # self.volume_x[i, j] / self.mass_x[i, j]
                    self.Adiag[l][i, j] -= coefficient * inv_rho
                    # self.Adiag[l][i, j] -= coefficient
                    # self.Adiag[l][i, j] += coefficient
                if self.classifications[l][i + 1, j] == Classification.Interior:
                    inv_rho = 1.0 # self.volume_x[i + 1, j] / self.mass_x[i + 1, j]
                    self.Adiag[l][i, j] -= coefficient * inv_rho
                    self.Ax[l][i, j] = coefficient * inv_rho
                    # self.Adiag[l][i, j] += coefficient
                    # self.Adiag[l][i, j] -= coefficient
                    # self.Ax[l][i, j] = -coefficient
                elif self.classifications[l][i + 1, j] == Classification.Empty:
                    inv_rho = 1.0 # self.volume_x[i + 1, j] / self.mass_x[i + 1, j]
                    self.Adiag[l][i, j] -= coefficient * inv_rho
                    # self.Adiag[l][i, j] += coefficient

                if self.classifications[l][i, j - 1] == Classification.Interior:
                    inv_rho = 1.0 # self.volume_y[i, j] / self.mass_y[i, j]
                    self.Adiag[l][i, j] -= coefficient * inv_rho
                    # self.Adiag[l][i, j] += coefficient
                if self.classifications[l][i, j + 1] == Classification.Interior:
                    inv_rho = 1.0 # self.volume_y[i, j + 1] / self.mass_y[i, j + 1]
                    self.Adiag[l][i, j] -= coefficient * inv_rho
                    self.Ay[l][i, j] = coefficient * inv_rho
                    # self.Ay[l][i, j] = -coefficient
                    # self.Adiag[l][i, j] += coefficient
                elif self.classifications[l][i, j + 1] == Classification.Empty:
                    inv_rho = 1.0 # self.volume_y[i, j + 1] / self.mass_y[i, j + 1]
                    self.Adiag[l][i, j] -= coefficient * inv_rho
                    # self.Adiag[l][i, j] += coefficient

                # if not self.solver.is_colliding(i - 1, j):
                #     inv_rho = 1.0  # self.volume_x[i, j] / self.mass_x[i, j]
                #     self.Adiag[l][i, j] -= coefficient * inv_rho
                #     # if self.solver.is_interior(i - 1, j):
                #     #     # self.Ax[0][i, j] += coefficient * inv_rho
                #     #     self.Ax[l][i, j] = coefficient * inv_rho

                # if not self.solver.is_colliding(i + 1, j):
                #     inv_rho = 1.0  # self.volume_x[i + 1, j] / self.mass_x[i + 1, j]
                #     self.Adiag[l][i, j] -= coefficient * inv_rho
                #     if self.solver.is_interior(i + 1, j):
                #         # self.Ax[0][i, j] += coefficient * inv_rho
                #         self.Ax[l][i, j] = coefficient * inv_rho

                # if not self.solver.is_colliding(i, j - 1):
                #     inv_rho = 1.0  # self.volume_y[i, j] / self.mass_y[i, j]
                #     self.Adiag[l][i, j] -= coefficient * inv_rho
                #     # if self.solver.is_interior(i, j - 1):
                #     #     # self.Ay[0][i, j] += coefficient * inv_rho
                #     #     self.Ay[l][i, j] = coefficient * inv_rho

                # if not self.solver.is_colliding(i, j + 1):
                #     inv_rho = 1.0  # self.volume_y[i, j + 1] / self.mass_y[i, j + 1]
                #     self.Adiag[l][i, j] -= coefficient * inv_rho
                #     if self.solver.is_interior(i, j + 1):
                #         # self.Ay[0][i, j] += coefficient * inv_rho
                #         self.Ay[l][i, j] = coefficient * inv_rho

    def initialize(self):
        self.b.fill(0.0)

        for l in range(self.n_grid_levels):
            self.Adiag[l].fill(0.0)
            self.Ax[l].fill(0.0)
            self.Ay[l].fill(0.0)

        self._initialize()
        self.classifications[0].copy_from(self.classification_c)

        for l in range(1, self.n_grid_levels):
            self.gridtype_init(l)
            self.preconditioner_init(l)

    @ti.func
    def neighbor_sum(self, Ax, Ay, z, nx, ny, i, j):
        Az = (
            Ax[(i - 1 + nx) % nx, j] * z[(i - 1 + nx) % nx, j]
            + Ax[i, j] * z[(i + 1) % nx, j]
            + Ay[i, (j - 1 + ny) % ny] * z[i, (j - 1 + ny) % ny]
            + Ay[i, j] * z[i, (j + 1) % ny]
        )

        return Az

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.i32):  # pyright: ignore
        # for i, j in self.r[l]:
        #     if (i + j) & 1 == phase:
        #         m, n = self.n_grid // (2**l), self.n_grid // (2**l)
        #         neighbors = self.neighbor_sum(self.Ax[l], self.Ay[l], self.z[l], m, n, i, j)
        #         self.z[l][i, j] = (self.r[l][i, j] + neighbors) / 4

        # phase: red/black Gauss-Seidel phase
        for i, j in self.r[l]:
            if self.classifications[l][i, j] == Classification.Interior and (i + j) & 1 == phase:
                # self.z[l][i, j] = (
                #     self.r[l][i, j]
                #     - self.neighbor_sum(self.Ax[l], self.Ay[l], self.z[l], self.m // (2**l), self.n // (2**l), i, j)
                # ) / self.Adiag[l][i, j]

                m, n = self.n_grid // (2**l), self.n_grid // (2**l)
                neighbors = self.neighbor_sum(self.Ax[l], self.Ay[l], self.z[l], m, n, i, j)
                self.z[l][i, j] += (self.r[l][i, j] - neighbors) / self.Adiag[l][i, j]
                # TODO: this was like this:
                # self.z[l][i, j] = (self.r[l][i, j] - neighbors) / self.Adiag[l][i, j]

    @ti.kernel
    def restrict(self, l: ti.template()):  # pyright: ignore
        for i, j in self.r[l]:
            if self.classifications[l][i, j] == Classification.Interior:
                m, n = self.n_grid // (2**l), self.n_grid // (2**l)
                Az = self.Adiag[l][i, j] * self.z[l][i, j]
                Az += self.neighbor_sum(self.Ax[l], self.Ay[l], self.z[l], m, n, i, j)
                res = self.r[l][i, j] - Az

                self.r[l + 1][i // 2, j // 2] += 0.25 * res

    @ti.kernel
    def prolongate(self, l: ti.template()):  # pyright: ignore
        for i, j in self.z[l]:
            self.z[l][i, j] += self.z[l + 1][i // 2, j // 2]

    def v_cycle(self):
        self.z[0].fill(0.0)
        for l in range(self.n_grid_levels - 1):
            for _ in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)

            self.r[l + 1].fill(0.0)
            self.z[l + 1].fill(0.0)
            self.restrict(l)

        # solve Az = r on the coarse grid
        for _ in range(self.bottom_smoothing):
            self.smooth(self.n_grid_levels - 1, 0)
            self.smooth(self.n_grid_levels - 1, 1)

        for l in reversed(range(self.n_grid_levels - 1)):
            self.prolongate(l)
            for _ in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self, max_iters=500):
        self.p.fill(0.0)
        self.As.fill(0.0)
        self.s.fill(0.0)
        self.r[0].copy_from(self.b)

        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.reduction[None]
        abs_tol = 1e-6
        rel_tol = 1e-6
        tol = max(abs_tol, initial_rTr * rel_tol)

        if initial_rTr < tol:
            return  # don't bother

        # print("init rTr = {}".format(init_rTr))

        # if init_rTr < tol:
        #     print("Converged: init rtr = {}".format(init_rTr))
        # else:

        # p0 = 0
        # r0 = b - Ap0 = b
        # z0 = M^-1r0
        # self.z.fill(0.0)
        self.v_cycle()

        # s0 = z0
        self.s.copy_from(self.z[0])

        # zTr
        self.reduce(self.z[0], self.r[0])
        old_zTr = self.reduction[None]

        for i in range(max_iters):
            # alpha = zTr / sAs
            self.compute_As()
            self.reduce(self.s, self.As)
            sAs = self.reduction[None]
            self.alpha[None] = old_zTr / (sAs + 1e-12)
            # self.alpha[None] = old_zTr / sAs

            # p = p + alpha * s
            self.update_p()

            # r = r - alpha * As
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.reduction[None]
            if rTr < tol:
                break

            # z = M^-1r
            self.v_cycle()

            self.reduce(self.z[0], self.r[0])
            new_zTr = self.reduction[None]

            # beta = zTrnew / zTrold
            # self.beta[None] = new_zTr / old_zTr
            self.beta[None] = new_zTr / (old_zTr + 1e-12)

            # s = z + beta * s
            self.update_s()
            old_zTr = new_zTr
            # iteration = i

            # if iteration % 100 == 0:
            #     print("iter {}, res = {}".format(iteration, rTr))

            # print("Converged to {} in {} iterations".format(rTr, iteration))
        # print("Pressure result: ")
        # print(self.pressure_c)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):  # pyright: ignore
        self.reduction[None] = 0.0
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            if self.classification_c[i, j] == Classification.Interior:
                self.reduction[None] += p[i, j] * q[i, j]

    @ti.kernel
    def compute_As(self):
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            if self.classification_c[i, j] == Classification.Interior:
                self.As[i, j] = (
                    self.Adiag[0][i, j] * self.s[i, j]
                    + self.Ax[0][i - 1, j] * self.s[i - 1, j]
                    + self.Ax[0][i, j] * self.s[i + 1, j]
                    + self.Ay[0][i, j - 1] * self.s[i, j - 1]
                    + self.Ay[0][i, j] * self.s[i, j + 1]
                )

    @ti.kernel
    def update_p(self):
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            if self.classification_c[i, j] == Classification.Interior:
                self.p[i, j] = self.p[i, j] + self.alpha[None] * self.s[i, j]

    @ti.kernel
    def update_r(self):
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            if self.classification_c[i, j] == Classification.Interior:
                self.r[0][i, j] = self.r[0][i, j] - self.alpha[None] * self.As[i, j]

    @ti.kernel
    def update_s(self):
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            if self.classification_c[i, j] == Classification.Interior:
                self.s[i, j] = self.z[0][i, j] + self.beta[None] * self.s[i, j]
