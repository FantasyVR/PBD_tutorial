"""
Jacobi solver
"""
import taichi as ti

ti.init(arch=ti.gpu)

n = 10
pos = ti.Vector.field(n=2, dtype=ti.f32, shape=n)
old_pos = ti.Vector.field(n=2, dtype=ti.f32, shape=n)
edge = ti.Vector.field(n=2, dtype=ti.i32, shape=n - 1)
rest_len = ti.field(dtype=ti.f32, shape=n - 1)
inv_mass = ti.field(dtype=ti.f32, shape=n)
vel = ti.Vector.field(n=2, dtype=ti.f32, shape=n)
h, MaxIte = 0.01, 100  # time step size: 10ms, Maximu iteration number
pause = True
gradient = ti.Vector.field(n=2, dtype=ti.f32, shape=n - 1)
constraint = ti.field(ti.f32, shape=n - 1)


@ti.kernel
def init_pos():
    for i in pos:
        pos[i] = ti.Vector([i * 0.1, 0]) + ti.Vector([0.4, 0.5])
    for i in edge:
        edge[i] = ti.Vector([i, i + 1])
    for i in range(n):
        inv_mass[i] = 1.0
    inv_mass[0] = 0.0


@ti.kernel
def init_constrint():
    for i in edge:
        idx0, idx1 = edge[i]
        rest_len[i] = (pos[idx0] - pos[idx1]).norm()


@ti.kernel
def seme_euler(h: ti.f32):
    gravity = ti.Vector([0.0, -9.8])
    for i in range(n):
        if inv_mass[i] != 0.0:
            vel[i] += h * gravity
            old_pos[i] = pos[i]
            pos[i] += vel[i] * h


@ti.kernel
def compute_gradient_constraint() -> ti.f32:
    dual_residual = 0.0
    for i in range(n - 1):
        idx0, idx1 = edge[i]
        dis = pos[idx0] - pos[idx1]
        constraint[i] = dis.norm() - rest_len[i]
        gradient[i] = dis.normalized()
        dual_residual += constraint[i]**2
    return dual_residual


@ti.kernel
def solve_constraints():
    for i in range(n - 1):
        idx0, idx1 = edge[i]
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        l = -constraint[i] / (invM0 + invM1)
        if invM0 != 0.0:
            pos[idx0] += 0.8 * invM0 * l * gradient[i]
        if invM1 != 0.0:
            pos[idx1] -= 0.8 * invM1 * l * gradient[i]





def solve():
    dual_residual = compute_gradient_constraint()
    solve_constraints()
    return dual_residual


@ti.kernel
def update_vel(h: ti.f32):
    for i in range(n):
        if inv_mass[i] != 0.0:
            vel[i] = (pos[i] - old_pos[i]) / h


def update(h):
    seme_euler(h)
    f = open(f"data/Jacobi.txt", 'a')
    for i in range(MaxIte):
        dual_residual = solve()
        f.write(f"{dual_residual}  \n")
    update_vel(h)


gui = ti.GUI("Display Rod", res=(500, 500))
init_pos()
init_constrint()
while gui.running:

    gui.get_event(ti.GUI.PRESS)
    if gui.is_pressed(ti.GUI.ESCAPE):
        gui.running = False
    elif gui.is_pressed(ti.GUI.SPACE):
        pause = not pause

    if not pause:
        update(h)

    positions = pos.to_numpy()
    begin_points = positions[:-1]
    end_points = positions[1:]
    gui.lines(begin_points, end_points, radius=4, color=0x00FF00)
    gui.circles(pos.to_numpy(), radius=5, color=0xFF0000)
    gui.show()
