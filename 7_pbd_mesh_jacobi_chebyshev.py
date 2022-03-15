import taichi as ti
import numpy as np 

ti.init(arch=ti.gpu)

N = 5
NV = (N+1)**2
NE = (N+1) * N * 2
positions = ti.Vector.field(2, ti.f32, NV)
old_positions = ti.Vector.field(2, ti.f32, NV)
pre_positions = ti.Vector.field(2, ti.f32, NV)
edge_indices = ti.Vector.field(2, ti.i32, NE)

inv_mass =ti.field(ti.f32, NV)
velocities = ti.Vector.field(2, ti.f32, NV)

rest_len = ti.field(ti.f32, NE)

constraint = ti.field(ti.f32, NE)
gradient = ti.Vector.field(2, ti.f32, 2 * NE)

rho = 0.0

@ti.kernel 
def init_pos():
    step = 1/N * 0.5
    for i, j in ti.ndrange(N+1, N+1):
        positions[i * (N+1) + j]  = ti.Vector([i, j]) * step  + ti.Vector([0.25,0.25])
    for i in range(NV):
        old_positions[i] = positions[i]
        inv_mass[i] = 1.0
    inv_mass[N] = 0.0
    inv_mass[NV-1] = 0.0

@ti.kernel 
def init_edge():
    for i, j in ti.ndrange(N+1, N):
        a = i * (N + 1) + j
        edge_indices[i * N + j] = ti.Vector([a, a+1])
    start = N * (N + 1)
    for i, j in ti.ndrange(N, N + 1):
        a = i * (N + 1) + j
        edge_indices[start + i + j * N] = ti.Vector([a, a + N + 1])

@ti.kernel 
def init_rest_len():
    for i in range(NE):
        idx0, idx1  = edge_indices[i] 
        rest_len[i] = (positions[idx0] - positions[idx1]).norm()

@ti.kernel 
def semi_euler(h: ti.f32):
    gravity = ti.Vector([0.0, -0.8])
    for i in range(NV):
        if inv_mass[i] != 0.0:
            velocities[i] += h * gravity
            old_positions[i] = positions[i]
            positions[i] += h * velocities[i]

@ti.kernel
def compute_constraint_gradient() -> ti.f32:
    dual_residual = 0.0
    for i in range(NE):
        idx0, idx1  = edge_indices[i] 
        dis = positions[idx0] - positions[idx1]
        constraint[i] = dis.norm() - rest_len[i]
        dual_residual += constraint[i]**2
        gradient[2 * i + 0] = dis.normalized()
        gradient[2 * i + 1] = -dis.normalized()
    return ti.sqrt(dual_residual)

@ti.kernel 
def solve_constraints():
    for i in range(NE):
        idx0, idx1  = edge_indices[i] 
        invM0, invM1 = inv_mass[idx0], inv_mass[idx1]
        l = -constraint[i] / (invM0 + invM1)
        if invM0 != 0.0:
            positions[idx0] += invM0 * l * gradient[2 * i + 0]
        if invM1 != 0.0:
            positions[idx1] += invM1 * l * gradient[2 * i + 1]

@ti.kernel
def update_v(h: ti.f32):
    for i in range(NV):
        if inv_mass[i] != 0.0:
            velocities[i] = (positions[i] - old_positions[i])/h

@ti.kernel 
def collision():
    for i in range(NV):
        if positions[i][1] < 0.0:
            positions[i][1] = 0.0

@ti.kernel
def copy_positions():
    for i in range(NV):
        pre_positions[i] = positions[i]

@ti.kernel
def apply_primal_chebyshev(omega: ti.f32):
    for i in range(NV):
        positions[i] *= omega
        positions[i] += (1 - omega) * pre_positions[i]

def apply_chebyshev(ite):
    omega = 1
    if ite <= 10:
        omega = 1.0
    elif ite == 11:
        omega = 2/(2-rho * rho)
    else:
        omega = 4/(4-rho*rho*omega)
    apply_primal_chebyshev(omega)
    return omega 

def update(h, maxIte, use_primal_chebyshev):
    semi_euler(h)
    dual_residual = [None] * maxIte
    for ite in range(maxIte):
        copy_positions()
        dual_residual[ite] = compute_constraint_gradient()
        solve_constraints()
        if use_primal_chebyshev:
            apply_chebyshev(ite)
        collision()
    update_v(h)
    return dual_residual

init_pos()
init_edge()
init_rest_len()
gui = ti.GUI("Diplay tri mesh", res=(600,600))
pause = False
h = 0.01
maxIte = 200

use_primal_chebyshev = False
dual_residual_file = "data/dual_residual.txt"
if use_primal_chebyshev:
    dual_residual_file = "data/chebyshev_dual_residual.txt"
residual_file = open(dual_residual_file, "w")

frame = 0
while gui.running:
    gui.get_event(ti.GUI.PRESS)
    if gui.is_pressed(ti.GUI.ESCAPE):
        gui.running = False
    elif gui.is_pressed(ti.GUI.SPACE):
        pause = not pause
    
    if not pause:
        dual_residual = update(h, maxIte, use_primal_chebyshev)
        for i in range(len(dual_residual)):
                residual_file.write(f"{dual_residual[i]} \n")

    poses = positions.to_numpy()
    edges = edge_indices.to_numpy()
    begin_line, end_line = [], []
    for i in range(edges.shape[0]):
        idx0, idx1 = edges[i]
        begin_line.append(poses[idx0])
        end_line.append(poses[idx1])
    gui.lines(np.asarray(begin_line), np.asarray(end_line), radius=2, color=0x0000FF) 
    gui.circles(positions.to_numpy(), radius=6, color=0xffaa33)
    static_points = np.array([poses[N], poses[NV-1]])
    gui.circles(static_points, radius=7, color=0xff0000)
    gui.show()
    frame += 1
    # if frame == 5:
    #     gui.running = False

residual_file.close()