import taichi as ti
import numpy as np 

ti.init(arch=ti.cpu)

N = 5
NV = (N+1)**2
NE = (N+1) * N * 2

positions = ti.Vector.field(2, ti.f32, NV)
edge_indices = ti.Vector.field(2, ti.i32, NE)

inv_mass =ti.field(ti.f32, NV)
velocities = ti.Vector.field(2, ti.f32, NV)

@ti.kernel 
def init_pos():
    step = 1/N * 0.5
    for i, j in ti.ndrange(N+1, N+1):
        positions[i * (N+1) + j]  = ti.Vector([i, j]) * step  + ti.Vector([0.25,0.25])

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
def semi_euler(h: ti.f32):
    gravity = ti.Vector([0.0, -0.8])
    for i in range(NV):
        velocities[i] += h * gravity
        positions[i] += h * velocities[i]

@ti.kernel 
def collision():
    for i in range(NV):
        if positions[i][1] < 0.0:
            positions[i][1] = 0.0

def update(h):
    semi_euler(h)
    collision()

init_pos()
init_edge()
gui = ti.GUI("Diplay tri mesh", res=(600,600))
pause = False
h = 0.01
while gui.running:
    gui.get_event(ti.GUI.PRESS)
    if gui.is_pressed(ti.GUI.ESCAPE):
        gui.running = False
    elif gui.is_pressed(ti.GUI.SPACE):
        pause = not pause
    
    if not pause:
        update(h)

    poses = positions.to_numpy()
    edges = edge_indices.to_numpy()
    begin_line, end_line = [], []
    for i in range(edges.shape[0]):
        idx1, idx2 = edges[i]
        begin_line.append(poses[idx1])
        end_line.append(poses[idx2])
    gui.lines(np.asarray(begin_line), np.asarray(end_line), radius=2, color=0x0000FF) 
    gui.circles(positions.to_numpy(), radius=6, color=0xffaa33)
    gui.show()