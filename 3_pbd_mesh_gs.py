import taichi as ti
import numpy as np 

ti.init(arch=ti.cpu)

N = 5
NV = (N+1)**2
NE = (N+1) * N * 2

positions = ti.Vector.field(2, ti.f32, NV)
edge_idices = ti.Vector.field(2, ti.i32, NE)

@ti.kernel 
def init_pos():
    step = 1/N * 0.5
    for i, j in ti.ndrange(N+1, N+1):
        positions[i * (N+1) + j]  = ti.Vector([i, j]) * step  + ti.Vector([0.25,0.25])

@ti.kernel 
def init_edge():
    for i, j in ti.ndrange(N+1, N):
        a = i * (N + 1) + j
        edge_idices[i * N + j] = ti.Vector([a, a+1])
    start = N * (N + 1)
    for i, j in ti.ndrange(N, N + 1):
        a = i * (N + 1) + j
        edge_idices[start + i + j * N] = ti.Vector([a, a + N + 1])

init_pos()
init_edge()
gui = ti.GUI("Diplay tri mesh", res=(600,600))
while gui.running:
    gui.get_event(ti.GUI.PRESS)
    if gui.is_pressed(ti.GUI.ESCAPE):
        gui.running = False

    poses = positions.to_numpy()
    edges = edge_idices.to_numpy()
    begin_line, end_line = [], []
    for i in range(edges.shape[0]):
        idx1, idx2 = edges[i]
        begin_line.append(poses[idx1])
        end_line.append(poses[idx2])
    gui.lines(np.asarray(begin_line), np.asarray(end_line), radius=2, color=0x0000FF) 
    gui.circles(positions.to_numpy(), radius=6, color=0xffaa33)
    gui.show()