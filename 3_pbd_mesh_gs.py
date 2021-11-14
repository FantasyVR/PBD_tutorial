import taichi as ti
import numpy as np 

ti.init(arch=ti.cpu)

N = 5
NV = (N+1)**2
NE = (N+1) * N * 2

positions = ti.Vector.field(2, ti.f32, NV)

@ti.kernel 
def init_pos():
    step = 1/N * 0.5
    for i, j in ti.ndrange(N+1, N+1):
        positions[i * (N+1) + j]  = ti.Vector([i, j]) * step  + ti.Vector([0.25,0.25])

init_pos()
gui = ti.GUI("Diplay tri mesh", res=(600,600))
while gui.running:
    gui.get_event(ti.GUI.PRESS)
    if gui.is_pressed(ti.GUI.ESCAPE):
        gui.running = False
        
    gui.circles(positions.to_numpy(), radius=5, color=0xFF000)
    gui.show()