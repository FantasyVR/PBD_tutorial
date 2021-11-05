import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

n = 5
pos = ti.Vector.field(n=2, dtype=ti.f32, shape=n)
edge = ti.Vector.field(n=2, dtype=ti.i32, shape=n - 1)
inv_mass = ti.field(dtype=ti.f32, shape=n)
vel = ti.Vector.field(n=2, dtype=ti.f32, shape=n)
h = 0.001  # time step size: 10ms
pause = True


@ti.kernel
def init_pos():
    for i in pos:
        pos[i] = ti.Vector([i * 0.1, 0]) + ti.Vector([0.4, 0.5])
    for i in edge:
        edge[i] = ti.Vector([i, i + 1])
    for i in range(n - 1):
        inv_mass[i + 1] = 1.0


@ti.kernel
def seme_euler(h: ti.f32):
    gravity = ti.Vector([0.0, -9.8])
    for i in range(n):
        if inv_mass[i] != 0.0:
            vel[i] += h * gravity
            pos[i] += vel[i] * h


gui = ti.GUI("Display Rod", res=(500, 500))
init_pos()
while gui.running:

    gui.get_event(ti.GUI.PRESS)
    if gui.is_pressed(ti.GUI.ESCAPE):
        gui.running = False
    elif gui.is_pressed(ti.GUI.SPACE):
        pause = not pause

    if not pause:
        seme_euler(h)

    positions = pos.to_numpy()
    begin_points = positions[:-1]
    end_points = positions[1:]
    gui.lines(begin_points, end_points, radius=4, color=0x00FF00)
    gui.circles(pos.to_numpy(), radius=5, color=0xFF0000)
    gui.show()
