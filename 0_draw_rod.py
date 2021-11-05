import taichi as ti

ti.init(arch=ti.cpu)

n = 5
pos = ti.Vector.field(n=2, dtype=ti.f32, shape=n)
edge = ti.Vector.field(n=2, dtype=ti.i32, shape=n - 1)


@ti.kernel
def init_pos():
    for i in pos:
        pos[i] = ti.Vector([i * 0.1, 0]) + ti.Vector([0.4, 0.5])
    for i in edge:
        edge[i] = ti.Vector([i, i + 1])


gui = ti.GUI("Display Rod", res=(500, 500))
init_pos()
while gui.running:
    positions = pos.to_numpy()
    begin_points = positions[:-1]
    end_points = positions[1:]
    gui.lines(begin_points, end_points, radius=4, color=0x00FF00)
    gui.circles(pos.to_numpy(), radius=5, color=0xFF0000)
    gui.show()