# Solve the wave equation:
# \u_t + \nabla u = 0, t > 0
# u = g,            t == 0


import taichi as ti
from datetime import datetime

ti.init(arch=ti.gpu, debug=True)

float_type = ti.f64
grid_res = (800, 800)
# dx = 1.0 / grid_res[0]
dx = 1.0/10
dt = 0.0001

b = ti.Vector([1.0, 1.0])
u = ti.field(float_type, grid_res)
u_t = ti.field(float_type, grid_res)

use_exact = False
record_video = False


# g is the initial condition of u at t = 0. However, g is not C^1 here to make the clamping boundary condition easier to implement.
@ti.func
def g(spatial_pos):
    res = 0.0
    # res = spatial_pos.x / 80
    if spatial_pos.x <= 40:
        res = 0.0
    else:
        res = 1.0
    # if spatial_pos.x <= 0.5*dx or spatial_pos.y <= 0.5*dx:
    #     res = 1.0 
    return res

@ti.kernel
def init_u():
    for i, j in u:
        u[i, j] = g(ti.Vector([(i+0.5)*dx, (j+0.5)*dx]))
        u_t[i, j] = 0.0


@ti.kernel
def exact(accumulated_time: float_type):
    pass
    # for i, j in u:
    #     u[i, j] = g(ti.Vector([(i+0.5)*dx, (j+0.5)*dx]) - b * accumulated_time)


@ti.kernel
def step(dt: float_type):
    for i, j in u:
        # \nabla u = \frac{1}{dx^2} (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4 * u[i, j])
        nabla_u = float_type(0.0)
        ul, ur, ut, ub = float_type(0.0), float_type(0.0), float_type(0.0), float_type(0.0)

        if i > 0:
            ul = u[i-1, j]
        else:
            ul = u[i, j] # TODO
        if i < grid_res[0] - 1:
            ur = u[i+1, j]
        else:
            ur = u[i, j]
        if j > 0:
            ut = u[i, j-1]
        else:
            ut = u[i, j]
        if j < grid_res[1] - 1:
            ub = u[i, j+1]
        else:
            ub = u[i, j]
        nabla_u = (ur + ul + ub + ut - 4 * u[i, j]) / (dx * dx)

        u_tt = nabla_u
        u_t[i, j] += u_tt * dt
    
    for i, j in u:
        u[i, j] += u_t[i, j] * dt


init_u()
gui = ti.GUI('Wave Equation', res=grid_res, background_color=0x0)

result_dir = "./result"
filename = datetime.now().strftime("video_%Y_%m_%d_%H_%M_%S") + "_" + ("exact" if use_exact else "numerical")
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False, video_filename=filename)

accumulated_time = 0.0
while gui.running and not gui.get_event(gui.ESCAPE):
    accumulated_time += dt
    print("accumulated time:", accumulated_time)
    
    if use_exact:
        exact(accumulated_time)
    else:
        for i in range(100):
            step(dt)
    
    gui.clear(0x0)
    gui.set_image(u)
    
    if record_video:
        video_manager.write_frame(gui.get_image())
    gui.show()

    if accumulated_time > 40.0:
        break

if record_video:
    video_manager.make_video(gif=True, mp4=True)
