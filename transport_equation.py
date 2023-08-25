# Solve the transport equation:
# u_t + b * Du = 0, t > 0
# u = g,            t == 0


import taichi as ti
from datetime import datetime

ti.init(arch=ti.gpu, debug=True)

float_type = ti.f64
grid_res = (800, 800)
# dx = 1.0 / grid_res[0]
dx = 1.0/10
dt = 0.01

b = ti.Vector([1.0, 1.0])
u = ti.field(float_type, grid_res)

use_exact = False
record_video = False


# g is the initial condition of u at t = 0. However, g is not C^1 here to make the clamping boundary condition easier to implement.
@ti.func
def g(spatial_pos):
    res = 0.0
    # if spatial_pos.x <= 0.5*dx or spatial_pos.y <= 0.5*dx:
    #     res = 1.0 
    if spatial_pos.x <= 40:
        res = 0.0
    else:
        res = 1.0
    return res

@ti.kernel
def init_u():
    for i, j in u:
        u[i, j] = g(ti.Vector([(i+0.5)*dx, (j+0.5)*dx]))


@ti.kernel
def exact(accumulated_time: float_type):
    for i, j in u:
        u[i, j] = g(ti.Vector([(i+0.5)*dx, (j+0.5)*dx]) - b * accumulated_time)


@ti.kernel
def step(dt: float_type):
    for i, j in u:
        Du = ti.Vector([0.0, 0.0], dt=float_type)
        if i > 0 and i < grid_res[0] - 1:
            Du.x = (u[i+1, j] - u[i - 1, j]) /(2*dx)
        elif i == 0:
            Du.x = (u[i + 1, j] - u[i, j]) / dx
        else:
            Du.x = (u[i, j] - u[i - 1, j]) / dx

        if j > 0 and j < grid_res[1] - 1:
            Du.y = (u[i, j+1] - u[i, j - 1]) / (2*dx)
        elif j == 0:
            Du.y = (u[i, j + 1] - u[i, j]) / dx
        else:
            Du.y = (u[i, j] - u[i, j - 1]) / dx
        u_t = - b.dot(Du)
        u[i, j] += u_t * dt


init_u()
gui = ti.GUI('Transport Equation', res=grid_res, background_color=0x0)

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
