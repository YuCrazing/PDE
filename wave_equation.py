# Solve the wave equation:
# \u_t - \nabla u = 0, t > 0
# u = g,            t == 0


import taichi as ti
from datetime import datetime

ti.init(arch=ti.gpu, debug=True)

float_type = ti.f64
scene_length = 80.0
grid_res = (80, 80)
# dx = 1.0 / grid_res[0]
dx = scene_length / grid_res[0]
dt = 0.001

b = ti.Vector([1.0, 1.0])
u = ti.field(float_type, grid_res)
prev_u = ti.field(float_type, grid_res)
u_tt = ti.field(float_type, grid_res)

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
    center = ti.Vector([scene_length / 2, scene_length / 2])
    if (spatial_pos - center).norm() <= scene_length / 8:
        res = 1.0
    else:
        res = 0.0
    return res

@ti.kernel
def init_u():
    for i, j in u:
        u[i, j] = g(ti.Vector([(i+0.5)*dx, (j+0.5)*dx]))
        prev_u[i, j] = u[i, j]
        u_tt[i, j] = 0.0


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

        u_tt[i, j] = nabla_u

    u_mx = float_type(0.0)
    for i, j in u:
        new_u = 2*u[i, j] - prev_u[i, j] + u_tt[i, j] * dt * dt
        prev_u[i, j] = u[i, j]
        u[i, j] = new_u
        ti.atomic_max(u_mx, abs(new_u))
    if u_mx > 1.0:
        print("u_mx:", u_mx)


init_u()
gui = ti.GUI('Wave Equation', res=grid_res, background_color=0x0)

result_dir = "./result"
filename = datetime.now().strftime("video_%Y_%m_%d_%H_%M_%S") + "_" + ("exact" if use_exact else "numerical")
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False, video_filename=filename)


import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X = np.arange(0, grid_res[0]*dx, dx)
Y = np.arange(0, grid_res[1]*dx, dx)
X, Y = np.meshgrid(X, Y)

accumulated_time = 0.0
frame = 0
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

    ax.clear()
    ax.plot_surface(X, Y, u.to_numpy(), rstride=1, cstride=1, cmap='viridis')
    # plt.pause(0.01)
    plt.savefig(f'plots/frames/foo_{frame:06d}.png', dpi=300)

    frame += 1

    if accumulated_time > 40.0:
        break

# plt.show()
import os
os.system(f"cd plots && ffmpeg -framerate 30 -pattern_type glob -i 'frames/*.png'  -c:v libx264 -pix_fmt yuv420p {filename}_plot.mp4")
if record_video:
    video_manager.make_video(gif=True, mp4=True)
