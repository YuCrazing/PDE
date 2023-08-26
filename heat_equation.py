# Solve the heat equation:
# \u_t - \laplacian u = 0, t > 0
# u = g,            t == 0
#
# Stability analysis of FTCS scheme:
# https://en.wikipedia.org/wiki/Von_Neumann_stability_analysis
# https://math.mit.edu/research/highschool/rsi/documents/2017Lee.pdf

import taichi as ti
from datetime import datetime
import numpy as np

ti.init(arch=ti.gpu, debug=True)

float_type = ti.f64
scene_length = 80.0
grid_res = (800, 800)
# dx = 1.0 / grid_res[0]
dx = scene_length / grid_res[0]
dt = 0.0003


assert dt < dx * dx / 4


u = ti.field(float_type, grid_res)
u_t = ti.field(float_type, grid_res)

use_exact = False
record_taichi = False
record_matplot = False


# g is the initial condition of u at t = 0. However, g is not C^1 here to make the clamping boundary condition easier to implement.
@ti.func
def g(spatial_pos):
    res = 0.0
    # case 1
    # # res = spatial_pos.x / scene_length

    # case 2
    # if spatial_pos.x <= scene_length / 2:
    #     res = 0.0
    # else:
    #     res = 1.0

    # case 3
    # # if spatial_pos.x <= 0.5*dx or spatial_pos.y <= 0.5*dx:
    # #     res = 1.0 

    # case 4
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


@ti.kernel
def exact(accumulated_time: float_type):
    pass

# Clamping boundary condition: clamp the value out of the boundary to the boundary
@ti.func
def sample(u: ti.template(), cid):
    cid = ti.max(0, ti.min(ti.Vector([grid_res[0], grid_res[1]]) - 1, cid))
    return u[cid]

@ti.func
def lerp(vl, vr, frac):
    return vl + frac * (vr - vl)

@ti.func
def bilerp(u: ti.template(), spatial_pos) -> float_type:
    grid_pos = spatial_pos/dx-0.5
    # floor: toward -inf
    # cast: toward zero
    # Here we use floor to ensure the returned value is the exact value at (0, 0) when pos=(0, 0)
    base_cid = ti.floor(grid_pos, ti.i32)
    frac = grid_pos - base_cid
    v00 = sample(u, ti.Vector([base_cid.x, base_cid.y]))
    v10 = sample(u, ti.Vector([base_cid.x+1, base_cid.y]))
    v01 = sample(u, ti.Vector([base_cid.x, base_cid.y+1]))
    v11 = sample(u, ti.Vector([base_cid.x+1, base_cid.y+1]))
    return lerp(lerp(v00, v10, frac.x), lerp(v01, v11, frac.x), frac.y)

@ti.kernel
def step(dt: float_type):
    for i, j in u:
        # \laplacian u = \frac{1}{dx^2} (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4 * u[i, j])
        if i > 0 and i < grid_res[0] - 1 and j > 0 and j < grid_res[1] - 1:
            laplacian_u = float_type(0.0)
            ur = bilerp(u, ti.Vector([(i+1+0.5)*dx, (j+0.5)*dx]))
            ul = bilerp(u, ti.Vector([(i-1+0.5)*dx, (j+0.5)*dx]))
            ut = bilerp(u, ti.Vector([(i+0.5)*dx, (j+1+0.5)*dx]))
            ub = bilerp(u, ti.Vector([(i+0.5)*dx, (j-1+0.5)*dx]))
            uc = bilerp(u, ti.Vector([(i+0.5)*dx, (j+0.5)*dx]))
            laplacian_u = (ur + ul + ut + ub - 4 * uc)/(dx*dx)

            # cnt = 0
            # if i > 0:
            #     laplacian_u += u[i-1, j]
            #     cnt += 1
            # if i < grid_res[0] - 1:
            #     laplacian_u += u[i+1, j]
            #     cnt += 1
            # if j > 0:
            #     laplacian_u += u[i, j-1]
            #     cnt += 1
            # if j < grid_res[1] - 1:
            #     laplacian_u += u[i, j+1]
            #     cnt += 1
            # laplacian_u = (laplacian_u - cnt * u[i, j]) / (dx * dx)
            u_t[i, j] = laplacian_u

    for i, j in u:
        if i > 0 and i < grid_res[0] - 1 and j > 0 and j < grid_res[1] - 1:
            u[i, j] += u_t[i, j] * dt


init_u()
gui = ti.GUI('Heat Equation', res=grid_res, background_color=0x0)

result_dir = "./result"
filename = datetime.now().strftime("video_%Y_%m_%d_%H_%M_%S") + "_" + ("exact" if use_exact else "numerical")
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False, video_filename=filename)


if record_matplot:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X = np.arange(0, grid_res[0]*dx, dx)
    Y = np.arange(0, grid_res[1]*dx, dx)
    X, Y = np.meshgrid(X, Y)

accumulated_time = 0.0
frame = 0
while gui.running and not gui.get_event(gui.ESCAPE):
    accumulated_time += dt*10
    print("accumulated time:", accumulated_time)
    
    if use_exact:
        exact(accumulated_time)
    else:
        for i in range(10):
            step(dt)

    gui.clear(0x0)
    gui.set_image(u.to_numpy()) # gui.set_image(u) not working occasionally
    
    if record_taichi:
        video_manager.write_frame(gui.get_image())
    
    gui.show()

    if record_matplot:
        ax.clear()
        ax.plot_surface(X, Y, u.to_numpy(), rstride=1, cstride=1, cmap='viridis')
        # plt.pause(0.01)
        plt.savefig(f'plots/frames/foo_{frame:06d}.png', dpi=300)

    frame += 1

    if accumulated_time > 1.0:
        break

if record_matplot:
    # plt.show()
    import os
    os.system(f"cd plots && ffmpeg -framerate 30 -pattern_type glob -i 'frames/*.png'  -c:v libx264 -pix_fmt yuv420p {filename}_plot.mp4")
if record_taichi:
    video_manager.make_video(gif=True, mp4=True)
