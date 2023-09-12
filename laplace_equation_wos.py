# Solve the laplace equation:
# \laplacian u(x) = 0, x \in U
# u(r, t) = f1, r = R1,
# u(r, t) = f2, r = R2

# Some examples:
# R1 = 3, R2 = 6, f1 = 5cos(t), f2 = 10sin(t)
# https://www.math.uh.edu/~pwalker/laplace6.pdf
# R1 = 1, R2 = 2, f1 = 0, f2 = 10sin(t)
# https://www.math.usm.edu/lambers/mat417/class0425.pdf


import taichi as ti
from datetime import datetime
import math

ti.init(arch=ti.gpu, debug=False)

float_type = ti.f64
# grid_res = (80, 80)
grid_res = (800, 800)
u = ti.field(float_type, grid_res)


use_case = 2
use_exact = False
record_taichi = False
record_matplot = False

scene_length = 0.0
dx = 0.0
if use_case == 1:
    scene_length = 14.0
    dx = scene_length / grid_res[0]
elif use_case == 2:
    scene_length = 6.0
    dx = scene_length / grid_res[0]
    
center = ti.Vector([scene_length/2, scene_length/2])

threshold = 0.0001


R1 = R2 = 0
if use_case == 1:
    R1 = 3
    R2 = 6
elif use_case == 2:
    R1 = 1
    R2 = 2

@ti.func
def inU(spatial_pos):
    p = spatial_pos - center
    r = p.norm()
    return r > R1 and r < R2

@ti.func
def inner_R1(spatial_pos):
    p = spatial_pos - center
    r = p.norm()
    return r <= R1

@ti.func
def outer_R2(spatial_pos):
    p = spatial_pos - center
    r = p.norm()
    return r >= R2

# Boundary condition
@ti.func
def g(spatial_pos):
    res = 0.0
    p = spatial_pos - center
    r = p.norm()
    sin_t = p.y / r
    cos_t = p.x / r

    if outer_R2(spatial_pos):
        if use_case == 1:
            res = 10*sin_t
        elif use_case == 2:
            res = sin_t
    elif inner_R1(spatial_pos):
        if use_case == 1:
            res = 5*cos_t
        elif use_case == 2:
            res = 0.0
    return res


# Projecting points near the boundary to the boundary (for WoS)
@ti.func
def g_prjection(spatial_pos):
    res = float_type(0.0)
    p = spatial_pos - center
    r = p.norm()
    sin_t = p.y / r
    cos_t = p.x / r

    if ti.abs(r-R2) < threshold:
        if use_case == 1:
            res = 10*sin_t
        elif use_case == 2:
            res = sin_t
    elif ti.abs(r-R1) < threshold:
        if use_case == 1:
            res = 5*cos_t
        elif use_case == 2:
            res = 0.0
    return res


@ti.kernel
def init_u():
    for i, j in u:
        u[i, j] = g(ti.Vector([(i+0.5)*dx, (j+0.5)*dx]))


@ti.kernel
def exact():
    for i, j in u:
        spatial_pos = ti.Vector([(i+0.5)*dx, (j+0.5)*dx])
        if inU(spatial_pos):
            p = spatial_pos - center
            r = p.norm()
            sin_t = p.y / r
            cos_t = p.x / r
            if use_case == 1:
                u[i, j] = (-5*r/9+20/r)*cos_t + (20*r/9-20/r)*sin_t
            else:
                u[i, j] = 2/3*(r-1/r)*sin_t


@ti.func
def largest_radius(spatial_pos):
    r = (spatial_pos - center).norm()
    return ti.min(ti.abs(R1-r), ti.abs(R2-r))

@ti.func
def sample_dir():
    theta = ti.random()*2*math.pi
    return ti.Vector([ti.cos(theta), ti.sin(theta)], float_type)

@ti.kernel
def wos_one_sample(u: ti.template(), current_samples: ti.i32):
    for i, j in u:
        spatial_pos = ti.Vector([(i+0.5)*dx, (j+0.5)*dx], float_type)
        if inU(spatial_pos):
            r = largest_radius(spatial_pos)
            if r < threshold:
                u[i, j] = g_prjection(spatial_pos)
            else:
                res = float_type(0.0)
                p = spatial_pos
                while r >= threshold:
                    dir = sample_dir()
                    p = p + dir * r
                    r = largest_radius(p)
                res = g_prjection(p)
                u[i, j] = res * (1.0 / current_samples) + u[i, j] * ((current_samples - 1.0) / current_samples)

init_u()
gui = ti.GUI('Laplace Equation (WoS)', res=grid_res, background_color=0x0)

result_dir = "./result"
filename = datetime.now().strftime("video_%Y_%m_%d_%H_%M_%S") + "_" + ("exact" if use_exact else "numerical")
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False, video_filename=filename)


if record_matplot:
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=24, azim=-109)

    X = np.arange(0, grid_res[0]*dx, dx)
    Y = np.arange(0, grid_res[1]*dx, dx)
    X, Y = np.meshgrid(X, Y)



frame = 0
while gui.running and not gui.get_event(gui.ESCAPE):
    
    if use_exact:
        exact()
    else:
        wos_one_sample(u, frame+1)
    
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
    print("frame:", frame)
    if frame >= 3000:
        break
if record_matplot:
    # plt.show()
    import os
    os.system(f"cd plots && ffmpeg -framerate 30 -pattern_type glob -i 'frames/*.png'  -c:v libx264 -pix_fmt yuv420p {filename}_plot.mp4")
if record_taichi:
    video_manager.make_video(gif=True, mp4=True)
