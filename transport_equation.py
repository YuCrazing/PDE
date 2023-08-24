# Solve the transport equation:
# u_t + b * Du = 0, t > 0
# u = g,            t == 0


import taichi as ti
from datetime import datetime

ti.init(arch=ti.gpu, debug=True)

float_type = ti.f64
grid_res = (80, 80)
# dx = 1.0 / grid_res[0]
dx = 1.0
dt = 0.1

b = ti.Vector([1.0, 1.0])
u = ti.field(float_type, grid_res)
u_temp = ti.field(float_type, grid_res)
u_t = ti.Vector.field(2, float_type, grid_res)

use_RK = 3
use_exact = True
record_video = False


# g is the initial condition of u at t = 0. However, g is not C^1 here to make the clamping boundary condition easier to implement.
@ti.func
def g(spatial_pos):
    res = 0.0
    if spatial_pos.x <= 0.5*dx or spatial_pos.y <= 0.5*dx:
        res = 1.0 
    return res

@ti.kernel
def init_boundary_condition():
    for i, j in u:
        u[i, j] = g(ti.Vector([(i+0.5)*dx, (j+0.5)*dx]))


@ti.kernel
def exact(accumulated_time: float_type):
    for i, j in u:
        u[i, j] = g(ti.Vector([(i+0.5)*dx, (j+0.5)*dx]) - b * accumulated_time)

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


@ti.func
def backtrace(u: ti.template(),  u_t: ti.template(), spatial_pos, dt):
    res = float_type(0.0)
    if ti.static(use_RK == 1):
        prev_spatial_pos = spatial_pos - bilerp(u_t, spatial_pos) * dt
        res = bilerp(u, prev_spatial_pos)
    elif ti.static(use_RK == 3):
        p = spatial_pos
        v1 = bilerp(u_t, p)
        p1 = p - 0.5 * dt * v1
        v2 = bilerp(u_t, p1)
        p2 = p - 0.75 * dt * v2
        v3 = bilerp(u_t, p2)
        p -= dt * ((2.0 / 9) * v1 + (1.0 / 3) * v2 + (4.0 / 9) * v3)
        res = bilerp(u, p)
    return res


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
        u_t[i, j] = - b * Du
    
    for i, j in u_temp:
        u_temp[i, j] = backtrace(u, u_t, ti.Vector([(i+0.5)*dx, (j+0.5)*dx], dt=float_type), dt)

    for i, j in u:
        u[i, j] = u_temp[i, j]


init_boundary_condition()
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
