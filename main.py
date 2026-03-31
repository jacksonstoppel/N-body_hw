import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def make_orbit_movie(x_vals, y_vals, t_vals, filename="earth_orbit.gif", fps=60, skip=10, euler_x_vals=None, euler_y_vals=None, euler_t_vals=None, 
                     vel_x = None, vel_y = None):
    # Main trajectory
    x_vals = np.asarray(x_vals)[::skip]
    y_vals = np.asarray(y_vals)[::skip]
    t_vals = np.asarray(t_vals)[::skip]


    # Check for velocity

    if vel_x is not None and vel_y is not None:
        vel_x = np.asarray(vel_x)[::skip]
        vel_y = np.asarray(vel_y)[::skip]
        speed_vals = np.sqrt(vel_x**2 + vel_y**2)
    else:
        speed_vals = None

    # Check whether Euler data was provided
    include_euler = (euler_x_vals is not None and euler_y_vals is not None and euler_t_vals is not None)

    if include_euler:
        euler_x_vals = np.asarray(euler_x_vals)[::skip]
        euler_y_vals = np.asarray(euler_y_vals)[::skip]
        euler_t_vals = np.asarray(euler_t_vals)[::skip]

        # Make sure both animations have the same number of frames
        n_frames = min(len(x_vals), len(y_vals), len(t_vals),
                       len(euler_x_vals), len(euler_y_vals), len(euler_t_vals))

        x_vals = x_vals[:n_frames]
        y_vals = y_vals[:n_frames]
        t_vals = t_vals[:n_frames]
        euler_x_vals = euler_x_vals[:n_frames]
        euler_y_vals = euler_y_vals[:n_frames]
        euler_t_vals = euler_t_vals[:n_frames]

        all_x = np.concatenate([x_vals, euler_x_vals])
        all_y = np.concatenate([y_vals, euler_y_vals])
    else:
        n_frames = min(len(x_vals), len(y_vals), len(t_vals))
        x_vals = x_vals[:n_frames]
        y_vals = y_vals[:n_frames]
        t_vals = t_vals[:n_frames]

        all_x = x_vals
        all_y = y_vals

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    x_pad = 0.1 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.1 * (y_max - y_min if y_max > y_min else 1.0)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_title("Voyager Trajectory")

    ax.plot(0, 0, 'yo', markersize=10, label="Sun")

    # Main orbit of leapfrog
    orbit_line, = ax.plot([], [], '-', lw=1.5, label="Jupiter")
    earth_point, = ax.plot([], [], 'o', markersize=8)

    # optional plot of euler/voyager
    if include_euler:
        euler_line, = ax.plot([], [], '--', lw=1.5, label="Voyager")
        euler_point, = ax.plot([], [], 's', markersize=6)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    vel_text  = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def init():
        orbit_line.set_data([], [])
        earth_point.set_data([], [])

        if include_euler:
            euler_line.set_data([], [])
            euler_point.set_data([], [])

        time_text.set_text("")
        vel_text.set_text("")

        if include_euler:
            return orbit_line, earth_point, euler_line, euler_point, time_text
        else:
            return orbit_line, earth_point, time_text

    def update(frame):
        # Leapfrog
        orbit_line.set_data(x_vals[:frame+1], y_vals[:frame+1])
        earth_point.set_data([x_vals[frame]], [y_vals[frame]])

        if speed_vals is not None:
            vel_text.set_text(f"v = {4.74047*speed_vals[frame]:.3f} km/s")
        else:
            vel_text.set_text("")

        if include_euler:
            euler_line.set_data(euler_x_vals[:frame+1], euler_y_vals[:frame+1])
            euler_point.set_data([euler_x_vals[frame]], [euler_y_vals[frame]])

            time_text.set_text(
                f"t = {12*t_vals[frame]:.3f} months"
            )

            return orbit_line, earth_point, euler_line, euler_point, time_text
        else:
            time_text.set_text(f"t = {12*t_vals[frame]:.3f} months")
            return orbit_line, earth_point, time_text
        
        

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)
    anim.save(filename, writer="pillow", fps=fps)
    plt.close(fig)

def gravity_accel(x, y, m_1=1):
    r = math.sqrt(x**2 + y**2)
    factor = -(4 * math.pi**2 * m_1) / (r**3)
    return factor * x, factor * y

def leapfrog(initial_pos, initial_vel, acceleration_func, tot_time, tot_steps, m_1=1):
    x, y = initial_pos
    vx, vy = initial_vel
    dt = tot_time / tot_steps

    pos_mat_x = [x]
    pos_mat_y = [y]
    vel_mat_x = [vx]
    vel_mat_y = [vy]
    time_mat = [0.0]

    ax, ay = acceleration_func(x, y, m_1)

    # half-step velocity
    vx_half = vx + 0.5 * ax * dt
    vy_half = vy + 0.5 * ay * dt

    for i in range(tot_steps):
        # full-step position
        x += vx_half * dt
        y += vy_half * dt

        pos_mat_x.append(x)
        pos_mat_y.append(y)
        time_mat.append((i + 1) * dt)

        # acceleration at new position
        ax, ay = acceleration_func(x, y, m_1)

        vel_mat_x.append(vx_half + 0.5 * ax * dt)
        vel_mat_y.append(vy_half + 0.5 * ay * dt)
        # full-step half-velocity update
        vx_half += ax * dt
        vy_half += ay * dt

    return pos_mat_x, pos_mat_y, vel_mat_x, vel_mat_y, time_mat

def leapfrog_voyager(initial_pos, initial_vel, pos_j_x, pos_j_y, acceleration_func, tot_time, tot_steps, m_1=1):
    x, y = initial_pos
    vx, vy = initial_vel
    dt = tot_time / tot_steps

    pos_mat_x = [x]
    pos_mat_y = [y]
    vel_mat_x = [vx]
    vel_mat_y = [vy]
    time_mat = [0.0]

    ax, ay = acceleration_func(x, y, pos_j_x[0], pos_j_y[0], m_1)

    # half-step velocity
    vx_half = vx + 0.5 * ax * dt
    vy_half = vy + 0.5 * ay * dt

    for i in range(tot_steps):
        # full-step position
        x += vx_half * dt
        y += vy_half * dt

        pos_mat_x.append(x)
        pos_mat_y.append(y)
        time_mat.append((i + 1) * dt)

        # acceleration at new position
        ax, ay = acceleration_func(x, y, pos_j_x[i], pos_j_y[i], m_1)

        vel_mat_x.append(vx_half + 0.5 * ax * dt)
        vel_mat_y.append(vy_half + 0.5 * ay * dt)
        # full-step half-velocity update
        vx_half += ax * dt
        vy_half += ay * dt

    return pos_mat_x, pos_mat_y, vel_mat_x, vel_mat_y, time_mat

def euler_method(initial_pos, initial_vel, acceleration_func, tot_time, tot_steps, m_1=1):
    x, y = initial_pos
    vx, vy = initial_vel
    dt = tot_time / tot_steps

    pos_mat_x = [x]
    pos_mat_y = [y]
    vel_mat_x = [vx]
    vel_mat_y = [vy]
    time_mat = [0.0]

    for i in range(tot_steps):
        ax, ay = acceleration_func(x, y, m_1)

        x += vx * dt
        y += vy * dt
        vx += ax * dt
        vy += ay * dt

        pos_mat_x.append(x)
        pos_mat_y.append(y)
        vel_mat_x.append(vx)
        vel_mat_y.append(vy)
        time_mat.append((i + 1) * dt)

    return pos_mat_x, pos_mat_y, vel_mat_x, vel_mat_y, time_mat

'''
initial_pos = [1.0, 0.0]
initial_vel = [0.0, 2*math.pi]

x_l, y_l, vx_l, vy_l, t_l = leapfrog(initial_pos, initial_vel, gravity_accel, 3, 900)
x_e, y_e, vx_e, vy_e, t_e = euler_method(initial_pos, initial_vel, gravity_accel, 3, 900)

make_orbit_movie(x_l, y_l, t_l, euler_t_vals=t_e, euler_x_vals=x_e, euler_y_vals=y_e, filename="earth_orbit_with_euler_problem1.gif")

v_leap = []
v_euler = []
e_euler = []
e_leap = []
for i in range(len(vx_l)):
    v_leap.append(math.sqrt(vx_l[i]**2 + vy_l[i]**2))
    v_euler.append(math.sqrt(vx_e[i]**2 + vy_e[i]**2))
    e_euler.append(0.5*v_euler[i]**2 - (4*math.pi**2)/(math.sqrt(x_e[i]**2 + y_e[i]**2)))
    e_leap.append(0.5*v_leap[i]**2 - (4*math.pi**2)/(math.sqrt(x_l[i]**2 + y_l[i]**2)))

plt.plot(t_e, v_euler, label="Euler", color="red")
plt.plot(t_l, v_leap, label="Leapfrog", color="blue")
plt.xlabel('t(years)')
plt.ylabel('v(AU/year)')
plt.legend(loc=1)
plt.show()

plt.plot(t_e, e_euler, label="Euler", color="red")
plt.plot(t_l, e_leap, label="Leapfrog", color="blue")
plt.xlabel('t(years)')
plt.ylabel('Energy')
plt.legend(loc=1)
plt.show()


# Start of problem 2
initial_pos = [1.0, 0.0]
initial_vel = [0.0, 0.8 * (2 * math.pi)]

x_l, y_l, vx_l, vy_l, t_l = leapfrog(initial_pos, initial_vel, gravity_accel, 5, 900)
x_e, y_e, vx_e, vy_e, t_e = euler_method(initial_pos, initial_vel, gravity_accel, 5, 900)

make_orbit_movie(x_l, y_l, t_l, euler_t_vals=t_e, euler_x_vals=x_e, euler_y_vals=y_e, filename="earth_orbit_with_euler_problem2.gif")

v_leap = []
v_euler = []
e_euler = []
e_leap = []
for i in range(len(vx_l)):
    v_leap.append(math.sqrt(vx_l[i]**2 + vy_l[i]**2))
    v_euler.append(math.sqrt(vx_e[i]**2 + vy_e[i]**2))
    e_euler.append(0.5*v_euler[i]**2 - (4*math.pi**2)/(math.sqrt(x_e[i]**2 + y_e[i]**2)))
    e_leap.append(0.5*v_leap[i]**2 - (4*math.pi**2)/(math.sqrt(x_l[i]**2 + y_l[i]**2)))

plt.plot(t_e, e_euler, label="Euler", color="red")
plt.plot(t_l, e_leap, label="Leapfrog", color="blue")
plt.xlabel('t(years)')
plt.ylabel('Energy')
plt.legend(loc=1)
plt.show()
'''

# Start of problem 3

# get the trajectory of Jupiter
initial_pos = [5.2, 0.0]
initial_vel =  [0.0, 2 * math.pi / math.sqrt(5.2)]
x_j, y_j, vx_j, vy_j, t_j = leapfrog(initial_pos, initial_vel, gravity_accel, 20, 900)
#make_orbit_movie(x_j, y_j, t_j, filename="jupiter_check.gif")

# define the potential that voyager will see
def gravity_accel_voyager(x, y, x_j, y_j, m_1=1, m_2 = 0.000954):
    r_s = math.sqrt(x**2 + y**2)
    r_j = math.sqrt((x - x_j)**2 + (y - y_j)**2)
    factor = -(4 * math.pi**2 * m_1) / (r_s**3) - (4 * math.pi**2 * m_1) / (r_j**3)
    return factor * x, factor * y

initial_pos_v = [1.0, 0.0]
initial_vel_v = [0.0,  1.4*(2 * math.pi)]
x_v, y_v, vx_v, vy_v, t_v = leapfrog_voyager(initial_pos_v, initial_vel_v, x_j, y_j, gravity_accel_voyager, 20, 900)
make_orbit_movie(x_j, y_j, t_j, euler_x_vals=x_v, euler_y_vals=y_v, euler_t_vals=t_v, vel_x=vx_v, vel_y=vy_v, filename="voyager_traj.gif")

r_mat = []
v_mat = []
for i in range(len(x_v)):
    r_mat.append(math.sqrt((x_v[i] - x_j[i])**2 + (y_v[i] - y_j[i])**2))


print(min(r_mat))
print(t_v[r_mat.index(min(r_mat))])