import math
import matplotlib.pyplot as plt



def leapfrog(initial_pos, initial_vel, acceleration_func, tot_time, tot_steps):
    pos_mat_x = [initial_pos[0]]
    pos_mat_y = [initial_pos[1]]
    time_mat = [0]
    time_step = tot_time/tot_steps
    pos_x = initial_pos[0]
    pos_y = initial_pos[1]
    theta = math.atan(pos_y/pos_x)
    v_12_x = initial_pos[1] + 0.5*acceleration_func(pos_x, pos_y, 1)*math.sin(theta)*time_step
    v_12_y = initial_vel[1] + 0.5*acceleration_func(pos_x, pos_y, 1)*math.cos(theta)*time_step
    i = 0
    while i <= tot_steps:
        theta = math.atan(pos_y/pos_x)
        pos_x += v_12_x*time_step
        pos_y += v_12_y*time_step
        pos_mat_x.append(pos_x)
        pos_mat_y.append(pos_y)
        v_12_x += 0.5*acceleration_func(pos_x, pos_y, 1)*math.cos(theta)*time_step
        v_12_y += 0.5*acceleration_func(pos_x, pos_y, 1)*math.sin(theta)*time_step
        i+= 1
        time_mat.append(i*time_step)
    return pos_mat_x, pos_mat_y, time_mat

def gravity_func(x, y, m_1):
    r = math.sqrt(x**2 + y**2)
    return -(4*(math.pi**2)*m_1)/(r**2)

x, y, t = leapfrog([1, 0], [0,0.00001], gravity_func, 10, 1000)

plt.plot(t, y)
plt.show()