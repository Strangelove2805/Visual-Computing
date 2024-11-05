import numpy as np
import matplotlib.pyplot as plt
import transforms as trans
import solar
from matplotlib import animation
from IPython.display import HTML


# 2D polygon for Earth: square of 2 by 2 units, in homogeneous coordinates.
earth = np.array([[-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [1, 1, 1, 1, 1]])

# 2D polygons for the moons, made by scaling down the Earth polygon.
moon1 = trans.scaling(0.3) @ earth
moon2 = trans.scaling(0.2) @ earth
moon3 = trans.scaling(0.1) @ earth

# Prepare a figure for animation.
fig, ax = plt.subplots(figsize=[8, 8])
ax.set_xlim((-6, 6))
ax.set_ylim((-6, 6))
ax.set_aspect('equal', adjustable='box')
earth_line, = ax.plot([], [], 'b', linewidth=2)
moon1_line, = ax.plot([], [], 'k', linewidth=2)
moon2_line, = ax.plot([], [], 'g', linewidth=2)
moon3_line, = ax.plot([], [], 'r', linewidth=2)

def init_earth_and_moons():
    earth_line.set_data([], [])
    moon1_line.set_data([], [])
    moon2_line.set_data([], [])
    moon3_line.set_data([], [])
    return (earth_line, moon1_line, moon2_line, moon3_line)

def animate_earth_and_moons(step):
    # Convert step=0..99 to theta=0..360(ish) for 360 degree rotation.
    theta = step / 100 * 360

    # Put the Earth and moons into the solar system.
    p_earth = solar.transform_earth(theta) @ earth; p_earth /= p_earth[2,:]
    p_moon1 = solar.transform_moon1(theta) @ moon1; p_moon1 /= p_moon1[2,:]
    p_moon2 = solar.transform_moon2(theta) @ moon2; p_moon2 /= p_moon2[2,:]
    p_moon3 = solar.transform_moon3(theta) @ moon3; p_moon3 /= p_moon3[2,:]

    # Update coordinates of all polygons.
    earth_line.set_data(p_earth[0,:], p_earth[1,:])
    moon1_line.set_data(p_moon1[0,:], p_moon1[1,:])
    moon2_line.set_data(p_moon2[0,:], p_moon2[1,:])
    moon3_line.set_data(p_moon3[0,:], p_moon3[1,:])
    return (earth_line, moon1_line, moon2_line, moon3_line)

anim_earth_and_moons = animation.FuncAnimation(fig, animate_earth_and_moons,
                                               init_func=init_earth_and_moons,
                                               frames=100, interval=40, blit=True)
plt.show()
