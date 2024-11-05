import numpy as np
import matplotlib.pyplot as plt
import transforms as trans
from matplotlib import animation
from IPython.display import HTML


# A 2D polygon: a square of 2 by 2 units, in homogeneous coordinates.
# Each column in the matrix defines the coordinates of one corner of the square.
p = np.array([[-1,  1, 1, -1, -1],
              [-1, -1, 1,  1, -1],
              [ 1,  1, 1,  1,  1]])

# The square should eventually rotate about this point.
point = (2, 3)
scale_factor = 1.5

# Prepare a figure for animation.
fig, ax = plt.subplots()
ax.set_xlim((-2, 6))
ax.set_ylim((-2, 6))
ax.set_aspect('equal', adjustable='box')
line, = ax.plot([], [], 'b', linewidth=3)
ax.plot(point[0], point[1], 'r.', markersize=20)

def init():
    line.set_data([], [])
    return (line,)

def animate(step):
    # Convert step=0..29 to theta=0..90ish for 90 degree rotation.
    angle = step * 3

    # Apply the transformation.
    transform = trans.rotation_scaling_and_translation(angle, scale_factor, point)
    pprime = transform @ p
    pprime /= pprime[2,:]

    # Update coordinates of the polygon.
    line.set_data(pprime[0,:], pprime[1,:])
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=30, interval=33, blit=True)

plt.show()

