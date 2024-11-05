import numpy as np
import matplotlib.pyplot as plt
import bezier_curves as bz

curve_control_points = [
    np.array([[10,10],[30,10],[30,70],[80,80]]),
    np.array([[70,10],[50,30],[30,70],[20,10]]),
    np.array([[50,50],[90,90],[10,90],[90,10]]),
    np.array([[50,90],[50,10],[10,10],[80,80]]),
]

curve_sample_points = [
    np.array([[10,      10     ], [22.34375, 19.53125], [33.75,    41.25   ], [50.78125, 64.84375], [80,      80.     ]]),
    np.array([[70,   10     ], [55.15625, 26.875  ], [41.25,    40.     ], [29.21875, 38.125  ], [20,      10.     ]]),
    np.array([[50,    50.   ], [61.875, 71.875], [55,    75.   ], [55.625, 55.625], [90,    10.   ]]),
    np.array([[50,      90.     ], [44.84375, 44.84375], [38.75,    28.75   ], [45.78125, 40.78125], [80,      80.     ]]),
]

for i,P in enumerate(curve_control_points):
    t = np.reshape(np.linspace(0,1,5),(-1,1))
    sampled_bezier_points = bz.cubic_bezier(t,P)
    print("Is curve",i,"sampled correctly?",np.allclose(sampled_bezier_points,curve_sample_points[i]))


# Visualise the curves
fig,ax = plt.subplots(1,len(curve_control_points),figsize=(16,4))
for i,P in enumerate(curve_control_points):
    t = np.reshape(np.linspace(0,1,20),(-1,1))
    sampled_bezier_points = bz.cubic_bezier(t,P)
    ax[i].plot(sampled_bezier_points[:,0],sampled_bezier_points[:,1])
    ax[i].set_title("Curve "+str(i))
    ax[i].scatter(sampled_bezier_points[:,0],sampled_bezier_points[:,1])
    ax[i].scatter(P[:,0],P[:,1])
plt.show()


curve_sampled_tangents = [
    np.array([[0.99999998, 0.        ], [0.52793411, 0.84928531], [0.47409982, 0.88047109], [0.722308,   0.69157149], [0.98058067, 0.19611613]]),
    np.array([[-0.70710677,  0.70710677], [-0.65252313,  0.75776879], [-0.86824313,  0.49613893], [-0.63473941, -0.77272624], [-0.16439899, -0.98639392]]),
    np.array([[ 0.70710678,  0.70710678], [-0.14142135,  0.98994947], [-0.70710676, -0.70710676], [ 0.3807498,  -0.92467809], [ 0.70710678, -0.70710678]]),
    np.array([[ 0,         -1.        ], [-0.25302774, -0.96745902], [-0.70710671, -0.70710671], [ 0.57842877,  0.81573288], [ 0.70710678,  0.70710678]]),
]

for i,P in enumerate(curve_control_points):
    t = np.reshape(np.linspace(0,1,5),(-1,1))
    sampled_bezier_points = bz.cubic_bezier(t,P)
    sampled_tangent_vectors = bz.cubic_bezier_tangents(t,P)
    print("Is curve",i,"sampled correctly?",np.allclose(sampled_tangent_vectors,curve_sampled_tangents[i]))


line_length = 3
fig,ax = plt.subplots(1,len(curve_control_points),figsize=(16,4))
for i,P in enumerate(curve_control_points):
    t = np.reshape(np.linspace(0,1,10),(-1,1))

    sampled_bezier_points = bz.cubic_bezier(t,P)
    sampled_tangent_vectors = bz.cubic_bezier_tangents(t,P)
    ax[i].scatter(sampled_bezier_points[:,0],sampled_bezier_points[:,1])
    #for j in range(sampled_bezier_points.shape[0]):
    x = [sampled_bezier_points[:,0]-line_length*sampled_tangent_vectors[:,0],sampled_bezier_points[:,0]+line_length*sampled_tangent_vectors[:,0]]
    y = [sampled_bezier_points[:,1]-line_length*sampled_tangent_vectors[:,1],sampled_bezier_points[:,1]+line_length*sampled_tangent_vectors[:,1]]
    ax[i].set_title("Curve "+str(i))
    ax[i].plot(x,y)
    ax[i].scatter(P[:,0],P[:,1])
plt.show()


curve_sample_points = [
    np.array([[10,      10     ], [22.34375, 19.53125], [33.75,    41.25   ], [50.78125, 64.84375], [80,      80.     ]]),
    np.array([[70,   10     ], [55.15625, 26.875  ], [41.25,    40.     ], [29.21875, 38.125  ], [20,      10.     ]]),
    np.array([[50,    50.   ], [61.875, 71.875], [55,    75.   ], [55.625, 55.625], [90,    10.   ]]),
    np.array([[50,      90.     ], [44.84375, 44.84375], [38.75,    28.75   ], [45.78125, 40.78125], [80,      80.     ]]),
]

for i,P in enumerate(curve_control_points):
    t = np.reshape(np.linspace(0,1,5),(-1,1))
    sampled_bezier_points = bz.bezier_decasteljau(t,P)[0,...].T
    print("Is curve",i,"sampled correctly?",np.allclose(sampled_bezier_points,curve_sample_points[i]))

fig,ax = plt.subplots(1,len(curve_control_points),figsize=(16,4))
for i,P in enumerate(curve_control_points):
    t = np.random.random((1,1))
    for j in range(4):
        sampled_bezier_points = bz.bezier_decasteljau(t,P,L=j)
        ax[i].plot(sampled_bezier_points[:,0],sampled_bezier_points[:,1])
        ax[i].scatter(sampled_bezier_points[:,0],sampled_bezier_points[:,1])
plt.show()