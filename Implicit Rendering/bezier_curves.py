import numpy as np


def cubic_bezier(t,P):
    """Sample a cubic Bezier
    t is an Mx1 array
    P is an Nx2 array
    Returns Y - an Mx2 array"""

    B = np.array([[1,0,0,0],[-3,3,0,0],[3,-6,3,0],[-1,3,-3,1]])
    Y = []

    for i in range(len(t)):
        T = [1, float(t[i]), float(t[i]**2), float(t[i]**3)]
        Y.append(np.linalg.multi_dot([T,B,P]))

    return np.array(Y)


def cubic_bezier_tangents(t,P):
    """Generate the Bezier tangent vector
    t is an Mx1 array
    P is an Nx2 array
    Returns normalised_tangents - an Mx2 array"""

    B = np.array([[1,0,0,0],[-3,3,0,0],[3,-6,3,0],[-1,3,-3,1]])
    Y = []

    for i in range(len(t)):
        T = [0, 1, float(2 * t[i]), 3 * float(t[i]**2)]
        
        dot_prod = np.linalg.multi_dot([T,B,P])
        norm = np.linalg.norm(dot_prod)
        Y.append(dot_prod/norm)

    return np.array(Y)


def bezier_decasteljau(t,P,L=None):
    """De Cateljau's algorithm
    t is an Mx1 array
    P is an Nx2 array
    L is a scalar value representing at which level you want to stop the calculation"""
    
    if L is None:
        L = P.shape[0]-1

    beta = [c for c in P] # values in this list are overridden

    n = len(beta)
    
    out = []
    
    for j in range(1, n):

        for k in range(n - j):

            beta[k] = beta[k] * (1 - t) + beta[k + 1] * t
            
            
            out.insert(0, beta[k].T)

    return np.array(out)
