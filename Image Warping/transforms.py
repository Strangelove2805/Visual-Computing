import numpy as np


def transform_pixel_nn(x, y, transform):
    """Transforms a source pixel coordinate (x, y) using 'transform', and rounds to the nearest pixel
    coordinate. Returns a tuple (x', y')."""

    p = np.array([x, y, 1]).T
    q = transform @ p
    
    return int(np.rint(q[0] / q[2])), int(np.rint(q[1] / q[2]))


def forward_mapping(source, transform):
    """Warps the 'source' image by the given 'transform' using forward mapping."""
    out = np.zeros_like(source)
    for y in range(source.shape[0]):
        for x in range(source.shape[1]):

            u, v = transform_pixel_nn(x,y,transform)
            if 0 <= u < out.shape[1] and 0 <= v < out.shape[0]:
                out[v, u, :] = source[y, x, :]
                
    return out


def get_backward_px(x, y, transform):

    p = np.array([x, y, 1]).T
    
    q = transform @ p

    return int(np.rint(q[0] / q[2])), int(np.rint(q[1] / q[2]))
    

def backward_mapping(source, transform):
    """Warps the 'source' image by the given 'transform' using backward mapping with nearest-neighbour interpolation."""
    out = np.zeros_like(source)
    tf_new = np.linalg.inv(transform)

    for y in range(source.shape[0]):
        for x in range(source.shape[1]):

            u, v = get_backward_px(x,y,tf_new)
            if 0 <= u < out.shape[1] and 0 <= v < out.shape[0]:
                out[y, x, :] = source[v, u, :]
                
    return out


def get_bilinear_px(x, y, transform):

    old = np.array([[y],[x],[1]])
    new = np.dot(transform, old)

    return new[0], new[1]


def backward_mapping_bilinear(source, transform):
    """Warps the 'source' image by the given 'transform' using backward mapping with bilinear interpolation."""
    out = np.zeros_like(source)
    tf_new = np.linalg.inv(transform)
    for i in range(len(out)):
        for j in range(len(out[0])):
            try:
                y, x = get_bilinear_px(i,j,tf_new)
                
                if x >= 0 and y >= 0:
                    pass
                else:
                    continue

                alpha = y - np.floor(y)
                beta = x - np.floor(x)
                
                try:
                    f1 = source[int(np.floor(x))][int(np.floor(y))]
                except:
                    f1 = (0,0,0)

                try:
                    f2 = source[int(np.floor(x))][int(np.ceil(y))]
                except:
                    f2 = (0,0,0)

                try:
                    f3 = source[int(np.ceil(x))][int(np.floor(y))]
                except:
                    f3 = (0,0,0)

                try:
                    f4 = source[int(np.ceil(x))][int(np.ceil(y))]
                except:
                    f4 = (0,0,0)

                f12 = ((1-alpha) * f1) + (alpha * f2)
                f34 = ((1-alpha) * f3) + (alpha * f4)

                out[i][j] = ((1-beta) * f12) + (beta * f34)

            except:
                continue
    return out


def undistort_point(u, v, camera_matrix, dist_coeffs):
    """Undistorts a pixel's coordinates (u, v) using the given camera matrix and
    distortion coefficients. Returns a tuple (u', v')."""
    
    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    px = camera_matrix[0][2]
    py = camera_matrix[1][2]
    k1 = dist_coeffs[0]
    k2 = dist_coeffs[1]
    k3 = dist_coeffs[2]
    
    x = (u - px) / fx
    y = (v - py) / fy
    
    xsq = x**2
    ysq = y**2
    r = np.sqrt(xsq + ysq)
    
    xprime = np.dot(x, (1 + (k1 * r**2) + (k2 * r**4) + (k3 * r**6)))
    yprime = np.dot(y, (1 + (k1 * r**2) + (k2 * r**4) + (k3 * r**6)))
    
    uprime = np.dot(xprime, fx) + px
    vprime = np.dot(yprime, fy) + py
    
    return uprime, vprime


def undistort_image_vectorised(image, camera_matrix, dist_coeffs):
    """Undistorts an image using the given camera matrix and distortion coefficients.
    Use vectorised operations to avoid slow for loops."""
    out = np.zeros_like(image)
    v = (np.array([([i] * len(image[0])) for i in range(len(image))])).flatten()
    u = (np.array([(list(range(len(image[0]))))] * len(image))).flatten()
    
    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    px = camera_matrix[0][2]
    py = camera_matrix[1][2]
    k1 = dist_coeffs[0]
    k2 = dist_coeffs[1]
    k3 = dist_coeffs[2]
    
    x = (u - px) / fx
    y = (v - py) / fy
    
    xsq = x**2
    ysq = y**2

    r = np.sqrt(xsq + ysq)
    
    arr = [(1 + (k1 * rr**2) + (k2 * rr**4) + (k3 * rr**6)) for rr in r]

    xprime = [np.dot(x[n], arr[n]) for n in range(len(x))]
    yprime = [np.dot(y[n], arr[n]) for n in range(len(y))]
    
    uprime = (np.array([(np.dot(xprime[n], fx) + px) for n in range(len(xprime))])).reshape(len(image), len(image[0]))
    vprime = (np.array([(np.dot(yprime[n], fy) + py) for n in range(len(yprime))])).reshape(len(image), len(image[0]))
    
    for i in range(len(image)):
        for j in range(len(image[0])):
            
            try:
                out[i][j] = image[int(vprime[i][j])][int(uprime[i][j])]
            except:
                continue

    return out

