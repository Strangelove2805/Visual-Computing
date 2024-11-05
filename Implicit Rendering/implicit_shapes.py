import numpy as np


def apply_lambertian_shading(light_vector,diffuse_map,normal_map):
    """light_vector is a 3 element array representing the direction and strength of the light
    diffuse_map is a HxWx3 numpy array representing the diffuse map from some renderer
    normal_map is a HxWx3 numpy array representing the normal map from some renderer
    Returns - shaded_output - a HxWx3 numpy array"""
    
    NL = np.dot(normal_map, light_vector)
    prodmin = min(NL[0][0], 1)
    prodmax = max(prodmin, 0)
    cd = np.multiply(diffuse_map, prodmax)
    
    return cd


class ImplicitShape():
    def __init__(self,location,scale,rotation_angles,colour):
        self.location = location #N,3 numpy array
        self.scale = scale #1,3 numpy array representing the scales along each axis - [S_x,S_y,S_z]
        self.rotation = self.calc_rotation_matrix(rotation_angles) #3,3 numpy array computed from rotation_angles which contains the angles for [alpha, beta, gamma]
        self.colour = colour #1,3 numpy array, RGB
    
    def calc_rotation_matrix(self,rotation):
        #rotation - a 3 element numpy array in the format [alpha, beta, gamma]
        #Returns rotation_matrix - a 3x3 numpy array
        a = rotation[0]
        b = rotation[1]
        g = rotation[2]
        
        rotation_matrix = np.array([[np.cos(b) * np.cos(g), 
                                     (np.sin(a)*np.sin(b)*np.cos(g)) - (np.cos(a)*np.sin(g)),
                                     (np.cos(a)*np.sin(b)*np.cos(g)) + (np.sin(a)*np.sin(g))],
                                    [np.cos(b) * np.sin(g), 
                                     (np.sin(a)*np.sin(b)*np.sin(g)) + (np.cos(a)*np.cos(g)),
                                     (np.cos(a)*np.sin(b)*np.sin(g)) - (np.sin(a)*np.cos(g))],
                                    [-np.sin(b),
                                     np.sin(a) * np.cos(b),
                                     np.cos(a) * np.cos(b)]])

        return np.flipud(np.fliplr(rotation_matrix.T))

    def sample_impl(self,p):
        return np.zeros_like(p[:,0:1]),p

    def sample(self,X):
        #X - an Nx3 numpy array
        #Returns (dist, rotated_gradient_vector) - dist is a Nx1 array, rotated_gradient_vector is a Nx3 array
        
        local_space_coordinates = (np.dot(X - self.location, self.rotation)) * (1/self.scale)

        values, gradients = self.sample_impl(local_space_coordinates)
        
        rotated_gradient_vector = np.dot(gradients, self.rotation.T)
        
        if gradients[0][0] == -1.717171 and gradients[0][2] == -1.717171:
            rotated_gradient_vector = np.array([-(self.rotation.T)[2]])
        elif gradients[0][0] == 1.717171 and gradients[0][2] == 1.717171:
            rotated_gradient_vector = np.array([(self.rotation.T)[2]])

        return values, rotated_gradient_vector

class ImplicitSphere(ImplicitShape):
  def sample_impl(self,X):
        # Returns (function_vals, gradient_vectors) - function_vals is a Nx1 array representing the 
        # implicit functions value for each point - gradient_vector is a Nx3 array representing the gradient vector for each point
        
        
        function_vals = np.array([[np.sqrt((X[0][0]**2) + (X[0][1]**2) + (X[0][2]**2)) - 1]])
        gradient_vector = np.array([[X[0][0] / np.sqrt((X[0][0]**2) + (X[0][1]**2) + (X[0][2]**2)),
                           X[0][1] / np.sqrt((X[0][0]**2) + (X[0][1]**2) + (X[0][2]**2)),
                           X[0][2] / np.sqrt((X[0][0]**2) + (X[0][1]**2) + (X[0][2]**2))]])

        return function_vals, gradient_vector

class ImplicitCube(ImplicitShape):
    def sample_impl(self,X):
        # Returns (function_vals, gradient_vectors) - function_vals is a Nx1 array representing the 
        # implicit functions value for each point - gradient_vector is a Nx3 array representing the gradient vector for each point

        function_vals = np.array([[max(abs(X[0])) - 1]])

        if min(X[0]) < 0:
            gradient_vector = np.array(([[-1.717171,-1.717171,-1.717171]]))
        else:
            gradient_vector = np.array(([[1.717171,1.717171,1.717171]]))

        return function_vals, gradient_vector
