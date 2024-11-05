import numpy as np


def scaling(scale_factor):
    """Returns a transform matrix for uniform scaling about the origin by 'scale_factor'."""
    
    matrix = np.array([[scale_factor, 0, 0],[0, scale_factor, 0],[0, 0, 1]])

    return matrix


def translation(point):
    """Returns a transform matrix for translation by 'point[0]' units
    along the x-axis and 'point[1]' units along the y-axis."""

    matrix = np.array([[1, 0, point[0]],[0, 1, point[1]],[0, 0, 1]])
    
    return matrix


def rotation(angle):
    """Returns a transform matrix for anti-clockwise rotation about the origin by 'angle' degrees."""
    
    angle = np.radians(angle)
    
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]])
    
    return matrix


def rotation_scaling_and_translation(angle, scale_factor, point):
    """Returns a compound transform for rotating by 'angle', scaling by 'scaling_factor',
    and translating by 'point'."""
    
    rotate = rotation(angle)
    scale = scaling(scale_factor)
    trans = translation(point)
    
    matrix = np.linalg.multi_dot([trans,scale,rotate])
    
    return matrix


def rotation_scaling_and_translation_postmultiplied(angle, scale_factor, point):
    """Returns a post-multiplied compound transform for rotating by 'angle',
    scaling by 'scaling_factor', and translating by 'point'."""
    
    rotate = rotation(angle)
    scale = scaling(scale_factor)
    trans = translation(point)
    
    matrix = np.linalg.multi_dot([trans,scale,rotate])
    
    return matrix.T
