import numpy as np 

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between_two_vectors(vector_1, vector_2):
    v1_u = unit_vector(vector_1)
    v2_u = unit_vector(vector_2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + (np.cos(angle) * (px - ox)) - (np.sin(angle) * (py - oy))
    qy = oy + (np.sin(angle) * (px - ox)) + (np.cos(angle) * (py - oy))
    return [int(np.ceil(qx)), int(np.ceil(qy))]

def translate_point(point, d_x, d_y):
    """Translate a point by d_x and d_y"""
    x, y = point
    return [int(np.ceil(x+d_x)), int(np.ceil(y+d_y))]