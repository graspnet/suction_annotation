import numpy as np
from math import pi

def rotation_matrix(alpha, beta, gamma):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    return R

# R = rotation_matrix(pi/4, pi/4, 0)
# approach = R[:, 0]
# binormal = R[:, 1]
# minor_normal = R[:, 2]

# axis_y = binormal
# axis_x = np.array([axis_y[1], -axis_y[0], 0])
# if np.linalg.norm(axis_x) == 0:
#     axis_x = np.array([1, 0, 0])
# axis_x = axis_x / np.linalg.norm(axis_x)
# axis_y = axis_y / np.linalg.norm(axis_y)
# axis_z = np.cross(axis_x, axis_y)
# R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
# approach = R2.T.dot(approach)
# print(approach)
# cos_t, sin_t = approach[0], -approach[2]
# t = np.arccos(cos_t)
# if sin_t < 0:
#     t = pi * 2 - t
# print(t * 180 / pi)

def matrix_to_dexnet_params(matrix):
    approach = matrix[:, 0]
    binormal = matrix[:, 1]
    axis_y = binormal
    axis_x = np.array([axis_y[1], -axis_y[0], 0])
    if np.linalg.norm(axis_x) == 0:
        axis_x = np.array([1, 0, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R = np.c_[axis_x, np.c_[axis_y, axis_z]]
    approach = R.T.dot(approach)
    cos_t, sin_t = approach[0], -approach[2]
    angle = np.arccos(cos_t)
    if sin_t < 0:
        angle = pi * 2 - angle
    return binormal, angle

def viewpoint_to_matrix(towards):
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2
    return matrix

def viewpoint_params_to_matrix(towards, angle):
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix

def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    axis_x = batch_towards
    ones = np.ones(axis_x.shape[0], dtype=axis_x.dtype)
    zeros = np.zeros(axis_x.shape[0], dtype=axis_x.dtype)
    axis_y = np.stack([-axis_x[:,1], axis_x[:,0], zeros], axis=-1)
    axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=-1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)
    sin = np.sin(batch_angle)
    cos = np.cos(batch_angle)
    R1 = np.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], axis=-1)
    R1 = R1.reshape([-1,3,3])
    R2 = np.stack([axis_x, axis_y, axis_z], axis=-1)
    matrix = np.matmul(R2, R1)
    return matrix.astype(np.float32)

def dexnet_params_to_matrix(binormal, angle):
    axis_y = binormal
    axis_x = np.array([axis_y[1], -axis_y[0], 0])
    if np.linalg.norm(axis_x) == 0:
        axis_x = np.array([1, 0, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix