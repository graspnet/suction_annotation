import numpy as np
import open3d as o3d
import os
import random
import argparse
from transforms3d.euler import euler2mat


def transform_points(points, trans):
    ones = np.ones([points.shape[0],1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    points_ = np.matmul(trans, points_.T).T
    return points_[:,:3]


def transform_normals(points, trans):
    ones = np.ones([points.shape[0],1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    trans[:3, 3] = 0
    points_ = np.matmul(trans, points_.T).T
    return points_[:,:3]

def trans_model(model):
    obj_dir = os.path.join(MODEL_ROOT, model, 'textured.obj')
    ply_dir = os.path.join(MODEL_ROOT, model, 'nontextured.ply')
    print(os.path.join(obj_dir))
    
    pc = o3d.io.read_point_cloud(ply_dir)

    points = np.asarray(pc.points)

    mat = np.zeros([4,4],dtype=np.float32)
    
    alpha = random.randint(0, 180) / 180.0 * np.pi
    beta = random.randint(0, 180) / 180.0 * np.pi
    gamma = random.randint(0, 180) / 180.0 * np.pi
    mat[:3,:3] = euler2mat(alpha, beta, gamma)
    mat[0,3] = random.randint(0, 180) / 180.0
    mat[1,3] = random.randint(0, 180) / 180.0
    mat[2,3] = random.randint(0, 180) / 180.0
    mat[3,3] = 1

    points = transform_points(points, mat)
    pc.points =  o3d.utility.Vector3dVector(points)

    return pc, mat 


parser = argparse.ArgumentParser()
parser.add_argument('--model_root', default='', help='Directory of graspnet model')
parser.add_argument("--model_idx", type=int, default=0, help='the index of model to visualize normals [default: 0]')
parser.add_argument("--normal_num", type=int, default=50, help='the number of normals to visualize')
args = parser.parse_args()

MODEL_ROOT = args.model_root
NORMAL_NUM = args.normal_num


if __name__ == "__main__":
    
    model = '%03d' % args.model_idx    
    print('Visualizing:', model)
    
    obj_dir = os.path.join(MODEL_ROOT, model, 'textured.obj')
    ply_dir = os.path.join(MODEL_ROOT, model, 'nontextured.ply')
    
    mesh = o3d.io.read_triangle_mesh(obj_dir)
    pc = o3d.io.read_point_cloud(ply_dir)

    points = np.asarray(pc.points)

    mat = np.zeros([4,4],dtype=np.float32)
    
    alpha = random.randint(0, 180) / 180.0 * np.pi
    beta = random.randint(0, 180) / 180.0 * np.pi
    gamma = random.randint(0, 180) / 180.0 * np.pi
    mat[:3,:3] = euler2mat(alpha, beta, gamma)
    mat[0,3] = random.randint(0, 180) / 180.0
    mat[1,3] = random.randint(0, 180) / 180.0
    mat[2,3] = random.randint(0, 180) / 180.0
    mat[3,3] = 1

    points = transform_points(points, mat)
    pc.points =  o3d.utility.Vector3dVector(points)

    pc, mat = trans_model(model)
    points = np.array(pc.points)

    normals = np.asarray(pc.normals)

    normals = transform_normals(normals, mat)

    suction_index = np.random.choice(points.shape[0], NORMAL_NUM)
    suction_points = points[suction_index]
    suction_normals = normals[suction_index]

    arrows = []

    for i in range(suction_points.shape[0]):
        suction_point = suction_points[i]
        suction_normal = suction_normals[i]
        if np.isnan(np.sum(suction_normal)):
            print('Nan')
            continue
        if np.sum(suction_normal) == 0:
            print('Zero')
            continue
        
        suction_normal = suction_normal / np.linalg.norm(suction_normal)
    

        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.001, cone_radius=0.0015, 
                                                        cylinder_height=0.005, cone_height=0.004)
        arrow_points = np.asarray(arrow.vertices)

        new_z = suction_normal
        new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
        new_y = new_y / np.linalg.norm(new_y)
        new_x = np.cross(new_y, new_z)

        R = np.c_[new_x, np.c_[new_y, new_z]]
        arrow_points = np.dot(R, arrow_points.T).T + suction_point[np.newaxis,:]

        arrow.vertices = o3d.utility.Vector3dVector(arrow_points)
        arrows.append(arrow)

    o3d.visualization.draw_geometries([pc, *arrows], width=1500)
