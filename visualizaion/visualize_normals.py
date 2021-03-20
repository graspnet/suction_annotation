import numpy as np
import open3d as o3d
import os
import random
from transforms3d.euler import euler2mat, quat2mat

# MODEL_ROOT = r'G:\MyProject\data\Grasping\models_new'
MODEL_ROOT = r'G:\MyProject\data\Grasping\graspnet\models'
POISSON_ROOT = r'G:\MyProject\data\Grasping\graspnet\models_poisson'
COMPUTE_NORMALS = ['071', '070', '056', '058', '045', '029', '055', '020', '019', '018']
SUCTION_NUM = 30


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
    
    # mesh_vis = o3d.io.read_triangle_mesh(obj_dir)
    # mesh = o3d.io.read_triangle_mesh(obj_dir)
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

if __name__ == "__main__":
    # 019, 20, 70, 71, 
    # models = [ '029_new', '046_new', '056_new', '057_new', '059_new']
    # models = ['029', '046', '056', '057', '059']
    
    # models = []
    # for obj_idx in range(75, 88):
    #     model = '%03d' % obj_idx
    #     models.append(model)

    models = ['045']
    for model in models:
    
        # if model in COMPUTE_NORMALS:
        #     continue
        
        print(model)
        
        obj_dir = os.path.join(MODEL_ROOT, model, 'textured.obj')
        # ply_dir = os.path.join(MODEL_ROOT, model, 'nontextured.ply')
        # print(os.path.join(obj_dir))
        
        # # mesh_vis = o3d.io.read_triangle_mesh(obj_dir)
        mesh = o3d.io.read_triangle_mesh(obj_dir)
        # pc = o3d.io.read_point_cloud(ply_dir)

        # points = np.asarray(pc.points)

        # mat = np.zeros([4,4],dtype=np.float32)
        
        # alpha = random.randint(0, 180) / 180.0 * np.pi
        # beta = random.randint(0, 180) / 180.0 * np.pi
        # gamma = random.randint(0, 180) / 180.0 * np.pi
        # mat[:3,:3] = euler2mat(alpha, beta, gamma)
        # mat[0,3] = random.randint(0, 180) / 180.0
        # mat[1,3] = random.randint(0, 180) / 180.0
        # mat[2,3] = random.randint(0, 180) / 180.0
        # mat[3,3] = 1

        # points = transform_points(points, mat)
        # pc.points =  o3d.utility.Vector3dVector(points)

        # pc, mat = trans_model(model)
        # points = np.array(pc.points)

        # normals = np.asarray(pc.normals)

        # normals = transform_normals(normals, mat)
        
        # # points = np.load(point_dir)
        # # normals = np.load(normal_dir)
        # # print(points.shape)
        # # print(normals.shape)
        # # print(np.isnan(np.linalg.norm(points, axis=-1)).sum())
        # # print(np.isnan(np.linalg.norm(normals, axis=-1)).sum())
        # # index = np.random.choice(points.shape[0], 15000)
        # # points = points[index]
        # # normals = normals[index]

        # # pc = o3d.geometry.PointCloud()
        # # pc.points = o3d.utility.Vector3dVector(points)
        # # pc.normals = o3d.utility.Vector3dVector(normals)
        # # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc)

        # suction_index = np.random.choice(points.shape[0], SUCTION_NUM)
        # suction_points = points[suction_index]
        # suction_normals = normals[suction_index]

        # arrows = []

        # for i in range(suction_points.shape[0]):
        #     suction_point = suction_points[i]
        #     suction_normal = suction_normals[i]
        #     if np.isnan(np.sum(suction_normal)):
        #         print('Nan')
        #         continue
        #     if np.sum(suction_normal) == 0:
        #         print('Zero')
        #         continue
        #     # print(np.linalg.norm(suction_normal))
        #     suction_normal = suction_normal / np.linalg.norm(suction_normal)
        

        #     arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.001, cone_radius=0.0015, 
        #                                                     cylinder_height=0.005, cone_height=0.004)
        #     arrow_points = np.asarray(arrow.vertices)

        #     new_z = suction_normal
        #     new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
        #     new_y = new_y / np.linalg.norm(new_y)
        #     new_x = np.cross(new_y, new_z)

        #     R = np.c_[new_x, np.c_[new_y, new_z]]
        #     arrow_points = np.dot(R, arrow_points.T).T + suction_point[np.newaxis,:]

        #     arrow.vertices = o3d.utility.Vector3dVector(arrow_points)
        #     arrows.append(arrow)

        # # o3d.io.write_triangle_mesh('test.ply', arrow)
        # # o3d.visualization.draw_geometries([mesh, *arrows], width=1500)
        # o3d.visualization.draw_geometries([pc, *arrows], width=1500)
        o3d.visualization.draw_geometries([mesh], width=1500)

