import os
import numpy as np
import open3d as o3d
import argparse
from transforms3d.euler import euler2mat
from utils.xmlhandler import xmlReader
from utils.rotation import viewpoint_to_matrix


def parse_posevector(posevector):
    mat = np.zeros([4,4],dtype=np.float32)
    alpha, beta, gamma = posevector[4:7]
    alpha = alpha / 180.0 * np.pi
    beta = beta / 180.0 * np.pi
    gamma = gamma / 180.0 * np.pi
    mat[:3,:3] = euler2mat(alpha, beta, gamma)
    mat[:3,3] = posevector[1:4]
    mat[3,3] = 1
    obj_idx = int(posevector[0])
    return obj_idx, mat


def generate_views(N, phi=(np.sqrt(5)-1)/2, center=np.zeros(3), R=1):
    points = []
    for i in range(N):
        zi = (2 * i + 1) / N - 1
        xi = np.sqrt(1 - zi**2) * np.cos(2 * i * np.pi * phi)
        yi = np.sqrt(1 - zi**2) * np.sin(2 * i * np.pi * phi)
        points.append([xi, yi, zi])
    points = R * np.array(points) + center
    return points


def transform_points(points, trans):
    ones = np.ones([points.shape[0],1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    points_ = np.matmul(trans, points_.T).T
    return points_[:,:3]


def get_model_grasps(datapath):
    dump = np.load(datapath)
    points = dump['points']
    normals = dump['normals']
    scores = dump['scores']
    collision = dump['collision']
    return points, normals, scores, collision


def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,height],
                         [width,0,height],
                         [0,depth,0],
                         [width,depth,0],
                         [0,depth,height],
                         [width,depth,height]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box


def create_mesh_cylinder(radius, height, R, t, collision):
    cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
    vertices = np.asarray(cylinder.vertices)[:, [2, 1, 0]]
    vertices[:, 0] += height / 2
    vertices = np.dot(R, vertices.T).T + t
    cylinder.vertices = o3d.utility.Vector3dVector(vertices)
    if collision:
        colors = np.array([0.7, 0, 0])
    else:
        colors = np.array([0, 0.7, 0])
    colors = np.expand_dims(colors, axis=0)
    colors = np.repeat(colors, vertices.shape[0], axis=0)
    cylinder.vertex_colors = o3d.utility.Vector3dVector(colors)

    return cylinder


def create_mesh_ball(R, t, collision):
    ball = o3d.geometry.TriangleMesh().create_sphere(radius=0.001)
    vertices = np.asarray(ball.vertices)
    vertices = np.dot(R, vertices.T).T + t
    ball.vertices = o3d.utility.Vector3dVector(vertices)
    if collision:
        colors = np.array([0.7, 0, 0])
    else:
        colors = np.array([0, 0.7, 0])
    colors = np.expand_dims(colors, axis=0)
    colors = np.repeat(colors, vertices.shape[0], axis=0)
    ball.vertex_colors = o3d.utility.Vector3dVector(colors)

    return ball


def create_table_cloud(width, height, depth, dx=0, dy=0, dz=0, grid_size=0.01):
    xmap = np.linspace(0, width, int(width/grid_size))
    ymap = np.linspace(0, depth, int(depth/grid_size))
    zmap = np.linspace(0, height, int(height/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, -ymap, -zmap], axis=-1)
    points = points.reshape([-1, 3])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def plot_sucker(radius, height, R, t, collision):
    '''
        center: target point
        R: rotation matrix
    '''
    return create_mesh_cylinder(radius, height, R, t, collision)


def generate_scene_model(dataset_root, scene_name, anno_idx, return_poses=False, align=False, camera='realsense'):

    if align:
        print('align')
        camera_poses = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'camera_poses.npy'))
        camera_pose = camera_poses[anno_idx]
        align_mat = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
        camera_pose = np.matmul(align_mat,camera_pose)
    print('Scene {}, {}'.format(scene_name, camera))
    scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', scene_name, camera, 'annotations', '%04d.xml'%anno_idx))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    mat_list = []
    model_list = []
    pose_list = []
    for posevector in posevectors:
        obj_idx, pose = parse_posevector(posevector)
        obj_list.append(obj_idx)
        mat_list.append(pose)

    for obj_idx, pose in zip(obj_list, mat_list):
        plyfile = os.path.join(dataset_root, 'models', '%03d'%obj_idx, 'nontextured.ply')
        model = o3d.io.read_point_cloud(plyfile)
        points = np.array(model.points)
        if align:
            pose = np.dot(camera_pose, pose)
        points = transform_points(points, pose)
        model.points = o3d.utility.Vector3dVector(points)
        model_list.append(model)
        pose_list.append(pose)

    if return_poses:
        return model_list, obj_list, pose_list
    else:
        return model_list


def transform_matrix(tx, ty, tz, rx, ry, rz):
    trans = np.eye(4)
    trans[:3,3] = np.array([tx, ty, tz])
    rot_x = np.array([[1,          0,           0],
                      [0, np.cos(rx), -np.sin(rx)],
                      [0, np.sin(rx),  np.cos(rx)]])
    rot_y = np.array([[ np.cos(ry), 0, np.sin(ry)],
                      [          0, 1,          0],
                      [-np.sin(ry), 0, np.cos(ry)]])
    rot_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                      [np.sin(rz),  np.cos(rz), 0],
                      [         0,           0, 1]])
    trans[:3,:3] = rot_x.dot(rot_y).dot(rot_z)
    return trans


def vis_collision_individual(scene_idx, anno_idx, camera, visu_num):
    scene_name = 'scene_%04d' % scene_idx
    model_list, obj_list, pose_list = generate_scene_model(DATASET_ROOT, scene_name, anno_idx, return_poses=True, camera=camera, align=True)
    table = create_table_cloud(1.0, 0.02, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.01)

    camera_poses = np.load(os.path.join(DATASET_ROOT, 'scenes', scene_name, camera, 'camera_poses.npy'.format(camera)))
    camera_pose = camera_poses[anno_idx]
    table.points = o3d.utility.Vector3dVector(transform_points(np.asarray(table.points), camera_pose))
    
    collision_dump = np.load(os.path.join(collisiondir, '{}_{}_collision.npz'.format(scene_idx, camera)))

    radius = 0.01
    height = 0.1

    num_obj = len(obj_list)
    for obj_i in range(len(obj_list)):
        print('Checking ' + str(obj_i+1) + ' / ' + str(num_obj))
        obj_idx = obj_list[obj_i]
        trans = pose_list[obj_i]
        sampled_points, normals, scores, _ = get_model_grasps('%s/%03d_labels.npz'%(sealdir, obj_idx))
        collision = collision_dump['arr_{}'.format(obj_i)]

        point_inds = np.random.choice(sampled_points.shape[0], visu_num)
        np.random.shuffle(point_inds)
        suckers = []
        sucker_params = []

        for point_ind in point_inds:
            target_point = sampled_points[point_ind]
            normal = normals[point_ind]
            
            R = viewpoint_to_matrix(normal)
            t = transform_points(target_point[np.newaxis,:], trans).squeeze()
            R = np.dot(trans[:3,:3], R)
            sucker = plot_sucker(radius, height, R, t, collision[point_ind])
            suckers.append(sucker)
            sucker_params.append([target_point[0],target_point[1],target_point[2],normal[0],normal[1],normal[2],radius, height])
            
        o3d.visualization.draw_geometries([table, *model_list, *suckers], width=1536, height=864)


def vis_collision_all(scene_idx, anno_idx, camera, visu_num):
    scene_name = 'scene_%04d' % scene_idx
    model_list, obj_list, pose_list = generate_scene_model(DATASET_ROOT, scene_name, anno_idx, return_poses=True, camera=camera, align=True)
    table = create_table_cloud(1.0, 0.02, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.01)

    camera_poses = np.load(os.path.join(DATASET_ROOT, 'scenes', scene_name, camera, 'camera_poses.npy'.format(camera)))
    camera_pose = camera_poses[anno_idx]
    table.points = o3d.utility.Vector3dVector(transform_points(np.asarray(table.points), camera_pose))
    
    collision_dump = np.load(os.path.join(collisiondir, '{}_{}_collision.npz'.format(scene_idx, camera)))

    radius = 0.01
    height = 0.1

    num_obj = len(obj_list)
    suckers = []
    for obj_i in range(len(obj_list)):
        print('Checking ' + str(obj_i+1) + ' / ' + str(num_obj))
        obj_idx = obj_list[obj_i]
        trans = pose_list[obj_i]
        sampled_points, normals, scores, _ = get_model_grasps('%s/%03d_labels.npz'%(sealdir, obj_idx))
        collision = collision_dump['arr_{}'.format(obj_i)]

        point_inds = np.random.choice(sampled_points.shape[0], visu_num)
        np.random.shuffle(point_inds)
        
        sucker_params = []

        for point_ind in point_inds:
            target_point = sampled_points[point_ind]
            normal = normals[point_ind]
            
            R = viewpoint_to_matrix(normal)
            t = transform_points(target_point[np.newaxis,:], trans).squeeze()
            R = np.dot(trans[:3,:3], R)
            sucker = plot_sucker(radius, height, R, t, collision[point_ind])
            suckers.append(sucker)
            sucker_params.append([target_point[0],target_point[1],target_point[2],normal[0],normal[1],normal[2],radius, height])
            
    o3d.visualization.draw_geometries([table, *model_list, *suckers], width=1536, height=864)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='', help='Directory of graspnet dataset')
parser.add_argument('--seal_dir', default='', help='Directory of seal annotation label')
parser.add_argument('--colli_dir', default='', help='Directory of the collision annotation results')
parser.add_argument('--camera', default='kinect', help='camera to use [default: kinect]')
parser.add_argument("--scene_idx", type=int, default=0, help='the index of scene to visualize [default: 0]')
parser.add_argument("--obj_idx", type=int, default=0, help='the index of the object to visualize [default: 0]')
parser.add_argument("--visu_num", type=int, default=10, help='the number of suctions to visualize on each object')
parser.add_argument('--visu_each', action='store_true', help='whether to each object in a scene.')
args = parser.parse_args()


DATASET_ROOT = args.dataset_root
sealdir = args.seal_dir
modeldir = os.path.join(DATASET_ROOT, 'models')
collisiondir = args.colli_dir
scene_idx = args.scene_idx
anno_idx = 0 # this doesn't matter since we annotate in 3D so the global pose are the same for all views (anno_idxs)
visu_num = args.visu_num
obj_i = args.obj_idx
camera = args.camera


if __name__ == '__main__':
    
    if not args.visu_each:
        vis_collision_individual(scene_idx, anno_idx, camera, visu_num)
    else:
        vis_collision_all(scene_idx, anno_idx, camera, visu_num)
