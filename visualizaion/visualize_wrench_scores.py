import os
import argparse
import numpy as np
import open3d as o3d
from transforms3d.euler import euler2mat
from utils.xmlhandler import xmlReader
from utils.rotation import viewpoint_to_matrix


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

def generate_scene_model(dataset_root, scene_name, anno_idx, return_poses=False, 
                            align=False, camera='realsense'):

    if align:
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


def create_table_cloud(width, height, depth, dx=0, dy=0, dz=0, grid_size=0.01):
    xmap = np.linspace(0, width, int(width/grid_size))
    ymap = np.linspace(0, depth, int(depth/grid_size))
    zmap = np.linspace(0, height, int(height/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, ymap, zmap], axis=-1)
    points = points.reshape([-1, 3])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def create_mesh_cylinder(radius, height, R, t, score):
    cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
    vertices = np.asarray(cylinder.vertices)[:, [2, 1, 0]]
    vertices[:, 0] += height / 2
    vertices = np.dot(R, vertices.T).T + t
    cylinder.vertices = o3d.utility.Vector3dVector(vertices)
    
    colors = np.array([score, 0, 0])
    colors = np.expand_dims(colors, axis=0)
    colors = np.repeat(colors, vertices.shape[0], axis=0)
    cylinder.vertex_colors = o3d.utility.Vector3dVector(colors)

    return cylinder


def plot_sucker(radius, height, R, t, score):
    '''
        center: target point
        R: rotation matrix
    '''
    return create_mesh_cylinder(radius, height, R, t, score)


def visu_score_gravity(scene_idx, anno_idx, camera, visu_num):
    
    radius = 0.002
    height = 0.05

    scene_name = 'scene_%04d' % scene_idx
    model_list, obj_list, pose_list = generate_scene_model(DATASET_ROOT, scene_name, anno_idx, 
                                        return_poses=True, align=True, camera=camera)
    
    table = create_table_cloud(1.0, 0.01, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.01)
    camera_poses = np.load(os.path.join(DATASET_ROOT, 'scenes', scene_name, camera, 'camera_poses.npy'))
    camera_pose = camera_poses[anno_idx]
    
    table.points = o3d.utility.Vector3dVector(transform_points(np.asarray(table.points), camera_pose))
    
    wrench_dump = np.load(os.path.join(wrenchdir, '{}_{}_wrench_scores.npz'.format(scene_idx, camera)))
    num_obj = len(obj_list)
    
    for obj_i in range(len(obj_list)):
        print('Checking ' + str(obj_i+1) + ' / ' + str(num_obj))
        obj_idx = obj_list[obj_i]
        print('object id:', obj_idx)
        trans = pose_list[obj_i]
        sampled_points, normals, _, _ = get_model_grasps('%s/%03d_labels.npz'%(sealdir, obj_idx))
        sampled_points = transform_points(sampled_points, trans)
        center = np.mean(sampled_points, axis=0)
        score = wrench_dump['arr_{}'.format(obj_i)]

        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.015, 
                                                            cylinder_height=0.2, cone_height=0.04)
        arrow_points = np.asarray(arrow.vertices)
        arrow_points[:, 2] = -arrow_points[:, 2]
        arrow_points = arrow_points + center[np.newaxis,:]
        arrow.vertices = o3d.utility.Vector3dVector(arrow_points)
        
        point_inds = np.random.choice(sampled_points.shape[0], visu_num)
        np.random.shuffle(point_inds)
        suckers = []

        for point_ind in point_inds:
            target_point = sampled_points[point_ind]
            normal = normals[point_ind]
            
            R = viewpoint_to_matrix(normal)
            t = target_point
            R = np.dot(trans[:3,:3], R)
            sucker = plot_sucker(radius, height, R, t, score[point_ind])
            suckers.append(sucker)
            
        o3d.visualization.draw_geometries([table, *model_list, *suckers, arrow], width=1536, height=864)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='', help='Directory of graspnet dataset')
parser.add_argument('--seal_dir', default='', help='Directory of seal annotation label')
parser.add_argument('--wrench_dir', default='', help='Directory of the wrench annotation results')
parser.add_argument('--camera', default='kinect', help='camera to use [default: kinect]')
parser.add_argument("--scene_idx", type=int, default=0, help='the index of scene to visualize [default: 0]')
parser.add_argument("--visu_num", type=int, default=50, help='the number of suctions to visualize')
args = parser.parse_args()


DATASET_ROOT = args.dataset_root
sealdir = args.seal_dir
modeldir = os.path.join(DATASET_ROOT, 'models')
wrenchdir = args.wrench_dir
scene_idx = args.scene_idx
anno_idx = 0 # this doesn't matter since we annotate in 3D so the global pose are the same for all views (anno_idxs)
camera = args.camera
visu_num = args.visu_num


if __name__ == "__main__":
    visu_score_gravity(scene_idx, anno_idx, camera, visu_num)

