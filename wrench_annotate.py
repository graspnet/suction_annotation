import os
import numpy as np
import open3d as o3d
from transforms3d.euler import euler2mat, quat2mat
from utils.xmlhandler import xmlReader
from PIL import Image
from multiprocessing import Process
import scipy.io as scio
from utils.rotation import viewpoint_params_to_matrix, viewpoint_to_matrix
import csv


DATASET_ROOT = '/DATA2/Benchmark/graspnet'
labeldir = '/DATA1/hanwen/grasping/annotation_v4_10w/radius_1cm/poisson'
modeldir = os.path.join(DATASET_ROOT, 'models')
# scenedir = os.path.join(DATASET_ROOT, 'scenes') + '/scene_{}/{}'
LOG = open('log_wrench.txt', 'w')
save_dir = '/DATA1/hanwen/grasping/wrench_scoreV2'

k = 15.6
radius = 0.01
wrench_thre = k * radius * np.pi * np.sqrt(2)

def log_string(out_str):
    LOG.write(out_str+'\n')
    LOG.flush()
    print(out_str)


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


def wrench_annotate(dataset_root, scene_idx, anno_idx, weight_dict, save_dir,
                            return_poses=False, align=False, camera='realsense'):

    scene_name = 'scene_%04d' % scene_idx

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
    # model_list = []
    # pose_list = []
    
    for posevector in posevectors:
        obj_idx, pose = parse_posevector(posevector)
        obj_list.append(obj_idx)
        mat_list.append(pose)

    g = 9.8

    wrench_scores = []
    for obj_idx, pose in zip(obj_list, mat_list):
        print('object id:', obj_idx)
        points, normals, _, _ = get_model_grasps('%s/%03d_labels.npz'%(labeldir, obj_idx))
        if align:
            pose = np.dot(camera_pose, pose)
        points = transform_points(points, pose)
        
        center = points.mean(axis=0)

        weight = weight_dict[obj_idx]
        if weight is None:
            log_string('None Weight! in scene {} of object {}'.format(scene_idx, obj_idx))
            break
        
        gravity = np.array([[0, 0, -1]], dtype=np.float32) * g

        single_scores = []
        for j, suction_point in enumerate(points):
            suction_axis = viewpoint_to_matrix(normals[j])
            suction2center = (center - suction_point)[np.newaxis, :]
            coord = np.matmul(suction2center, suction_axis)

            gravity_proj = np.matmul(gravity, suction_axis)
            
            # print('gravity:', gravity_proj.shape)
            # print('coord:', coord.shape)

            torque_y = gravity_proj[0, 0] * coord[0, 2] - gravity_proj[0, 2] * coord[0, 0]
            torque_x = -gravity_proj[0, 1] * coord[0, 2] + gravity_proj[0, 2] * coord[0, 1]

            torque = np.sqrt(torque_x**2 + torque_y**2)

            # dist = np.linalg.norm(suction2center)
            # score = 1 / (1 + np.exp(-wrench_thre/torque_max + 1))
            # score = 1 / (1 + np.exp((torque_max - wrench_thre)*100))
            score = 1 - min(1, torque / wrench_thre)

            single_scores.append(score)
        
        single_scores = np.array(single_scores, dtype=np.float32)
        wrench_scores.append(single_scores)
    
    print('saving:', '{}/{}_{}_wrench_scores.npz'.format(save_dir, scene_idx, camera))
    np.savez('{}/{}_{}_wrench_scores.npz'.format(save_dir, scene_idx, camera), *wrench_scores)


if __name__ == "__main__":
    
    csvFile = open("weights.csv", "r", encoding="utf-8-sig")
    reader = csv.reader(csvFile)
    
    obj_dict = {}
    for id, item in enumerate(reader):
        # print(id, item)
        if item[0] != '':
            obj_dict[id] = float(item[0])
        elif item[1] != '':
            obj_dict[id] = float(item[1])
        else:
            obj_dict[id] = None
    
    anno_idx = 0
    camera = 'kinect'
    # camera = 'realsense'

    for scene_idx in range(190):
        # scene_idx = 30
        
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        wrench_annotate(DATASET_ROOT, scene_idx, anno_idx, obj_dict, save_dir, return_poses=True, align=True, camera=camera)

