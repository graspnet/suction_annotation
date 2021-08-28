import os
import numpy as np
import argparse
from transforms3d.euler import euler2mat
from utils.xmlhandler import xmlReader
from utils.rotation import viewpoint_to_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='', help='Directory of graspnet dataset')
parser.add_argument('--seal_dir', default='', help='Directory of seal annotation label')
parser.add_argument('--save_dir', default='', help='Directory to save wrench annotation')
parser.add_argument('--camera', default='kinect', help='camera to use [default: kinect]')
args = parser.parse_args()


DATASET_ROOT = args.dataset_root
sealdir = args.seal_dir
modeldir = os.path.join(DATASET_ROOT, 'models')
LOG = open('log_wrench.txt', 'w')
save_dir = args.save_dir
camera = args.camera
anno_idx = 0 # this doesn't matter since we annotate in 3D so the global pose are the same for all views (anno_idxs)


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


def wrench_annotate(dataset_root, scene_idx, anno_idx, save_dir, align=False, camera='realsense'):

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
    
    for posevector in posevectors:
        obj_idx, pose = parse_posevector(posevector)
        obj_list.append(obj_idx)
        mat_list.append(pose)

    g = 9.8

    wrench_scores = []
    for obj_idx, pose in zip(obj_list, mat_list):
        print('object id:', obj_idx)
        points, normals, _, _ = get_model_grasps('%s/%03d_labels.npz'%(sealdir, obj_idx))
        if align:
            pose = np.dot(camera_pose, pose)
        points = transform_points(points, pose)
        
        center = points.mean(axis=0)

        gravity = np.array([[0, 0, -1]], dtype=np.float32) * g

        single_scores = []
        for j, suction_point in enumerate(points):
            suction_axis = viewpoint_to_matrix(normals[j])
            suction2center = (center - suction_point)[np.newaxis, :]
            coord = np.matmul(suction2center, suction_axis)

            gravity_proj = np.matmul(gravity, suction_axis)

            torque_y = gravity_proj[0, 0] * coord[0, 2] - gravity_proj[0, 2] * coord[0, 0]
            torque_x = -gravity_proj[0, 1] * coord[0, 2] + gravity_proj[0, 2] * coord[0, 1]
            torque = np.sqrt(torque_x**2 + torque_y**2)

            score = 1 - min(1, torque / wrench_thre)

            single_scores.append(score)
        
        single_scores = np.array(single_scores, dtype=np.float32)
        wrench_scores.append(single_scores)
    
    print('saving:', '{}/{}_{}_wrench_scores.npz'.format(save_dir, scene_idx, camera))
    np.savez('{}/{}_{}_wrench_scores.npz'.format(save_dir, scene_idx, camera), *wrench_scores)


if __name__ == "__main__":

    for scene_idx in range(190):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        wrench_annotate(DATASET_ROOT, scene_idx, anno_idx, save_dir, align=True, camera=camera)

