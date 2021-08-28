import os
import numpy as np
import open3d as o3d
import argparse
from transforms3d.euler import euler2mat
from utils.xmlhandler import xmlReader
from multiprocessing import Process
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


def scene_collision_detection(scene_idx, anno_idx, save_dir, sample_size=0.005, outlier=0.08, camera='realsense'):
    scene_name = 'scene_%04d' % scene_idx
    model_list, obj_list, pose_list = generate_scene_model(DATASET_ROOT, scene_name, anno_idx, return_poses=True, align=True, camera=camera)
    table = create_table_cloud(1.0, 0.02, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.001)
    
    height = 0.1
    radius = 0.01
    collision_masks = []

    # merge scene
    scene = [np.array(table.points)]
    for model in model_list:
        scene.append(np.array(model.points))
    scene = np.concatenate(scene, axis=0)

    for i, (obj_idx, trans) in enumerate(zip(obj_list, pose_list)):
        # print(obj_idx)
        log_string('Obj-' + str(obj_idx))
        points, normals, _, collision = get_model_grasps('%s/%03d_labels.npz'%(sealdir, obj_idx))
        # collision = np.array(points.shape[0]).astype(np.bool)
        
        # crop scene
        scene_trans = transform_points(scene, np.linalg.inv(trans))
        xmin, xmax = points[:,0].min(), points[:,0].max()
        ymin, ymax = points[:,1].min(), points[:,1].max()
        zmin, zmax = points[:,2].min(), points[:,2].max()
        xlim = ((scene_trans[:,0] > xmin-outlier) & (scene_trans[:,0] < xmax+outlier))
        ylim = ((scene_trans[:,1] > ymin-outlier) & (scene_trans[:,1] < ymax+outlier))
        zlim = ((scene_trans[:,2] > zmin-outlier) & (scene_trans[:,2] < zmax+outlier))
        workspace = scene_trans[xlim & ylim & zlim]
        # # sample workspace. This can be used to speed up the annotation
        # o3dcloud = o3d.PointCloud()
        # o3dcloud.points = o3d.Vector3dVector(workspace)
        # o3dcloud = o3d.voxel_down_sample(o3dcloud, sample_size)
        # workspace = np.array(o3dcloud.points, dtype=np.float32)
        # print(workspace.shape[0])
        
        
        collision_mask = collision.astype(np.bool)
        # print(points.shape)
        # print(collision_mask.shape)
        log_string('Suction points: ' + str(points.shape))
        log_string('Collision masks: ' + str(collision_mask.shape))
        for j, grasp_point in enumerate(points):
            print('Scene-' + str(scene_idx) + ' Obj-' + str(obj_idx) + ' ' + '{}/{}'.format(j,len(points)))
            # log_string('Scene-' + str(scene_idx) + ' Obj-' + str(obj_idx) + ' ' + '{}/{}'.format(j,len(points)))
            
            grasp_poses = viewpoint_to_matrix(normals[j])
            target = workspace-grasp_point
            target = np.matmul(target, grasp_poses)
            
            target_yz = target[:, 1:3]
            target_r = np.linalg.norm(target_yz, axis=-1)
            mask1 = target_r < radius
            mask2 = ((target[:,0] > 0.005) & (target[:,0] < height))
            
            mask = np.any(mask1 & mask2)
            collision_mask[j] = (collision_mask[j] | mask)

        collision_masks.append(collision_mask)

    np.savez('{}/{}_{}_collision.npz'.format(save_dir, scene_idx, camera), *collision_masks)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='', help='Directory of graspnet dataset')
parser.add_argument('--seal_dir', default='', help='Directory of seal annotation label')
parser.add_argument('--save_dir', default='', help='Directory to save the annotation results')
parser.add_argument('--camera', default='kinect', help='camera to use [default: kinect]')
parser.add_argument("--pool_size", type=int, default=30)
args = parser.parse_args()


DATASET_ROOT = args.dataset_root
sealdir = args.seal_dir
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
modeldir = os.path.join(DATASET_ROOT, 'models')
scenedir = os.path.join(DATASET_ROOT, 'scenes') + '/scene_{}/{}'
camera = args.camera
anno_idx = 0 # this doesn't matter since we annotate in 3D so the global pose are the same for all views (anno_idxs)
pool_size = args.pool_size


LOG = open('log_scene_' + camera + '.txt', 'w')

def log_string(out_str):
    LOG.write(out_str+'\n')
    LOG.flush()
    print(out_str)

if __name__ == '__main__':

    scene_list = []
    for i in range(190):
        scene_list.append(i)
    
    pool_size = min(pool_size, len(scene_list))
    if pool_size > 1:
        # use multi-thread computation to speed up the annotation
        pool = []
        for _ in range(pool_size):
            scene_idx = scene_list.pop(0)
            pool.append(Process(target=scene_collision_detection, args=(scene_idx,anno_idx,save_dir,0.008,0.11,camera)))
        [p.start() for p in pool]
        while len(scene_list) > 0:
            for idx, p in enumerate(pool):
                if not p.is_alive():
                    pool.pop(idx)
                    scene_idx = scene_list.pop(0)
                    p = Process(target=scene_collision_detection, args=(scene_idx,anno_idx,save_dir,0.008,0.11,camera))
                    p.start()
                    pool.append(p)
                    break
        while len(pool) > 0:
            for idx, p in enumerate(pool):
                if not p.is_alive():
                    pool.pop(idx)
                    break
    else:
        # This is a single-thread version which is deprecated (too slow)
        for scene_idx in scene_list:
            scene_collision_detection(scene_idx, anno_idx, save_dir, sample_size=0.008, outlier=0.11, camera=camera)
        
    
