import os
import numpy as np
import open3d as o3d
from transforms3d.euler import euler2mat, quat2mat
from utils.xmlhandler import xmlReader
from PIL import Image
from multiprocessing import Process
import scipy.io as scio
from utils.rotation import viewpoint_params_to_matrix, viewpoint_to_matrix


DATASET_ROOT = '/DATA2/Benchmark/graspnet'
labeldir = '/DATA1/hanwen/grasping/annotation_v4_10w/radius_1cm/poisson'
save_dir = '/DATA1/hanwen/grasping/scene_collision_mask'
modeldir = os.path.join(DATASET_ROOT, 'models')
scenedir = os.path.join(DATASET_ROOT, 'scenes') + '/scene_{}/{}'


def log_string(out_str):
    LOG.write(out_str+'\n')
    LOG.flush()
    print(out_str)

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


def parse_posevector(posevector):
    mat = np.zeros([4,4],dtype=np.float32)
    alpha, beta, gamma = posevector[4:7]
    alpha = alpha / 180.0 * np.pi
    beta = beta / 180.0 * np.pi
    gamma = gamma / 180.0 * np.pi
    mat[:3,:3] = euler2mat(alpha, beta, gamma)
    mat[:3,3] = posevector[1:4]
    mat[3,3] = 1
    # obj_idx = id_scene2obj(int(posevector[0]))
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


def create_mesh_cylinder(radius, height, R, t, collision):
    cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
    vertices = np.asarray(cylinder.vertices)
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


def get_scene(scene_idx, anno_idx, align=False, camera='realsense'):
    camera_split = 'data' if camera == 'realsense' else 'data_kinect'
    colors = np.array(Image.open(os.path.join(scenedir.format(scene_idx, camera_split), 'rgb', '%04d.png'%anno_idx)), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(scenedir.format(scene_idx, camera_split), 'depth', '%04d.png'%anno_idx)))
    intrinsics = np.load(os.path.join(scenedir.format(scene_idx, camera_split), 'camK.npy'))
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    s = 1000.0
    camera_poses = np.load(os.path.join('camera_poses', '{}_pose.npy'.format(camera)))
    camera_pose = camera_poses[anno_idx]
    if align:
        align_mat = np.load(os.path.join('camera_poses', '{}_alignment.npy'.format(camera)))
        camera_pose = align_mat.dot(camera_pose)

    xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths / s
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask]
    colors = colors[mask]
    points = transform_points(points, camera_pose)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud


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


def adjust_world_coordinate(scene_idx, anno_idx, rx=0, ry=0, rz=0, camera='realsense', remove_outlier=True):
    scene_cloud = get_scene(scene_idx, anno_idx, camera)
    box = create_mesh_box(0.5, 0.01, 0.5, dx=-0.25, dy=-0.25, dz=0.)
    scene_points = np.array(scene_cloud.points)
    trans = transform_matrix(0, 0, -0.477, rx, ry, rz)
    scene_points = transform_points(scene_points, trans)
    if remove_outlier:
        mask = (scene_points[:, 2] < 0)
        scene_points = scene_points[mask]
        colors = np.array(scene_cloud.colors)
        colors = colors[mask]
        scene_cloud.colors = o3d.utility.Vector3dVector(colors)
        print(scene_points.shape[0])
    scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
    print(trans)
    np.save('{}_calibration.npy'.format(camera), trans)
    o3d.visualization.draw_geometries([scene_cloud, box])


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
        points, normals, _, collision = get_model_grasps('%s/%03d_labels.npz'%(labeldir, obj_idx))
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
        # # sample workspace
        # o3dcloud = o3d.PointCloud()
        # o3dcloud.points = o3d.Vector3dVector(workspace)
        # o3dcloud = o3d.voxel_down_sample(o3dcloud, sample_size)
        # workspace = np.array(o3dcloud.points, dtype=np.float32)
        # print(workspace.shape[0])
        
        # # remove empty grasp points
        # min_dists = compute_min_dist(points, workspace)
        # points_in_scene = (min_dists < 0.01)
        
        collision_mask = collision.astype(np.bool)
        # print(points.shape)
        # print(collision_mask.shape)
        log_string('Suction points: ' + str(points.shape))
        log_string('Collision masks: ' + str(collision_mask.shape))
        for j, grasp_point in enumerate(points):
            print('Scene-' + str(scene_idx) + ' Obj-' + str(obj_idx) + ' ' + '{}/{}'.format(j,len(points)))
            # log_string('Scene-' + str(scene_idx) + ' Obj-' + str(obj_idx) + ' ' + '{}/{}'.format(j,len(points)))
            # if not points_in_scene[j]:
            #     collision_mask[j] = True
            #     continue
            
            grasp_poses = viewpoint_to_matrix(normals[j])
            target = workspace-grasp_point
            target = np.matmul(target, grasp_poses)
            # target = target.reshape([num_views, num_angles, num_depths, -1, 3])
            
            target_yz = target[:, 1:3]
            target_r = np.linalg.norm(target_yz, axis=-1)
            mask1 = target_r < radius
            mask2 = ((target[:,0] > 0.005) & (target[:,0] < height))
            
            mask = np.any(mask1 & mask2)
            collision_mask[j] = (collision_mask[j] | mask)

        collision_masks.append(collision_mask)

    np.savez('{}/{}_{}_collision.npz'.format(save_dir, scene_idx, camera), *collision_masks)


if __name__ == '__main__':
    # scene_idx = 119
    anno_idx = 0
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    camera = 'realsense'
    # camera = 'kinect'
    
    LOG = open('log_scene_' + camera + '_Jul16.txt', 'w')
    pool_size = 30
    # pending_list = [81, 126, 177]
    scene_list = [51, 116, 129, 130]

    # scene_list = []
    # for i in range(160, 190):
    #     scene_list.append(i)
    
    # scene_list = [50, 51, 52, 53, 54, 55, 56, 78, 115, 116, 119, 120, 145, 175] # kinect
    # scene_list = [34, 35, 39, 137]  # realsense

    pool_size = min(pool_size, len(scene_list))
    pool = []
    for _ in range(pool_size):
        scene_idx = scene_list.pop(0)
        # save_dir = '/data/fred/graspnet/scene_mask2/scene_{}'.format(scene_idx)
        pool.append(Process(target=scene_collision_detection, args=(scene_idx,anno_idx,save_dir,0.008,0.11,camera)))
    [p.start() for p in pool]
    while len(scene_list) > 0:
        for idx, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(idx)
                scene_idx = scene_list.pop(0)
                # save_dir = '/data/fred/graspnet/scene_mask2/scene_{}'.format(scene_idx)
                p = Process(target=scene_collision_detection, args=(scene_idx,anno_idx,save_dir,0.008,0.11,camera))
                p.start()
                pool.append(p)
                break
    while len(pool) > 0:
        for idx, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(idx)
                break

    # for scene_idx in scene_list:
    #     # if scene_idx in invalid_scenes:
    #     #     continue
    #     scene_collision_detection(scene_idx, anno_idx, save_dir, sample_size=0.008, outlier=0.11, camera=camera)
        
    
