import numpy as np
import open3d as o3d
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--model_root', default='', help='Directory of graspnet models')
parser.add_argument('--poisson_root', default='', help='Directory of point clouds generated by poisson disk sampling from mesh') # https://github.com/fwilliams/point-cloud-utils
parser.add_argument('--save_root', default='', help='Directory to save the annotation results')
parser.add_argument('--voxel_size', type=float, default=0.005, help='voxel size (meter) to sample suction points [default: 0.005]')
args = parser.parse_args()


ORIGIN_MODEL_ROOT = args.model_root
POISSON_MODEL_ROOT = args.poisson_root
SAVE_ROOT = args.save_root
USE_POISSON = False
if POISSON_MODEL_ROOT != '':
      USE_POISSON = True
VOXEL_SIZE = args.voxel_size


if __name__ == "__main__":
      radius = 0.01
      num_split = 72
      radian_bins = []
      x = []
      y = []
      ideal_vertices = []
      
      for i in range(num_split):
            radian = 2 * np.pi * i / num_split - np.pi
            radian_bins.append(radian)
            x.append(radius * np.cos(radian + np.pi / num_split))
            y.append(radius * np.sin(radian + np.pi / num_split))
            ideal_vertices.append(np.array([x[-1], y[-1], 0]))
      
      radian_bins.append(np.pi)

      ideal_lengths = []
      for j in range(1, len(ideal_vertices)):
            ideal_lengths.append(np.linalg.norm(ideal_vertices[j] - ideal_vertices[j-1]))
      ideal_lengths.append(np.linalg.norm(ideal_vertices[0] - ideal_vertices[-1]))

      models = []
      for i in range(88):
            models.append('%03d' % i)
      
      for model in models:
            ply_dir = os.path.join(ORIGIN_MODEL_ROOT, model, 'nontextured.ply')
            print('Reading object from:', ply_dir)
            
            mesh = o3d.io.read_triangle_mesh(ply_dir)
            origin_points = np.asarray(mesh.vertices)
            pc = o3d.io.read_point_cloud(ply_dir)
            normals = np.asarray(mesh.vertex_normals)
            
            if not USE_POISSON:
                  points = np.asarray(mesh.vertices)
            if USE_POISSON:                 
                  poisson_dir = os.path.join(POISSON_MODEL_ROOT, model + '.npz')     
                  data = np.load(poisson_dir)
                  points = data['points']
            
            if normals.shape[0] == 0:
                  print('No normals.')
                  continue

            sample_size = VOXEL_SIZE
            min_bound = np.array([[origin_points[:, 0].min()], [origin_points[:, 1].min()], [origin_points[:, 2].min()]]).astype(np.float64)
            max_bound = np.array([[origin_points[:, 0].max()], [origin_points[:, 1].max()], [origin_points[:, 2].max()]]).astype(np.float64)
            sampled_pc, sampled_idx = pc.voxel_down_sample_and_trace(sample_size, min_bound, max_bound)
            idx1, idx2 = np.where(sampled_idx != -1)
            
            _, unique_idx = np.unique(idx1, return_index=True)
            sampled_idx = sampled_idx[idx1[unique_idx], idx2[unique_idx]]
            suction_index = np.unique(sampled_idx)
            suction_points = origin_points[suction_index]
            suction_normals = normals[suction_index]

            suction_points = np.expand_dims(suction_points, axis=1)
            indices = []
            scores = []
            collisions = []
            cnt = 0
            cnt_2 = 0
            cnt_3 = 0
            
            for suction_idx in range(suction_points.shape[0]):
                  suction_failed = False
                  suction_point = suction_points[suction_idx]
                  n = suction_normals[suction_idx]
                  n = n / np.linalg.norm(n)
                  
                  if np.isnan(np.sum(n)):
                        print('Nan normal')
                        continue
                  if np.sum(n) == 0:
                        print('Zero normal')
                        continue

                  new_z = n
                  new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
                  new_y = new_y / np.linalg.norm(new_y)

                  new_x = np.cross(new_y, new_z)
                  new_x = new_x / np.linalg.norm(new_x)

                  new_x = np.expand_dims(new_x, axis=1)
                  new_y = np.expand_dims(new_y, axis=1)
                  new_z = np.expand_dims(new_z, axis=1)

                  new_coords = np.concatenate((new_x, new_y, new_z), axis=-1)
                  rot_matrix = new_coords
                  translated_points = points - suction_point
                  transformed_points = np.dot(translated_points, rot_matrix)
                  
                  transformed_xy = transformed_points[:, 0:2]
                  dist = np.linalg.norm(transformed_xy, axis=-1)
                  
                  mask_above = transformed_points[:, 2] > 0.005
                  mask_wi_radius = dist < radius
                  collision_mask = mask_above & mask_wi_radius
                  collision = np.any(collision_mask)
                  collisions.append(collision)

                  indices.append(suction_index[suction_idx])

                  idx = np.where((dist > radius - 0.001) & (dist < radius + 0.001))[0]
                  transformed_points = transformed_points[idx]                  
                  
                  if collision:
                        cnt_3 += 1

                  if transformed_points.shape[0] == 0:
                        suction_failed = True
                  else:
                        topview_radian = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
                        if topview_radian.max() - topview_radian.min() < 1.9 * np.pi:
                              cnt_2 += 1
                        bin_points_list = []
                        real_vertices = []
                        
                        for j in range(1, len(radian_bins)):
                              l_limit = radian_bins[j - 1]
                              r_limit = radian_bins[j]

                              bin_points = transformed_points[np.where((topview_radian > l_limit) & (topview_radian < r_limit))]
                              if bin_points.shape[0] == 0:
                                    suction_failed = True
                                    cnt += 1
                                    break
                              bin_points_list.append(bin_points)
                              real_vertices.append(np.array([x[j-1], y[j-1], bin_points[:, 2].max()]))
                  
                  if suction_failed:
                        quality = 0
                        scores.append(quality)
                  else:
                        real_lengths = []
                        for j in range(1, len(real_vertices)):
                              real_lengths.append(np.linalg.norm(real_vertices[j] - real_vertices[j-1]))
                        real_lengths.append(np.linalg.norm(real_vertices[0] - real_vertices[-1]))

                        assert len(ideal_lengths) == len(real_lengths)

                        deform_ratios = []
                        for i in range(len(ideal_lengths)):
                              deform_ratios.append(ideal_lengths[i] / real_lengths[i])
                        
                        deform = min(deform_ratios)
                        
                        vertices = np.vstack(real_vertices)
                        A = np.zeros((vertices.shape[0], 2), dtype=np.float32)
                        ones = np.ones((A.shape[0], 1))
                        A = np.hstack((A, ones))
                        b = vertices[:, 2]

                        w, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)
                        s = (1.0 / A.shape[0]) * np.square(np.linalg.norm(np.dot(A, w) - b)) * 100000
                        # print('sse: ', s)
                        fit = np.exp(-s)
                        # quality = deform
                        quality = fit * deform
                        scores.append(quality)

            print('Failed grasping count:', cnt)
            print('Less than 1.9Pi count:', cnt_2)
            print('Collision count:', cnt_3)
            
            suction_points = np.squeeze(suction_points)
            suction_index = np.array(indices, dtype=np.int32)
            suction_points = origin_points[suction_index]
            suction_normals = normals[suction_index]

            if not USE_POISSON:
                  save_dir = os.path.join(SAVE_ROOT, 'origin')
                  os.makedirs(save_dir, exist_ok=True)
                  print('Saving:' + '{}/{}_labels.npz'.format(save_dir, model))
                  print()
                  np.savez('{}/{}_labels.npz'.format(save_dir, model),
                        points=suction_points,
                        normals=suction_normals,
                        indices=suction_index,
                        collision=np.array(collisions, dtype=np.bool),
                        scores=np.array(scores, dtype=np.float32))
            else:
                  save_dir = os.path.join(SAVE_ROOT, 'poisson')
                  os.makedirs(save_dir, exist_ok=True)
                  os.makedirs(save_dir, exist_ok=True)
                  print('Saving:' + '{}/{}_labels.npz'.format(save_dir, model))
                  print()
                  np.savez('{}/{}_labels.npz'.format(save_dir, model),
                        points=suction_points,
                        normals=suction_normals,
                        indices=suction_index,
                        collision=np.array(collisions, dtype=np.bool),
                        scores=np.array(scores, dtype=np.float32))
