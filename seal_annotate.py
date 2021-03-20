import numpy as np
import open3d as o3d
import os
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.use("TkAgg")

# ORIGIN_MODEL_ROOT = '/DATA1/hanwen/grasping/models'
ORIGIN_MODEL_ROOT = '/DATA2/Benchmark/graspnet/models'
POISSON_MODEL_ROOT = '/DATA1/hanwen/grasping/models_poisson'
# POISSON_MODEL_ROOT = '/DATA1/hanwen/grasping/poisson_old'
SAVE_ROOT = '/DATA1/hanwen/grasping/annotation_v4_10w/radius_1cm'

# ORIGIN_MODEL_ROOT = r'G:\MyProject\Grasping\models'
# POISSON_MODEL_ROOT = r'G:\MyProject\Grasping\models_poisson'
# SAVE_ROOT = r'G:\MyProject\Grasping\annotation_v3'

USE_POISSON = True
VOXEL_SIZE = 0.005

COMPUTE_NORMALS = ['071', '070', '056', '058', '045', '029', '055', '020', '019', '018']

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

      # models = os.listdir(ORIGIN_MODEL_ROOT)
      # models = ['018', '020', '022', '023']
      models = []
      for i in range(75, 88):
            models.append('%03d' % i)
      
      for model in models:
            if '0' not in model:
                  continue
            obj_dir = os.path.join(ORIGIN_MODEL_ROOT, model, 'textured.obj')
            ply_dir = os.path.join(ORIGIN_MODEL_ROOT, model, 'nontextured.ply')
            print('Processing:', obj_dir)
            mesh = o3d.io.read_triangle_mesh(ply_dir)
            origin_points = np.asarray(mesh.vertices)
            # pc = o3d.geometry.PointCloud(mesh.vertices)
            pc = o3d.io.read_point_cloud(ply_dir)

            # if model in COMPUTE_NORMALS:
            #       mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)
            
            if not USE_POISSON:
                  points = np.asarray(mesh.vertices)
                  
            if USE_POISSON:
                  # poisson_dir = os.path.join(POISSON_MODEL_ROOT, model)
                  # print('Processing:', poisson_dir)
                  # poisson_v_file = os.path.join(poisson_dir, 'points.npy')
                  # poisson_n_file = os.path.join(poisson_dir, 'normals.npy')
                  # points = np.load(poisson_v_file)                  
                  poisson_dir = os.path.join(POISSON_MODEL_ROOT, model + '.npz')     
                  data = np.load(poisson_dir)
                  points = data['points']
                  print('Poisson Point:', points.shape)
                  # normals = data['normals']

            # print('max x:', points[:, 0].max())
            # print('min x:', points[:, 0].min())

            # print('max y:', points[:, 1].max())
            # print('min y:', points[:, 1].min())

            # print('max z:', points[:, 2].max())
            # print('min z:', points[:, 2].min())
            
            if normals.shape[0] == 0:
                  print('No normals.')
                  continue
            
            # colors = np.array([144, 144, 144]) / 255
            # colors = np.expand_dims(colors, axis=0)
            # colors = np.repeat(colors, origin_points.shape[0], axis=0)

            sample_size = VOXEL_SIZE
            min_bound = np.array([[origin_points[:, 0].min()], [origin_points[:, 1].min()], [origin_points[:, 2].min()]]).astype(np.float64)
            max_bound = np.array([[origin_points[:, 0].max()], [origin_points[:, 1].max()], [origin_points[:, 2].max()]]).astype(np.float64)
            sampled_pc, sampled_idx = pc.voxel_down_sample_and_trace(sample_size, min_bound, max_bound)
            idx1, idx2 = np.where(sampled_idx != -1)
            
            _, unique_idx = np.unique(idx1, return_index=True)
            sampled_idx = sampled_idx[idx1[unique_idx], idx2[unique_idx]]
            suction_index = np.unique(sampled_idx)
            # print('suction idx:', suction_index.dtype)
            # suction_index = np.random.choice(origin_points.shape[0], 500)
            suction_points = origin_points[suction_index]
            suction_normals = normals[suction_index]

            suction_points = np.expand_dims(suction_points, axis=1)
            # expanded_points = np.expand_dims(points, axis=0)
            # dist = np.linalg.norm(suction_points - expanded_points, axis=-1)
            # print('dist: ', dist.shape)
            indices = []
            scores = []
            collisions = []
            cnt = 0
            cnt_2 = 0
            cnt_3 = 0
            
            for suction_idx in range(suction_points.shape[0]):
                  suction_failed = False
                  suction_point = suction_points[suction_idx]
                  # suction_point = np.squeeze(suction_point, axis=0)
                  # print('suction point:', np.squeeze(suction_point, axis=0))
                  # suction_point[:, 2] = 10
                  
                  # ring_points = points[idx] - suction_point
                  # print('ring: ', ring_points.shape)

                  n = suction_normals[suction_idx]
                  # print('normal:', n)
                  n = n / np.linalg.norm(n)

                  # dim = np.argmax(np.abs(n))
                  # if dim == 0:
                  #       topview_radian = np.arctan2(ring_points[:, 1], ring_points[:, 2])
                  #       if topview_radian.max() - topview_radian.min() < 1.9 * np.pi:
                  #             cnt_3 += 1
                  # if dim == 1:
                  #       topview_radian = np.arctan2(ring_points[:, 0], ring_points[:, 2])
                  #       if topview_radian.max() - topview_radian.min() < 1.9 * np.pi:
                  #             cnt_3 += 1
                  # if dim == 2:
                  #       topview_radian = np.arctan2(ring_points[:, 0], ring_points[:, 1])
                  #       if topview_radian.max() - topview_radian.min() < 1.9 * np.pi:
                  #             cnt_3 += 1

                  # seed_v = np.random.random(3)
                  # seed_v = seed_v / np.linalg.norm(seed_v)

                  new_z = n
                  # print(new_z.dtype)
                  # print('normal:', n)
                  if np.isnan(np.sum(n)):
                        # cnt_3 += 1
                        print('Nan normal')
                        continue
                  if np.sum(n) == 0:
                        # cnt_3 += 1
                        print('Zero normal')
                        continue

                  
                  new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
                  new_y = new_y / np.linalg.norm(new_y)

                  new_x = np.cross(new_y, new_z)
                  new_x = new_x / np.linalg.norm(new_x)

                  new_x = np.expand_dims(new_x, axis=1)
                  new_y = np.expand_dims(new_y, axis=1)
                  new_z = np.expand_dims(new_z, axis=1)

                  new_coords = np.concatenate((new_x, new_y, new_z), axis=-1)
                  rot_matrix = new_coords
                  # rot_matrix = np.linalg.inv(new_coords).T
                  # print('rot_matrix:', rot_matrix)
                  translated_points = points - suction_point
                  # print('translated_points:', translated_points.shape)
                  transformed_points = np.dot(translated_points, rot_matrix)
                  # print('max x:', transformed_points[:, 0].max())
                  # print('min x:', transformed_points[:, 0].min())

                  # print('max y:', transformed_points[:, 1].max())
                  # print('min y:', transformed_points[:, 1].min())

                  # print('max z:', transformed_points[:, 2].max())
                  # print('min z:', transformed_points[:, 2].min())
                  # transformed_points = rotated_points
                  # suction_xy = suction_point[:, 0:2]

                  transformed_xy = transformed_points[:, 0:2]
                  dist = np.linalg.norm(transformed_xy, axis=-1)
                  # print('dist:', dist.shape)
                  
                  mask_above = transformed_points[:, 2] > 0.005
                  mask_wi_radius = dist < radius
                  collision_mask = mask_above & mask_wi_radius
                  collision = np.any(collision_mask)
                  collisions.append(collision)
                  # indices.append(sampled_idx[suction_idx])
                  indices.append(suction_index[suction_idx])

                  idx = np.where((dist > radius - 0.001) & (dist < radius + 0.001))[0]
                  # print('idx:', idx.shape)
                  transformed_points = transformed_points[idx]                  
                  
                  if collision:
                        cnt_3 += 1

                  if transformed_points.shape[0] == 0:
                        suction_failed = True
                  else:
                        transformed_points[:, 2] -= 10
                        
                        # pc = o3d.geometry.PointCloud()
                        # pc.points = o3d.utility.Vector3dVector(ring_points)
                        # # pc.colors = o3d.utility.Vector3dVector(colors)
                        # o3d.visualization.draw_geometries([pc])

                        # pc = o3d.geometry.PointCloud()
                        # pc.points = o3d.utility.Vector3dVector(transformed_points)
                        # # pc.colors = o3d.utility.Vector3dVector(colors)
                        # o3d.visualization.dratransformed_pointsw_geometries([pc])
                        
                        # plt.scatter(transformed_points[:, 0], transformed_points[:, 1], alpha=0.6)
                        # plt.show()
                        topview_radian = np.arctan2(transformed_points[:, 1], transformed_points[:, 0])
                        # print(topview_radian.shape)
                        # print(topview_radian.min())
                        # print(topview_radian.max())
                        if topview_radian.max() - topview_radian.min() < 1.9 * np.pi:
                              cnt_2 += 1
                        bin_points_list = []
                        real_vertices = []
                        
                        for j in range(1, len(radian_bins)):
                              l_limit = radian_bins[j - 1]
                              r_limit = radian_bins[j]

                              bin_points = transformed_points[np.where((topview_radian > l_limit) & (topview_radian < r_limit))]
                              if bin_points.shape[0] == 0:
                                    # print('l_limit:', l_limit)
                                    # print('r_limit:', r_limit)
                                    suction_failed = True
                                    cnt += 1
                                    break
                              bin_points_list.append(bin_points)
                              real_vertices.append(np.array([x[j-1], y[j-1], bin_points[:, 2].max()]))
                  
                  if suction_failed:
                        quality = 0
                        scores.append(quality)
                        # colors[suction_index[suction_idx]] = np.array([quality, 0, 0])
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

                        # colors[suction_index[suction_idx]] = np.array([quality, 0, 0])

                        # if suction_point[0, 2] > -0.04 and suction_point[0, 2] < 0.03:
                        #       print('Deform:', deform)
                        #       print('Fit:', fit)
                        #       print('Quality:', quality)
                        #       print('W:', np.dot(A, w))
                        #       print('Projected:', b)
                        #       print('\n')
                        
                        # if quality > 0.9:
                        #     colors[suction_index[suction_idx]] = np.array([1, 0, 0])
                        # else:
                        #     colors[suction_index[suction_idx]] = np.array([0, 0, 0])

                  # if collision:
                  #       colors[suction_index[suction_idx]] = np.array([1, 0, 0])

            print('Failed grasping count:', cnt)
            print('Less than 1.9Pi count:', cnt_2)
            print('Collision count:', cnt_3)

            # pc = o3d.geometry.PointCloud()
            # pc.points = o3d.utility.Vector3dVector(origin_points)
            # pc.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([pc])
            
            suction_points = np.squeeze(suction_points)
            suction_index = np.array(indices, dtype=np.int32)
            suction_points = origin_points[suction_index]
            suction_normals = normals[suction_index]
            print('suction points:', suction_points.shape)
            print('normals:', suction_normals.shape)
            print('indices:', suction_index.shape)
            print('collision:', np.array(collisions, dtype=np.bool).shape)
            print('scores:', np.array(scores, dtype=np.float32).shape)
            # for i in range(sampled_idx.shape[0]):
            #       if sampled_idx[i] != suction_index[i]:
                  # print('sampled_idx:', sampled_idx[i])
                  # print('suction_index:', suction_index[i])
                        # print('Unequal index')
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
                  # save_file = os.path.join(save_dir, 'colored.pcd')
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
            
            
    




