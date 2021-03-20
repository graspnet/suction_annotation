import numpy as np
import open3d as o3d
import os


if __name__ == "__main__":
    model_root = r'G:\MyProject\data\Grasping\models'
    annotation_root = r'G:\MyProject\data\Grasping\annotation_v4_10w\radius_1cm\poisson'
    obj_idx = 60
    visu_num = 100 

    obj_dir = os.path.join(model_root, '%03d' % obj_idx, 'textured.obj')
    annotation_dir = os.path.join(annotation_root, '%03d_labels.npz' % obj_idx)
    mesh = o3d.io.read_triangle_mesh(obj_dir)
    annotation = np.load(annotation_dir)
    anno_points = annotation['points']
    anno_scores = annotation['scores']

    suction_index = np.random.choice(anno_points.shape[0], visu_num)
    suction_points = anno_points[suction_index]
    suction_scores = anno_scores[suction_index]

    # np.random.choice(points.shape[0], 25)
    # suction_idxs = np.where(anno_colors[:, 0] == 1)[0]
    # print(type(suction_idxs))
    vis_list = [mesh]
    # print('len:', len(suction_idxs[0]))
    for idx in range(len(suction_points)):
        # print(suction_idxs[idx])
        suction_point = suction_points[idx]
        # print('suction point:', suction_point.shape)
        suction_score = suction_scores[idx]
        ball = o3d.geometry.TriangleMesh.create_sphere(0.002).translate(suction_point)
        ball_v = np.asarray(ball.vertices)
        ball_colors = np.zeros((ball_v.shape[0], 3), dtype=np.float32)
        ball_colors[:, 0] = suction_score
        ball.vertex_colors = o3d.utility.Vector3dVector(ball_colors)
        vis_list.append(ball)
    
    o3d.visualization.draw_geometries(vis_list)

    # vertices = np.asarray(mesh.vertices)
    # print('Max x:', vertices[..., 0].max())
    # vec = np.array([vertices[..., 0].max()-0.003, 0, 0], dtype=np.float64)
    # ball = mesh.create_sphere(0.002).translate(vec)
    # vertices = np.asarray(ball.vertices)
    # print(vertices.shape)
    # colors = np.zeros((vertices.shape[0], 3), dtype=np.float32)
    # print(colors.shape)
    # colors[:, 0] = 1.0
    # ball.vertex_colors = o3d.utility.Vector3dVector(colors)
    # print(np.asarray(ball.vertex_colors))
    # o3d.visualization.draw_geometries([mesh, ball])

