import numpy as np
import argparse
import open3d as o3d
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', default='', help='Directory of graspnet models')
    parser.add_argument('--anno_root', default='', help='Directory of the annotation results')
    parser.add_argument('--obj_idx', type=int, default=0, help='index of the object to visualize [default: 0]')
    parser.add_argument('--visu_num', type=int, default=100, help='number of suction points to visualize [default: 100]')
    args = parser.parse_args()

    model_root = args.model_root
    annotation_root = args.anno_root
    obj_idx = args.obj_idx
    visu_num = args.visu_num

    obj_dir = os.path.join(model_root, '%03d' % obj_idx, 'textured.obj')
    annotation_dir = os.path.join(annotation_root, '%03d_labels.npz' % obj_idx)
    mesh = o3d.io.read_triangle_mesh(obj_dir)
    annotation = np.load(annotation_dir)
    anno_points = annotation['points']
    anno_scores = annotation['scores']

    suction_index = np.random.choice(anno_points.shape[0], visu_num)
    suction_points = anno_points[suction_index]
    suction_scores = anno_scores[suction_index]

    vis_list = [mesh]
    for idx in range(len(suction_points)):
        suction_point = suction_points[idx]
        suction_score = suction_scores[idx]
        ball = o3d.geometry.TriangleMesh.create_sphere(0.002).translate(suction_point)
        ball_v = np.asarray(ball.vertices)
        ball_colors = np.zeros((ball_v.shape[0], 3), dtype=np.float32)
        ball_colors[:, 0] = suction_score
        ball.vertex_colors = o3d.utility.Vector3dVector(ball_colors)
        vis_list.append(ball)
    
    o3d.visualization.draw_geometries(vis_list)
