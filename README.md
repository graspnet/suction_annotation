# Annotation Tools for SuctionNet-1Billion

Annotation tools for SuctionNet-1Billion dataset of RA-L paper "SuctionNet-1Billion:  A  Large-Scale  Benchmark  for  Suction  Grasping" .



## Annotation

The annotation process can be split into 3 parts: seal annotation, wrench annotation and scene collision annotation. To annotate wrench scores and scene collisions, please make sure you have finished seal annotation. The relative orders of wrench annotation and scene collision annotation don't matter. 

### Seal Annotation

Before annotation, it's recommended to generate dense point clouds for higher annotation quality.  To generate dense point clouds from mesh models, one should first install the [Point Cloud Utils](https://github.com/fwilliams/point-cloud-utils) library and run:

`python create_dense_pcd.py --model_root /path/to/graspnet/models --save_root /path/to/save/dense/point/clouds`

Then one can annotate seal scores as following:

`python seal_annotate.py --model_root /path/to/graspnet/models --poisson_root /path/to/dense/point/clouds --save_root /path/to/save/annotations`

One can also annotate without dense point clouds by simply leaving  `--poisson_root` empty.

### Wrench Annotation

 After finishing the seal annotation, run:

`python wrench_annotate.py --dataset_root /path/to/graspnet/dataset --seal_dir /path/to/seal/annotations --save_dir /path/to/save/wrench/annotations --camera camera_name (kinect or realsense)`

### Scene Collision Annotation

After finishing the seal annotation, run:

`python scene_collision_check.py --dataset_root /path/to/graspnet/dataset --seal_dir /path/to/seal/annotations --save_dir /path/to/save/collision/annotations --camera camera_name (kinect or realsense)`



## Visualization

We provide visualization tools to check the results of annotations.

### Check Normals

To check the normals of some model, run:

`python visualization/visualize_normals.py --model_root /path/to/graspnet/models --model_idx model_index`

### Check Seal Scores

To check seal scores, run:

`python visualization/visualize_seal_scores.py --model_root /path/to/graspnet/models --anno_root /path/to/annotation/results --obj_idx object_index`

### Check Wrench Scores

To check wrench scores, run:

`python visualization/visualize_wrench_scores.py --dataset_root /path/to/graspnet/dataset --seal_dir /path/to/seal/annotations --wrench_dir /path/to/wrench/annotations --camera camera_name(kinect or realsense) --scene_idx scene_index` 

### Check Scene Collision Annotations

To check scene collision annotations, run:

`python visualization/visualize_scene_collision.py --dataset_root /path/to/graspnet/dataset --seal_dir /path/to/seal/annotations --colli_dir /path/to/collision/annotations --camera camera_name(kinect or realsense) --obj_idx object_index --visu_each`