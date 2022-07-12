import os, sys
sys.path.append('../')
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import json
import math

import torch

import open3d as o3d
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from src.utils.joint_estimation import aggregate_dense_prediction_r

from utils3d.mesh.utils import as_mesh
from utils3d.render.pyrender import get_pose, PyRenderer

from hydra.experimental import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import hydra

from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D
from src.utils.misc import sample_point_cloud

def get_arrow(origin=[0, 0, 0], end=None, color=[0, 0, 0]):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    vec_Arr = np.array(end) - np.array(origin)
    vec_len = np.linalg.norm(vec_Arr)
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * vec_len,
        cone_radius=0.06,
        cylinder_height=0.8 * vec_len,
        cylinder_radius=0.04,
    )
    mesh_arrow.paint_uniform_color(color)
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(origin))
    return mesh_arrow


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:   
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                        z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))
    qTrans_Mat *= scale
    return qTrans_Mat

def sum_downsample_points(point_list, voxel_size=0.01, nb_neighbors=20, std_ratio=2.0):
    points = np.concatenate([np.asarray(x.points) for x in point_list], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor / ((tensor ** 2).sum(dim, keepdim=True).sqrt() + 1.0e-5)

def visualize_pairs(pcds):
    colors = [[1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0, 1, 0], [1, 1, 1]]
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
    for i in range(len(pcds)):
        pcds[i].paint_uniform_color(colors[i])

    o3d.visualization.draw_geometries(pcds + [coordinate])


if __name__ == '__main__':
    visalize_middle = False
    # read data
    root = 'data/refrigerator_b005-0001_pcd'
    frame1 = '00012'
    frame2 = '00005'

    pcd1 = o3d.io.read_point_cloud(f'{root}/{frame1}.pcd')
    pcd2 = o3d.io.read_point_cloud(f'{root}/{frame2}.pcd')

    original_pcd1 = o3d.geometry.PointCloud(pcd1)
    original_pcd2 = o3d.geometry.PointCloud(pcd2)

    if visalize_middle:
        visualize_pairs([original_pcd1, original_pcd2])

    pc1 = np.asarray(pcd1.points)
    pc2 = np.asarray(pcd2.points)
    bound_max = np.maximum(pc1.max(0), pc2.max(0))
    bound_min = np.minimum(pc1.min(0), pc2.min(0))
    center = (bound_max + bound_min) / 2
    scale = (bound_max - bound_min).max() * 1.1
 
    # Normalize the two point clouds
    center_transform = np.eye(4)
    center_transform[:3, 3] = -center
    pcd1.transform(center_transform)
    pcd1.scale(1 / scale, np.zeros((3, 1)))

    pcd2.transform(center_transform)
    pcd2.scale(1 / scale, np.zeros((3, 1)))

    src_pcd = sum_downsample_points([pcd1], 0.02, 50, 0.1)
    dst_pcd = sum_downsample_points([pcd2], 0.02, 50, 0.1)
    if visalize_middle:
        visualize_pairs([src_pcd, dst_pcd])

    with initialize(config_path='configs/'):
        config = compose(
            config_name='config',
            overrides=[
                'experiment=Ditto_s2m.yaml',
            ], return_hydra_config=True)
    config.datamodule.opt.train.data_dir = 'data/'
    config.datamodule.opt.val.data_dir = 'data/'
    config.datamodule.opt.test.data_dir = 'data/'

    model = hydra.utils.instantiate(config.model)
    ckpt = torch.load('data/Ditto_s2m.ckpt')
    device = torch.device(0)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.eval().to(device)

    generator = Generator3D(
        model.model,
        device=device,
        threshold=0.4,
        seg_threshold=0.5,
        input_type='pointcloud',
        refinement_step=0,
        padding=0.1,
        resolution0=32
    )

    pc_start = np.asarray(src_pcd.points)
    pc_end = np.asarray(dst_pcd.points)

    pc_start, _ = sample_point_cloud(pc_start, 8192)
    pc_end, _ = sample_point_cloud(pc_end, 8192)
    sample = {
        'pc_start': torch.from_numpy(pc_start).unsqueeze(0).to(device).float(),
        'pc_end': torch.from_numpy(pc_end).unsqueeze(0).to(device).float()
    }

    mesh_dict, mobile_points_all, c, stats_dict = generator.generate_mesh(sample)
    with torch.no_grad():
        joint_type_logits, joint_param_revolute, joint_param_prismatic = model.model.decode_joints(mobile_points_all, c)

    static_part = mesh_dict[0].as_open3d
    moving_part = mesh_dict[1].as_open3d

    joint_type_prob = joint_type_logits.sigmoid().mean()
    if joint_type_prob.item()< 0.5:
        motion_type = 'rot'
    else:
        motion_type = 'trans'

    if motion_type == 'rot':
        # axis voting
        joint_r_axis = (
            normalize(joint_param_revolute[:, :, :3], -1)[0].cpu().numpy()
        )
        joint_r_t = joint_param_revolute[:, :, 3][0].cpu().numpy()
        joint_r_p2l_vec = (
            normalize(joint_param_revolute[:, :, 4:7], -1)[0].cpu().numpy()
        )
        joint_r_p2l_dist = joint_param_revolute[:, :, 7][0].cpu().numpy()
        p_seg = mobile_points_all[0].cpu().numpy()
        pivot_point = p_seg + joint_r_p2l_vec * joint_r_p2l_dist[:, np.newaxis]

        (
            joint_axis_pred,
            pivot_point_pred,
            config_pred,
        ) = aggregate_dense_prediction_r(
            joint_r_axis, pivot_point, joint_r_t, method="mean"
        )
    else:
        # axis voting
        joint_p_axis = (
            normalize(joint_param_prismatic[:, :, :3], -1)[0].cpu().numpy()
        )
        joint_axis_pred = joint_p_axis.mean(0)
        joint_p_t = joint_param_prismatic[:, :, 3][0].cpu().numpy()
        config_pred = joint_p_t.mean()
        
        pivot_point_pred = mesh_dict[1].bounds.mean(0)

    motion_state = config_pred
    motion_axis = joint_axis_pred
    # motion_origin = pivot_point_pred
    motion_origin = np.cross(motion_axis, np.cross(pivot_point_pred, motion_axis))

    # Make the object back
    center_transform = np.eye(4)
    center_transform[:3, 3] = center
    static_part.scale(scale, np.zeros((3, 1)))
    static_part.transform(center_transform)
    moving_part.scale(scale, np.zeros((3, 1)))
    moving_part.transform(center_transform)

    motion_origin = motion_origin * scale + center

    motion_arrow = get_arrow(motion_origin, motion_axis+motion_origin)

    visualize_pairs([static_part, moving_part, motion_arrow, original_pcd1])

    print(motion_type)
    print(motion_axis)
    print(motion_origin)
    print(motion_state)

    