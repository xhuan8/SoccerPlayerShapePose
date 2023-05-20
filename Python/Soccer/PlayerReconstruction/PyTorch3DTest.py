from importlib.resources import path
from operator import truediv
import os
from pickle import FALSE, TRUE
import sys
from urllib.parse import uses_relative
import torch
import cv2
import numpy as np
import torch.optim as optim
import neural_renderer as nr
import copy
import torch.nn as nn
import timeit
import random

from models.regressor import SingleInputRegressor
from predict.predict_3D import *

from predict.predict_joints2D import predict_joints2D
from predict.predict_silhouette_pointrend import predict_silhouette_pointrend
from predict.predict_densepose import predict_densepose, apply_colormap

from utils.label_conversions import convert_multiclass_to_binary_labels, \
    convert_2Djoints_to_gaussian_heatmaps

from models.regressor_relate import PoseRelationModule
from models.regressor import SingleInputRegressor
from models.smpl_official import SMPL
from renderers.weak_perspective_pyrender_renderer import Renderer

from metrics.silhouettes_joints_metrics import compute_silh_error_metrics, compute_j2d_mean_l2_pixel_error
from utils.cam_utils import get_intrinsics_matrix, perspective_project_torch, \
    convert_weak_perspective_to_camera_translation, convert_weak_perspective_to_camera_translation_torch, \
    convert_camera_translation_to_weak_perspective
from renderers.nmr_renderer import NMRRenderer

import config

from losses.multi_task_loss import HomoscedasticUncertaintyWeightedMultiTaskLoss
from metrics.train_loss_and_metrics_tracker import TrainingLossesAndMetricsTracker
from utils.checkpoint_utils import load_training_info_from_checkpoint

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras, SfMPerspectiveCameras
from pytorch3d.renderer import (
    look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights
)

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from global_var import *
from global_utils import *

def TestPyTorch3D():
    device = torch.device('cuda:0')
    # Set-up SMPL model.
    smpl = SMPL('PlayerReconstruction/additional/smpl', batch_size=1).to(device)
    proxy_rep_input_wh = 512
    FOCAL_LENGTH = 5000.
    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)
    faces = torch.cat(1 * [faces[None, :]], dim=0)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    silhouette = np.load('PlayerReconstruction/view_1_sil.npy')
    #silhouette = np.flipud(silhouette)
    #silhouette = np.fliplr(silhouette).copy()

    cameras = PerspectiveCameras(device=device, 
            focal_length=((FOCAL_LENGTH, FOCAL_LENGTH),),  # (fx_screen, fy_screen)
            principal_point=((proxy_rep_input_wh / 2, proxy_rep_input_wh / 2),),  # (px_screen, py_screen)
            image_size=((proxy_rep_input_wh, proxy_rep_input_wh),),  # (imwidth, imheight)
            )
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color = (1.0, 1.0, 1.0))
    raster_settings = RasterizationSettings(image_size=proxy_rep_input_wh, 
        #blur_radius = 0.0,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=100, 
        perspective_correct=False,)
    #print(np.log(1.  / 1e-4 - 1.) * blend_params.sigma)
    silhouette_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, 
            raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params))

    translation = torch.Tensor([0.011280, 0.15844, 19]).to(device).unsqueeze(0)
    bodypose = [[[[0.9159, -0.4007, -0.0230],
          [0.2405,  0.5020,  0.8308],
          [-0.3214, -0.7664,  0.5561]],

         [[0.9979, -0.0650, -0.0026],
          [0.0647,  0.9961, -0.0592],
          [0.0064,  0.0589,  0.9982]],

         [[0.9946,  0.0997,  0.0277],
          [-0.0750,  0.8792, -0.4705],
          [-0.0713,  0.4659,  0.8820]],

         [[0.9282,  0.3445,  0.1406],
          [-0.0738,  0.5408, -0.8379],
          [-0.3647,  0.7674,  0.5274]],

         [[0.9928,  0.1044,  0.0590],
          [0.0645, -0.0497, -0.9967],
          [-0.1011,  0.9933, -0.0561]],

         [[0.9920, -0.0344, -0.1216],
          [0.0336,  0.9994, -0.0089],
          [0.1218,  0.0048,  0.9925]],

         [[0.9487,  0.0867,  0.3040],
          [-0.0432,  0.9882, -0.1468],
          [-0.3132,  0.1262,  0.9413]],

         [[0.9934,  0.0639, -0.0951],
          [-0.0223,  0.9218,  0.3869],
          [0.1124, -0.3823,  0.9172]],

         [[0.9985, -0.0141, -0.0521],
          [0.0120,  0.9992, -0.0391],
          [0.0526,  0.0384,  0.9979]],

         [[0.9825, -0.1811,  0.0446],
          [0.1591,  0.9384,  0.3068],
          [-0.0974, -0.2943,  0.9507]],

         [[0.9592,  0.0669, -0.2747],
          [-0.0341,  0.9919,  0.1224],
          [0.2807, -0.1080,  0.9537]],

         [[0.8477, -0.0791, -0.5246],
          [-0.0203,  0.9833, -0.1810],
          [0.5301,  0.1640,  0.8319]],

         [[0.9804,  0.1842, -0.0703],
          [-0.1904,  0.9770, -0.0961],
          [0.0510,  0.1076,  0.9929]],

         [[0.9861, -0.1526,  0.0659],
          [0.1558,  0.9867, -0.0462],
          [-0.0580,  0.0559,  0.9968]],

         [[0.8768, -0.1591, -0.4537],
          [0.2672,  0.9457,  0.1848],
          [0.3997, -0.2833,  0.8718]],

         [[0.7509,  0.6602, -0.0167],
          [-0.6154,  0.6903, -0.3805],
          [-0.2396,  0.2960,  0.9246]],

         [[0.5074, -0.8617,  0.0063],
          [0.8091,  0.4738, -0.3476],
          [0.2965,  0.1814,  0.9376]],

         [[0.1960, -0.5281, -0.8263],
          [-0.3019,  0.7692, -0.5632],
          [0.9330,  0.3599, -0.0087]],

         [[0.0077,  0.5370,  0.8436],
          [0.1944,  0.8267, -0.5280],
          [-0.9809,  0.1680, -0.0980]],

         [[0.6016, -0.7736, -0.1989],
          [0.7417,  0.6335, -0.2205],
          [0.2966, -0.0149,  0.9549]],

         [[0.7312,  0.6083,  0.3087],
          [-0.5935,  0.7904, -0.1515],
          [-0.3361, -0.0724,  0.9390]],

         [[0.9450,  0.3266, -0.0184],
          [-0.3086,  0.9089,  0.2806],
          [0.1083, -0.2595,  0.9596]],

         [[0.9600, -0.2654,  0.0887],
          [0.2580,  0.9622,  0.0868],
          [-0.1084, -0.0604,  0.9923]]]]
    global_orient = [[[[0.9550,  0.1267,  0.2682],
          [0.0699, -0.9748,  0.2118],
          [0.2883, -0.1835, -0.9398]]]]
    betas = [[1.4782,  1.4144, -0.4814,  0.7859,  1.7714,  0.3643,  0.6220, -1.0343,
          0.3492,  1.5779]]
    bodypose = torch.from_numpy(np.array(bodypose)).float().to(device)
    global_orient = torch.from_numpy(np.array(global_orient)).float().to(device)
    betas = torch.from_numpy(np.array(betas)).float().to(device)
    bodypose.requires_grad = True
    global_orient.requires_grad = True
    betas.requires_grad = True
    translation.requires_grad = True
    #print(bodypose.requires_grad)
    params = [bodypose, global_orient, betas, translation]
    optimiser = optim.Adam(params, lr=0.001)
    cv2.imwrite('PlayerReconstruction/2.png', silhouette * 255)
    for epoch in range(1000):
        pred_smpl_output = smpl(body_pose=bodypose,
                                global_orient=global_orient,
                                betas=betas,
                                pose2rot=False)
        teapot_mesh = Meshes(verts=pred_smpl_output.vertices, faces=faces)
        sil_image = silhouette_renderer(meshes_world=teapot_mesh, T=translation, R=cam_R)
        sil_image = sil_image[...,3]

        translation_nmr = torch.unsqueeze(translation, dim=1)
        pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                faces=faces,
                                t=translation_nmr,
                                mode='silhouettes')

        optimiser.zero_grad()
        loss = torch.sum((torch.from_numpy(silhouette).float().to(device).unsqueeze(0) - pred_silhouettes) ** 2)
        print(loss.item())
        #print(global_orient)
        #print(betas)
        loss.backward()
        optimiser.step()
        #print(global_orient)
        #print(betas)
    cv2.imwrite('PlayerReconstruction/1{}.png'.format(epoch), pred_silhouettes.cpu().detach().numpy()[0] * 255)
#TestPyTorch3D()
def TestRegressor():
    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)
    proxy_rep_input_wh = 512
    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)
    faces = torch.cat(1 * [faces[None, :]], dim=0)

    params = list(regressor.parameters())
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    checkpoint = torch.load(player_recon_check_points_path, map_location=device)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])

    silhouette = np.load('PlayerReconstruction/view_1_sil.npy')
    silhouette_rotate = np.flipud(silhouette.copy())
    silhouette_rotate = np.fliplr(silhouette_rotate).copy()
    joints2D = np.load('PlayerReconstruction/view_1_j2d.npy')
    proxy_rep = create_proxy_representation(silhouette, joints2D,
                                            in_wh=proxy_rep_input_wh,
                                            out_wh=config.REGRESSOR_IMG_WH)
    proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
    proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

    cameras = PerspectiveCameras(device=device, 
            focal_length=((config.FOCAL_LENGTH, config.FOCAL_LENGTH),),  # (fx_screen, fy_screen)
            principal_point=((proxy_rep_input_wh / 2, proxy_rep_input_wh / 2),),  # (px_screen, py_screen)
            image_size=((proxy_rep_input_wh, proxy_rep_input_wh),),  # (imwidth, imheight)
            )
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color = (1.0, 1.0, 1.0))
    raster_settings = RasterizationSettings(image_size=proxy_rep_input_wh, 
        #blur_radius = 0.0,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=100, 
        perspective_correct=False,)
    #print(np.log(1.  / 1e-4 - 1.) * blend_params.sigma)
    silhouette_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, 
            raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params))

    joints2D_loss_fun = nn.MSELoss(reduction='mean')

    best_val = float('inf')
    best_image = None
    for epoch in range(2000):
        pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep)
        if pred_pose.shape[-1] == 24 * 3:
            pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
            pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
        elif pred_pose.shape[-1] == 24 * 6:
            pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)
        pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                betas=pred_shape,
                                pose2rot=False)
        translation = \
            convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
        teapot_mesh = Meshes(verts=pred_smpl_output.vertices, faces=faces)
        sil_image = silhouette_renderer(meshes_world=teapot_mesh, T=translation, R=cam_R)
        sil_image = sil_image[...,3]

        pred_joints_all = pred_smpl_output.joints
        pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
        pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
        pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                        proxy_rep_input_wh)
        joints2D_label = torch.from_numpy(joints2D[:, :2]).float().to(device)
        joints2D_pred = pred_joints2d_coco[0]
        joints2D_label = (2.0 * joints2D_label) / config.REGRESSOR_IMG_WH - 1.0  # normalising j2d label
        joints2D_pred = (2.0 * joints2D_pred) / config.REGRESSOR_IMG_WH - 1.0  # normalising j2d label

        optimiser.zero_grad()
        loss = torch.sum((torch.from_numpy(silhouette_rotate).float().to(device).unsqueeze(0) - sil_image) ** 2)
        joints2D_loss = joints2D_loss_fun(joints2D_pred, joints2D_label)
        loss += joints2D_loss
        print(loss.item())
        if (loss.item() < best_val):
            best_image = sil_image
            best_val = loss.item()
        #print(global_orient)
        #print(betas)
        loss.backward()
        optimiser.step()
        #print(global_orient)
        #print(betas)
    print('best_val')
    print(best_val)
    cv2.imwrite('PlayerReconstruction/1{}.png'.format(epoch), best_image.cpu().detach().numpy()[0] * 255)

#TestRegressor()
def TestJoint():
    device = torch.device('cuda:0')
    # Set-up SMPL model.
    smpl = SMPL('PlayerReconstruction/additional/smpl', batch_size=1).to(device)
    proxy_rep_input_wh = 512
    FOCAL_LENGTH = 5000.
    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)
    faces = torch.cat(1 * [faces[None, :]], dim=0)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    silhouette = np.load('PlayerReconstruction/view_1_sil.npy')

    translation = torch.Tensor([0.011280, 0.15844, 19]).to(device).unsqueeze(0)
    bodypose = [[[[0.9159, -0.4007, -0.0230],
          [0.2405,  0.5020,  0.8308],
          [-0.3214, -0.7664,  0.5561]],

         [[0.9979, -0.0650, -0.0026],
          [0.0647,  0.9961, -0.0592],
          [0.0064,  0.0589,  0.9982]],

         [[0.9946,  0.0997,  0.0277],
          [-0.0750,  0.8792, -0.4705],
          [-0.0713,  0.4659,  0.8820]],

         [[0.9282,  0.3445,  0.1406],
          [-0.0738,  0.5408, -0.8379],
          [-0.3647,  0.7674,  0.5274]],

         [[0.9928,  0.1044,  0.0590],
          [0.0645, -0.0497, -0.9967],
          [-0.1011,  0.9933, -0.0561]],

         [[0.9920, -0.0344, -0.1216],
          [0.0336,  0.9994, -0.0089],
          [0.1218,  0.0048,  0.9925]],

         [[0.9487,  0.0867,  0.3040],
          [-0.0432,  0.9882, -0.1468],
          [-0.3132,  0.1262,  0.9413]],

         [[0.9934,  0.0639, -0.0951],
          [-0.0223,  0.9218,  0.3869],
          [0.1124, -0.3823,  0.9172]],

         [[0.9985, -0.0141, -0.0521],
          [0.0120,  0.9992, -0.0391],
          [0.0526,  0.0384,  0.9979]],

         [[0.9825, -0.1811,  0.0446],
          [0.1591,  0.9384,  0.3068],
          [-0.0974, -0.2943,  0.9507]],

         [[0.9592,  0.0669, -0.2747],
          [-0.0341,  0.9919,  0.1224],
          [0.2807, -0.1080,  0.9537]],

         [[0.8477, -0.0791, -0.5246],
          [-0.0203,  0.9833, -0.1810],
          [0.5301,  0.1640,  0.8319]],

         [[0.9804,  0.1842, -0.0703],
          [-0.1904,  0.9770, -0.0961],
          [0.0510,  0.1076,  0.9929]],

         [[0.9861, -0.1526,  0.0659],
          [0.1558,  0.9867, -0.0462],
          [-0.0580,  0.0559,  0.9968]],

         [[0.8768, -0.1591, -0.4537],
          [0.2672,  0.9457,  0.1848],
          [0.3997, -0.2833,  0.8718]],

         [[0.7509,  0.6602, -0.0167],
          [-0.6154,  0.6903, -0.3805],
          [-0.2396,  0.2960,  0.9246]],

         [[0.5074, -0.8617,  0.0063],
          [0.8091,  0.4738, -0.3476],
          [0.2965,  0.1814,  0.9376]],

         [[0.1960, -0.5281, -0.8263],
          [-0.3019,  0.7692, -0.5632],
          [0.9330,  0.3599, -0.0087]],

         [[0.0077,  0.5370,  0.8436],
          [0.1944,  0.8267, -0.5280],
          [-0.9809,  0.1680, -0.0980]],

         [[0.6016, -0.7736, -0.1989],
          [0.7417,  0.6335, -0.2205],
          [0.2966, -0.0149,  0.9549]],

         [[0.7312,  0.6083,  0.3087],
          [-0.5935,  0.7904, -0.1515],
          [-0.3361, -0.0724,  0.9390]],

         [[0.9450,  0.3266, -0.0184],
          [-0.3086,  0.9089,  0.2806],
          [0.1083, -0.2595,  0.9596]],

         [[0.9600, -0.2654,  0.0887],
          [0.2580,  0.9622,  0.0868],
          [-0.1084, -0.0604,  0.9923]]]]
    global_orient = [[[[0.9550,  0.1267,  0.2682],
          [0.0699, -0.9748,  0.2118],
          [0.2883, -0.1835, -0.9398]]]]
    betas = [[1.4782,  1.4144, -0.4814,  0.7859,  1.7714,  0.3643,  0.6220, -1.0343,
          0.3492,  1.5779]]
    bodypose = torch.from_numpy(np.array(bodypose)).float().to(device)
    global_orient = torch.from_numpy(np.array(global_orient)).float().to(device)
    betas = torch.from_numpy(np.array(betas)).float().to(device)
    translation_nmr = torch.unsqueeze(translation, dim=1)

    pred_smpl_output = smpl(body_pose=bodypose,
                                global_orient=global_orient,
                                betas=betas,
                                pose2rot=False)

    pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                            faces=faces,
                            t=translation_nmr,
                            mode='silhouettes')
    cv2.imwrite('Data/Temp/{}.png'.format(0), pred_silhouettes.cpu().detach().numpy()[0] * 255)
    for index in range(bodypose.shape[1]):
        bodypose[0][index] = torch.eye(3, dtype=bodypose.dtype, device=device)
        pred_smpl_output = smpl(body_pose=bodypose,
                                global_orient=global_orient,
                                betas=betas,
                                pose2rot=False)

        pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                faces=faces,
                                t=translation_nmr,
                                mode='silhouettes')
        cv2.imwrite('Data/Temp/{}.png'.format(index + 1), pred_silhouettes.cpu().detach().numpy()[0] * 255)
#TestJoint()
def TestOptimization():
    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    regressor.eval()

    #losses_on = ['joints2D', 'silhouette']
    losses_on = ['joints2D',]
    init_loss_weights = {'joints2D': 1.0, 'silhouette': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    #metrics_to_track = ['joints2D_l2es', 'silhouette_iou']
    metrics_to_track = ['joints2D_l2es']
    save_val_metrics = metrics_to_track

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)
    proxy_rep_input_wh = 512
    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)
    faces = torch.cat(1 * [faces[None, :]], dim=0)

    params = list(regressor.parameters())
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    checkpoint = torch.load(player_recon_check_points_path, map_location=device)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])

    silhouette = np.load('PlayerReconstruction/view_1_sil.npy')
    silhouette_rotate = np.flipud(silhouette.copy())
    silhouette_rotate = np.fliplr(silhouette_rotate).copy()
    joints2D = np.load('PlayerReconstruction/view_1_j2d.npy')
    proxy_rep = create_proxy_representation(silhouette, joints2D,
                                            in_wh=proxy_rep_input_wh,
                                            out_wh=config.REGRESSOR_IMG_WH)
    proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
    proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

    joints2D_loss_fun = nn.MSELoss(reduction='mean')

    with torch.no_grad():
        pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep)
        # Convert pred pose to rotation matrices
        if pred_pose.shape[-1] == 24 * 3:
            pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
            pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
        elif pred_pose.shape[-1] == 24 * 6:
            pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

        body_pose = pred_pose_rotmats[:, 1:]
        global_orient = pred_pose_rotmats[:, 0].unsqueeze(1)
        betas = pred_shape
        translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                                init_loss_weights=init_loss_weights,
                                                                reduction='mean')
    criterion.to(device)

    # ----------------------- Optimiser -----------------------
    body_poses_without_hands_feet = torch.cat([body_pose[:, :6, :, :],
                                            body_pose[:, 8:21, :, :]],
                                            dim=1)

    current_epoch = 1
    best_epoch_val_metrics = {}
    best_epoch = current_epoch
    best_model_wts = copy.deepcopy([body_pose.cpu().detach().numpy(), 
                                    global_orient.cpu().detach().numpy(), 
                                    betas.cpu().detach().numpy(), 
                                    translation.cpu().detach().numpy()])
    for metric in save_val_metrics:
        best_epoch_val_metrics[metric] = np.inf
    load_logs = False

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                        metrics_to_track=metrics_to_track,
                                                        img_wh=config.REGRESSOR_IMG_WH,
                                                        log_path=log_path,
                                                        load_logs=load_logs,
                                                        current_epoch=current_epoch)
    metrics_tracker.initialise_loss_metric_sums()

    # optimize global orientation and translation
    global_orient.requires_grad = True
    betas.requires_grad = False
    translation.requires_grad = True
    body_poses_without_hands_feet.requires_grad = False
    params = [global_orient, translation]
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    for epoch in range(1, player_recon_single_view_iteration + 1):
        body_poses_with_hand_feet = torch.cat([body_poses_without_hands_feet[:, :6, :, :],
                        body_pose[:, 6:8, :, :],
                        body_poses_without_hands_feet[:, 6:19, :, :],
                        body_pose[:, 21:, :, :]],
                        dim=1)
        pred_smpl_output = smpl(body_pose=body_poses_with_hand_feet,
                                global_orient=global_orient,
                                betas=betas,
                                pose2rot=False)

        if (epoch == 1):
            rend_img = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                            cam=pred_cam_wp.cpu().detach().numpy()[0])
            cv2.imwrite('Data/Temp/0.png', rend_img)

        pred_joints_all = pred_smpl_output.joints
        pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
        pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
        pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                        proxy_rep_input_wh)

        
        # Need to expand cam_ts for NMR i.e.  from (B, 3) to
        # (B, 1, 3)
        translation_nmr = torch.unsqueeze(translation, dim=1)
        pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                    faces=faces,
                                    t=translation_nmr,
                                    mode='silhouettes')

        pred_dict_for_loss = {'joints2D': pred_joints2d_coco[0],
                                'silhouette': pred_silhouettes}
        target_dict_for_loss = {'joints2D': torch.from_numpy(joints2D[:, :2]).float().to(device),
                                'silhouette': torch.from_numpy(silhouette).float().to(device).unsqueeze(0)}

        # ---------------- BACKWARD PASS ----------------
        optimiser.zero_grad()
        loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

        # ---------------- TRACK LOSS AND METRICS
        # ----------------
        num_train_inputs_in_batch = 1
        metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                            pred_dict_for_loss, target_dict_for_loss,
                                            num_train_inputs_in_batch)
        metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                            pred_dict_for_loss, target_dict_for_loss,
                                            num_train_inputs_in_batch)
        metrics_tracker.update_per_epoch()

        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
        if epoch == 1:
            for metric in metrics_to_track:
                print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                    metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                metric,
                                                                metrics_tracker.history['val_' + metric][-1]))

        loss.backward()
        optimiser.step()
    
    print(pred_joints2d_coco)
    #print(body_poses_with_hand_feet)
    for metric in metrics_to_track:
        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                    metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                metric,
                                                                metrics_tracker.history['val_' + metric][-1]))
    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                    cam=convert_camera_translation_to_weak_perspective(translation.cpu().detach().numpy()[0], config.FOCAL_LENGTH, proxy_rep_input_wh))
    cv2.imwrite('Data/Temp/1.png', best_image)

    # optimize pose
    global_orient.requires_grad = False
    betas.requires_grad = False
    translation.requires_grad = False
    body_poses_without_hands_feet.requires_grad = True
    params = [body_poses_without_hands_feet]
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    for epoch in range(1, player_recon_single_view_iteration + 1):
        body_poses_with_hand_feet = torch.cat([body_poses_without_hands_feet[:, :6, :, :],
                        body_pose[:, 6:8, :, :],
                        body_poses_without_hands_feet[:, 6:19, :, :],
                        body_pose[:, 21:, :, :]],
                        dim=1)
        pred_smpl_output = smpl(body_pose=body_poses_with_hand_feet,
                                global_orient=global_orient,
                                betas=betas,
                                pose2rot=False)

        pred_joints_all = pred_smpl_output.joints
        pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
        pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
        pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                        proxy_rep_input_wh)

        # Need to expand cam_ts for NMR i.e.  from (B, 3) to
        # (B, 1, 3)
        translation_nmr = torch.unsqueeze(translation, dim=1)
        pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                    faces=faces,
                                    t=translation_nmr,
                                    mode='silhouettes')

        pred_dict_for_loss = {'joints2D': pred_joints2d_coco[0],
                                'silhouette': pred_silhouettes}
        target_dict_for_loss = {'joints2D': torch.from_numpy(joints2D[:, :2]).float().to(device),
                                'silhouette': torch.from_numpy(silhouette).float().to(device).unsqueeze(0)}

        # ---------------- BACKWARD PASS ----------------
        optimiser.zero_grad()
        loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

        # ---------------- TRACK LOSS AND METRICS
        # ----------------
        num_train_inputs_in_batch = 1
        metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                            pred_dict_for_loss, target_dict_for_loss,
                                            num_train_inputs_in_batch)
        metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                            pred_dict_for_loss, target_dict_for_loss,
                                            num_train_inputs_in_batch)
        metrics_tracker.update_per_epoch()

        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]

        loss.backward()
        optimiser.step()

    #print(body_poses_with_hand_feet)
    print(pred_joints2d_coco)
    for metric in metrics_to_track:
        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                    metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                metric,
                                                                metrics_tracker.history['val_' + metric][-1]))
    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                    cam=convert_camera_translation_to_weak_perspective(translation.cpu().detach().numpy()[0], config.FOCAL_LENGTH, proxy_rep_input_wh))
    cv2.imwrite('Data/Temp/2.png', best_image)

    #losses_on = ['joints2D', 'silhouette']
    losses_on = ['silhouette',]
    init_loss_weights = {'joints2D': 1.0, 'silhouette': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    #metrics_to_track = ['joints2D_l2es', 'silhouette_iou']
    metrics_to_track = ['silhouette_iou']
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                                init_loss_weights=init_loss_weights,
                                                                reduction='mean')
    criterion.to(device)
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                        metrics_to_track=metrics_to_track,
                                                        img_wh=config.REGRESSOR_IMG_WH,
                                                        log_path=log_path,
                                                        load_logs=load_logs,
                                                        current_epoch=current_epoch)
    metrics_tracker.initialise_loss_metric_sums()

    save_val_metrics = metrics_to_track
    # optimize shape
    global_orient.requires_grad = False
    betas.requires_grad = True
    translation.requires_grad = False
    body_poses_without_hands_feet.requires_grad = False
    params = [betas]
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    for epoch in range(1, player_recon_single_view_iteration + 1):
        body_poses_with_hand_feet = torch.cat([body_poses_without_hands_feet[:, :6, :, :],
                        body_pose[:, 6:8, :, :],
                        body_poses_without_hands_feet[:, 6:19, :, :],
                        body_pose[:, 21:, :, :]],
                        dim=1)
        pred_smpl_output = smpl(body_pose=body_poses_with_hand_feet,
                                global_orient=global_orient,
                                betas=betas,
                                pose2rot=False)

        pred_joints_all = pred_smpl_output.joints
        pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
        pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
        pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                        proxy_rep_input_wh)

        # Need to expand cam_ts for NMR i.e.  from (B, 3) to
        # (B, 1, 3)
        translation_nmr = torch.unsqueeze(translation, dim=1)
        pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                    faces=faces,
                                    t=translation_nmr,
                                    mode='silhouettes')

        pred_dict_for_loss = {'joints2D': pred_joints2d_coco[0],
                                'silhouette': pred_silhouettes}
        target_dict_for_loss = {'joints2D': torch.from_numpy(joints2D[:, :2]).float().to(device),
                                'silhouette': torch.from_numpy(silhouette).float().to(device).unsqueeze(0)}

        # ---------------- BACKWARD PASS ----------------
        optimiser.zero_grad()
        loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

        # ---------------- TRACK LOSS AND METRICS
        # ----------------
        num_train_inputs_in_batch = 1
        metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                            pred_dict_for_loss, target_dict_for_loss,
                                            num_train_inputs_in_batch)
        metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                            pred_dict_for_loss, target_dict_for_loss,
                                            num_train_inputs_in_batch)
        metrics_tracker.update_per_epoch()

        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]

        loss.backward()
        optimiser.step()

    #print(body_poses_with_hand_feet)
    for metric in metrics_to_track:
        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                    metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                metric,
                                                                metrics_tracker.history['val_' + metric][-1]))
    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                    cam=convert_camera_translation_to_weak_perspective(translation.cpu().detach().numpy()[0], config.FOCAL_LENGTH, proxy_rep_input_wh))
    cv2.imwrite('Data/Temp/3.png', best_image)
#TestOptimization()
def train_regressor(load_checkpoint=True, data_argument=False):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if not os.path.exists(player_recon_train_regressor_checkpoints_folder):
        os.makedirs(player_recon_train_regressor_checkpoints_folder, exist_ok=True)
    if not os.path.exists(player_recon_train_regressor_logs_folder):
        os.makedirs(player_recon_train_regressor_logs_folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints2D', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa']
    epochs_per_save = 10

    with open('Data/train_set.xml', 'r') as fs:
        train_set = set(json.load(fs))

    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # ----------------------- Optimiser -----------------------
    params = list(regressor.parameters()) + list(criterion.parameters())
    optimiser = optim.Adam(params, lr=0.0001)

    # ----------------------- Resuming -----------------------
    checkpoint_path = os.path.join(player_recon_train_regressor_checkpoints_folder, 'best.tar')
    check_point_loaded = False
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        check_point_loaded = True

        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        #criterion.load_state_dict(checkpoint['criterion_state_dict'])
        print("Regressor loaded. Weights from:", checkpoint_path)

    else:
        checkpoint = torch.load(player_recon_check_points_path, map_location=device)
        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", player_recon_check_points_path)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    if check_point_loaded:
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = \
            load_training_info_from_checkpoint(checkpoint, save_val_metrics)
        load_logs = False
    else:
        current_epoch = 1
        best_epoch_val_metrics = {}
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(regressor.state_dict())
        load_logs = False
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)
    # Starting training loop
    num_epochs = 300 + current_epoch
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    for epoch in range(current_epoch, num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        starttime = timeit.default_timer()
        
        for is_train in [False, True]:
            if (is_train):
                print('Training.')
                regressor.train()
            else:
                print('eval')
                regressor.eval()
                #regressor.fix()

            games = os.listdir(player_recon_broad_view_opt_folder)
            if is_train:
                random.shuffle(games)
            for game in games:
                if is_train:
                    if game not in train_set:
                        continue
                else:
                    if game in train_set:
                        continue

                game_full = os.path.join(player_crop_broad_image_folder, game)
                game_dst = os.path.join(player_broad_proxy_folder, game)
                game_label = os.path.join(player_recon_broad_view_opt_folder, game)
                scenes = os.listdir(game_full)
                if is_train:
                    random.shuffle(scenes)
                for scene in scenes:
                    scene_full = os.path.join(game_full, scene)
                    #print('process {}'.format(scene_full))
                    scene_dst = os.path.join(game_dst, scene)
                    scene_label = os.path.join(game_label, scene)
                    players = os.listdir(scene_full)

                    body_pose_batch = []
                    global_orient_batch = []
                    betas_batch = []
                    translation_batch = []
                    joints2D_batch = []
                    silhouette_batch = []
                    proxy_rep_batch = []
                    for player in players:
                        player_full = os.path.join(scene_full, player)
                        if (player == '1' or os.path.isfile(player_full)):
                            continue
                        #print(player_full)
                        player_dst = os.path.join(scene_dst, player)
                        player_label = os.path.join(scene_label, player)
                        
                        view_full = os.path.join(player_full, "player.png")
                        j2d_full = os.path.join(player_dst, 'player_j2d.xml')
                        sil_full = os.path.join(player_dst, 'player_sil.npy')
                        with open(j2d_full, 'r') as fs:
                            joints2D = np.array(json.load(fs))
                        silhouette = np.load(sil_full)
                        joints2D_batch.append(joints2D[:, :2])
                        silhouette_batch.append(silhouette[0])

                        label_full = os.path.join(player_label, "data.npz")
                        if not os.path.exists(label_full):
                            continue
                        target_param = np.load(label_full)
                        body_pose = target_param['body_pose']
                        global_orient = target_param['global_orient']
                        betas = target_param['betas']
                        translation = target_param['translation']

                        body_pose_batch.append(body_pose[0])
                        global_orient_batch.append(global_orient[0])
                        betas_batch.append(betas[0])
                        translation_batch.append(translation[0])

                        proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                        in_wh=proxy_rep_input_wh,
                                                        out_wh=config.REGRESSOR_IMG_WH)
                        proxy_rep_batch.append(proxy_rep)

                    faces_batch = torch.cat(len(joints2D_batch) * [faces[None, :]], dim=0)
                    body_pose = torch.from_numpy(np.array(body_pose_batch)).float().to(device)
                    betas = torch.from_numpy(np.array(betas_batch)).float().to(device)
                    global_orient = torch.from_numpy(np.array(global_orient_batch)).float().to(device)
                    translation = torch.from_numpy(np.array(translation_batch)).float().to(device)
                    target_pose_rotmats = torch.cat([global_orient[:, :, :, :],
                                body_pose[:, :, :, :]],
                                dim=1)
                    target_smpl_output = smpl(body_pose=body_pose,
                                            global_orient=global_orient,
                                            betas=betas,
                                            pose2rot=False)
                    target_vertices = target_smpl_output.vertices
                    target_joints_all = target_smpl_output.joints
                    target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]

                    proxy_rep = torch.from_numpy(np.array(proxy_rep_batch)).float().to(device)

                    pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep)

                    # Convert pred pose to rotation matrices
                    if pred_pose.shape[-1] == 24 * 3:
                        pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                        pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                    elif pred_pose.shape[-1] == 24 * 6:
                        pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                    pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                            global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                            betas=pred_shape,
                                            pose2rot=False)
                    pred_vertices = pred_smpl_output.vertices
                    pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
                    pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                                    proxy_rep_input_wh)

                    smpl_joints = pred_smpl_output.joints

                    pred_joints_all = pred_smpl_output.joints
                    pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
                    pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
                    pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
                    pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                                    proxy_rep_input_wh)

                    rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                        -1, -1)
                    translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
                    #pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
                    #                                                config.FOCAL_LENGTH, proxy_rep_input_wh)
                    #pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

                    # Need to expand cam_ts for NMR i.e.  from (B, 3)
                    # to
                    # (B, 1, 3)
                    translation_nmr = torch.unsqueeze(translation, dim=1)
                    pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                                faces=faces_batch,
                                                t=translation_nmr,
                                                mode='silhouettes')

                    pred_dict_for_loss = {'joints2D': pred_joints2d_coco,
                                            'verts': pred_vertices,
                                            'shape_params': pred_shape,
                                            'pose_params_rot_matrices': pred_pose_rotmats,
                                            'joints3D': pred_joints_coco}
                    target_dict_for_loss = {'joints2D': torch.from_numpy(np.array(joints2D_batch)[:, :, :2]).float().to(device),
                                            'verts': target_vertices,
                                            'shape_params': betas,
                                            'pose_params_rot_matrices': target_pose_rotmats,
                                            'joints3D': target_joints_coco}

                    # ---------------- BACKWARD PASS ----------------
                    if (is_train):
                        optimiser.zero_grad()
                    loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)
                    if (is_train):
                        loss.backward()
                        optimiser.step()

                    # ---------------- TRACK LOSS AND METRICS
                    # ----------------
                    num_train_inputs_in_batch = len(joints2D_batch)
                    if (is_train):
                        metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                            pred_dict_for_loss, target_dict_for_loss,
                                                            num_train_inputs_in_batch)
                    else:
                        metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                            pred_dict_for_loss, target_dict_for_loss,
                                                            num_train_inputs_in_batch)
                #    break
                #break

            #print(is_train)
            if not is_train:
                # ----------------------- UPDATING LOSS AND METRICS HISTORY
                # -----------------------
                metrics_tracker.update_per_epoch()

                # ----------------------------------- SAVING
                # -----------------------------------
                save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                        best_epoch_val_metrics)

                for metric in save_val_metrics:
                    print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                        metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                    metric,
                                                                    metrics_tracker.history['val_' + metric][-1]))
                if save_model_weights_this_epoch:
                    for metric in save_val_metrics:
                        best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
                    print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                    best_model_wts = copy.deepcopy(regressor.state_dict())
                    best_epoch = epoch
                    print("Best model weights updated!")

                    save_dict = {'epoch': epoch,
                                    'best_epoch': best_epoch,
                                    'best_epoch_val_metrics': best_epoch_val_metrics,
                                    'model_state_dict': regressor.state_dict(),
                                    'best_model_state_dict': best_model_wts,
                                    'optimiser_state_dict': optimiser.state_dict(),
                                    'criterion_state_dict': criterion.state_dict()}
                    model_save_path = os.path.join(player_recon_train_regressor_checkpoints_folder, 'best.tar')
                    torch.save(save_dict, model_save_path)

                if epoch % epochs_per_save == 0:
                    # Saving current epoch num, best epoch num, best validation
                    # metrics
                    # (occurred in best
                    # epoch num), current regressor state_dict, best regressor
                    # state_dict, current
                    # optimiser state dict and current criterion state_dict
                    # (i.e.
                    # multi-task loss weights).
                    save_dict = {'epoch': epoch,
                                    'best_epoch': best_epoch,
                                    'best_epoch_val_metrics': best_epoch_val_metrics,
                                    'model_state_dict': regressor.state_dict(),
                                    'best_model_state_dict': best_model_wts,
                                    'optimiser_state_dict': optimiser.state_dict(),
                                    'criterion_state_dict': criterion.state_dict()}
                    model_save_path = os.path.join(player_recon_train_regressor_checkpoints_folder, 'model')
                    torch.save(save_dict,
                                model_save_path + '_epoch{}'.format(epoch) + '.tar')
                    print('Model saved! Best Val Metrics:\n',
                            best_epoch_val_metrics,
                            '\nin epoch {}'.format(best_epoch))
                metrics_tracker.initialise_loss_metric_sums()
        endtime = timeit.default_timer()
        print('epoch time: {:.3f}'.format(endtime - starttime))

    print('Training Completed. Best Val Metrics:\n',
          best_epoch_val_metrics)
    print('Best epoch: ', best_epoch)

def evaluate_model(load_checkpoint=True, folder='Data/STA', 
                   path=player_recon_train_regressor_checkpoints_folder,
                   postfix='', test_pose=False, save_vis=True, unrefine=False):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if save_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints2D', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa', 'pose_mses', 'shape_mses']
    epochs_per_save = 10

    with open('Data/train_set.xml', 'r') as fs:
        train_set = set(json.load(fs))

    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # ----------------------- Optimiser -----------------------
    params = list(regressor.parameters()) + list(criterion.parameters())
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    # ----------------------- Resuming -----------------------
    checkpoint_path = os.path.join(path, 'best.tar')
    check_point_loaded = False
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        check_point_loaded = True

        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        #criterion.load_state_dict(checkpoint['criterion_state_dict'])
        print("Regressor loaded. Weights from:", checkpoint_path)

    else:
        checkpoint = torch.load(player_recon_check_points_path, map_location=device)
        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", player_recon_check_points_path)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    if check_point_loaded:
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = \
            load_training_info_from_checkpoint(checkpoint, save_val_metrics)
        load_logs = False
    else:
        current_epoch = 1
        best_epoch_val_metrics = {}
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(regressor.state_dict())
        load_logs = False
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)
    # Starting training loop
    num_epochs = 10 + current_epoch
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    starttime = timeit.default_timer()
        
    print('eval')
    regressor.eval()

    games = os.listdir(player_recon_broad_view_opt_folder+postfix)
    for game in games:
        if game in train_set:
            is_train = True
        else:
            is_train = False

        game_full = os.path.join(player_crop_broad_image_folder+postfix, game)
        game_dst = os.path.join(player_broad_proxy_folder+postfix+'unrefine' if unrefine else player_broad_proxy_folder+postfix, game)
        game_label = os.path.join(player_recon_broad_view_opt_folder+postfix, game)
        if save_vis and not is_train:
            game_vis = os.path.join(folder, game)
            remake_dir(game_vis)
        if not is_train:
            print('process {}'.format(game_full))
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            
            scene_dst = os.path.join(game_dst, scene)
            scene_label = os.path.join(game_label, scene)
            if save_vis and not is_train:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_full)
            body_pose_batch = []
            global_orient_batch = []
            betas_batch = []
            translation_batch = []
            joints2D_batch = []
            silhouette_batch = []
            proxy_rep_batch = []
            player_vis_batch = []
            image_batch = []
            for player in players:
                player_full = os.path.join(scene_full, player)
                if (player == '1' or os.path.isfile(player_full)):
                    continue
                #print(player_full)
                player_dst = os.path.join(scene_dst, player)
                player_label = os.path.join(scene_label, player)
                if save_vis and not is_train:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)
                    player_vis_batch.append(player_vis)
                        
                view_full = os.path.join(player_full, "player.png")
                image = cv2.imread(view_full)
                image_batch.append(image)
                j2d_full = os.path.join(player_dst, 'player_j2d.xml')
                sil_full = os.path.join(player_dst, 'player_sil.npy')
                with open(j2d_full, 'r') as fs:
                    joints2D = np.array(json.load(fs))
                silhouette = np.load(sil_full)
                joints2D_batch.append(joints2D[:, :2])
                silhouette_batch.append(silhouette[0])

                label_full = os.path.join(player_label, "data.npz")
                if not os.path.exists(label_full):
                    continue
                target_param = np.load(label_full)
                body_pose = target_param['body_pose']
                global_orient = target_param['global_orient']
                betas = target_param['betas']
                translation = target_param['translation']

                body_pose_batch.append(body_pose[0])
                global_orient_batch.append(global_orient[0])
                betas_batch.append(betas[0])
                translation_batch.append(translation[0])

                proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                in_wh=proxy_rep_input_wh,
                                                out_wh=config.REGRESSOR_IMG_WH)
                proxy_rep_batch.append(proxy_rep)

            faces_batch = torch.cat(len(joints2D_batch) * [faces[None, :]], dim=0)
            body_pose = torch.from_numpy(np.array(body_pose_batch)).float().to(device)
            betas = torch.from_numpy(np.array(betas_batch)).float().to(device)
            global_orient = torch.from_numpy(np.array(global_orient_batch)).float().to(device)
            translation = torch.from_numpy(np.array(translation_batch)).float().to(device)
            target_pose_rotmats = torch.cat([global_orient[:, :, :, :],
                        body_pose[:, :, :, :]],
                        dim=1)
            target_smpl_output = smpl(body_pose=body_pose,
                                    global_orient=global_orient,
                                    betas=betas,
                                    pose2rot=False)
            target_vertices = target_smpl_output.vertices
            target_joints_all = target_smpl_output.joints
            target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]

            proxy_rep = torch.from_numpy(np.array(proxy_rep_batch)).float().to(device)

            pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep)

            # Convert pred pose to rotation matrices
            if pred_pose.shape[-1] == 24 * 3:
                pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
            elif pred_pose.shape[-1] == 24 * 6:
                pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

            if not test_pose:
                pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
            else:
                pred_smpl_output = smpl(body_pose=body_pose,
                                    global_orient=global_orient,
                                    betas=pred_shape,
                                    pose2rot=False)
            pred_vertices = pred_smpl_output.vertices
            pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
            pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                            proxy_rep_input_wh)

            smpl_joints = pred_smpl_output.joints

            pred_joints_all = pred_smpl_output.joints
            pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
            pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
            pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                            proxy_rep_input_wh)

            rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                -1, -1)
            translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
            #                                                config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

            # Need to expand cam_ts for NMR i.e.  from (B, 3)
            # to
            # (B, 1, 3)
            translation_nmr = torch.unsqueeze(translation, dim=1)
            pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                        faces=faces_batch,
                                        t=translation_nmr,
                                        mode='silhouettes')

            if not is_train and save_vis:
                for i in range(len(joints2D_batch)):
                    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[i], 
                                    cam=pred_cam_wp.cpu().detach().numpy()[i], img=image_batch[i])
                    cv2.imwrite(os.path.join(player_vis_batch[i], 'player.png'), best_image)

            pred_dict_for_loss = {'joints2D': pred_joints2d_coco,
                                    'verts': pred_vertices,
                                    'shape_params': pred_shape,
                                    'pose_params_rot_matrices': pred_pose_rotmats,
                                    'joints3D': pred_joints_coco}
            target_dict_for_loss = {'joints2D': torch.from_numpy(np.array(joints2D_batch)[:, :, :2]).float().to(device),
                                    'verts': target_vertices,
                                    'shape_params': betas,
                                    'pose_params_rot_matrices': target_pose_rotmats,
                                    'joints3D': target_joints_coco}

            # ---------------- BACKWARD PASS ----------------
            loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

            # ---------------- TRACK LOSS AND METRICS
            # ----------------
            num_train_inputs_in_batch = len(joints2D_batch)
            if not is_train:
                metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                    pred_dict_for_loss, target_dict_for_loss,
                                                    num_train_inputs_in_batch)
            else:
                metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                    pred_dict_for_loss, target_dict_for_loss,
                                                    num_train_inputs_in_batch)
        #    break
        #break

    #print(is_train)
    # ----------------------- UPDATING LOSS AND METRICS HISTORY
    # -----------------------
    metrics_tracker.update_per_epoch()

    # ----------------------------------- SAVING
    # -----------------------------------
    for metric in save_val_metrics:
        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
            metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                        metric,
                                                        metrics_tracker.history['val_' + metric][-1]))

    endtime = timeit.default_timer()
    print('epoch time: {:.3f}'.format(endtime - starttime))

def evaluate_model_relate(load_checkpoint=True, folder='Data/relate', 
                   path=player_recon_train_regressor_checkpoints_folder,
                   path1=player_recon_train_regressor_checkpoints_folder+'relate',
                   postfix='', test_pose=False, save_vis=True, game_process='', opt=False):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if save_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints2D', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa', 'pose_mses', 'shape_mses']
    epochs_per_save = 10

    with open('Data/train_set.xml', 'r') as fs:
        train_set = set(json.load(fs))

    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)

    pose_relation = PoseRelationModule()
    pose_relation.to(device)
    pose_relation.eval()

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # ----------------------- Optimiser -----------------------
    params = list(regressor.parameters()) + list(criterion.parameters())
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    # ----------------------- Resuming -----------------------
    checkpoint_path = os.path.join(path, 'best.tar')
    check_point_loaded = False
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        check_point_loaded = True

        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        #criterion.load_state_dict(checkpoint['criterion_state_dict'])
        print("Regressor loaded. Weights from:", checkpoint_path)

    else:
        checkpoint = torch.load(player_recon_check_points_path, map_location=device)
        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", player_recon_check_points_path)

    checkpoint_path = os.path.join(path1, 'best.tar')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    pose_relation.load_state_dict(checkpoint['best_model_state_dict'])
    print("Pose_relation loaded. Weights from:", checkpoint_path)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    if check_point_loaded:
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = \
            load_training_info_from_checkpoint(checkpoint, save_val_metrics)
        load_logs = False
    else:
        current_epoch = 1
        best_epoch_val_metrics = {}
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(regressor.state_dict())
        load_logs = False
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)
    # Starting training loop
    num_epochs = 10 + current_epoch
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    starttime = timeit.default_timer()
        
    print('eval')
    regressor.eval()

    games = os.listdir(player_recon_broad_view_opt_folder+postfix)
    for game in games:
        if game in train_set:
            is_train = True
        else:
            is_train = False

        if game_process != '' and game_process != game:
            continue
        game_full = os.path.join(player_crop_broad_image_folder+postfix, game)
        game_dst = os.path.join(player_broad_proxy_folder+postfix, game)
        game_label = os.path.join(player_recon_broad_view_opt_folder+postfix, game)
        game_box = os.path.join(player_crop_broad_folder, game)
        if save_vis and not is_train:
            game_vis = os.path.join(folder, game)
            remake_dir(game_vis)
        if not is_train:
            print('process {}'.format(game_full))
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            
            scene_dst = os.path.join(game_dst, scene)
            scene_label = os.path.join(game_label, scene)
            if len(scene) == 2:
                scene_box = os.path.join(game_box, str(int(scene)//10))
            else:
                scene_box = os.path.join(game_box, scene)
            boxes_full = os.path.join(scene_box, 'boxes.xml')
            index_full = os.path.join(scene_box, 'index.xml')
            with open(boxes_full, 'r') as fs:
                boxes = json.load(fs)
            with open(index_full, 'r') as fs:
                indexes = json.load(fs)
            if save_vis and not is_train:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_full)
            body_pose_batch = []
            global_orient_batch = []
            betas_batch = []
            translation_batch = []
            joints2D_batch = []
            silhouette_batch = []
            proxy_rep_batch = []
            player_vis_batch = []
            image_batch = []
            boxes_batch = []
            for player in players:
                player_full = os.path.join(scene_full, player)
                if (player == '1' or os.path.isfile(player_full)):
                    continue
                #print(player_full)
                for iii in range(len(indexes)):
                    if indexes[iii] == player:
                        boxes_batch.append(boxes[iii])
                        break
                player_dst = os.path.join(scene_dst, player)
                player_label = os.path.join(scene_label, player)
                if save_vis and not is_train:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)
                    player_vis_batch.append(player_vis)
                        
                view_full = os.path.join(player_full, "player.png")
                image = cv2.imread(view_full)
                image_batch.append(image)
                j2d_full = os.path.join(player_dst, 'player_j2d.xml')
                sil_full = os.path.join(player_dst, 'player_sil.npy')
                with open(j2d_full, 'r') as fs:
                    joints2D = np.array(json.load(fs))
                silhouette = np.load(sil_full)
                joints2D_batch.append(joints2D[:, :2])
                silhouette_batch.append(silhouette[0])

                label_full = os.path.join(player_label, "data.npz")
                if not os.path.exists(label_full):
                    continue
                target_param = np.load(label_full)
                body_pose = target_param['body_pose']
                global_orient = target_param['global_orient']
                betas = target_param['betas']
                translation = target_param['translation']

                body_pose_batch.append(body_pose[0])
                global_orient_batch.append(global_orient[0])
                betas_batch.append(betas[0])
                translation_batch.append(translation[0])

                proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                in_wh=proxy_rep_input_wh,
                                                out_wh=config.REGRESSOR_IMG_WH)
                proxy_rep_batch.append(proxy_rep)

            faces_batch = torch.cat(len(joints2D_batch) * [faces[None, :]], dim=0)
            body_pose = torch.from_numpy(np.array(body_pose_batch)).float().to(device)
            betas = torch.from_numpy(np.array(betas_batch)).float().to(device)
            global_orient = torch.from_numpy(np.array(global_orient_batch)).float().to(device)
            translation = torch.from_numpy(np.array(translation_batch)).float().to(device)
            target_pose_rotmats = torch.cat([global_orient[:, :, :, :],
                        body_pose[:, :, :, :]],
                        dim=1)
            target_smpl_output = smpl(body_pose=body_pose,
                                    global_orient=global_orient,
                                    betas=betas,
                                    pose2rot=False)
            target_vertices = target_smpl_output.vertices
            target_joints_all = target_smpl_output.joints
            target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            boxes_batch = torch.from_numpy(np.array(boxes_batch)).float().to(device)

            proxy_rep = torch.from_numpy(np.array(proxy_rep_batch)).float().to(device)

            pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep)

            # Convert pred pose to rotation matrices
            if pred_pose.shape[-1] == 24 * 3:
                pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
            elif pred_pose.shape[-1] == 24 * 6:
                pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

            pred_pose_rotmats = pose_relation([pred_pose_rotmats, boxes_batch])

            if opt:
                pred_pose_rotmats = pred_pose_rotmats.detach()
                pred_shape = pred_shape.detach()
                pred_cam_wp = pred_cam_wp.detach()
                for i in range(len(joints2D_batch)):
                    optimize_camera(pred_pose_rotmats[i, 1:].unsqueeze(0), pred_pose_rotmats[i, 0].unsqueeze(0).unsqueeze(1), pred_shape[i].unsqueeze(0),
                                    joints2D_batch[i], silhouette_batch[i], pred_cam_wp[i].unsqueeze(0))

            if not test_pose:
                pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
            else:
                pred_smpl_output = smpl(body_pose=body_pose,
                                    global_orient=global_orient,
                                    betas=pred_shape,
                                    pose2rot=False)
            pred_vertices = pred_smpl_output.vertices
            pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
            pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                            proxy_rep_input_wh)

            smpl_joints = pred_smpl_output.joints

            pred_joints_all = pred_smpl_output.joints
            pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
            pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
            pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                            proxy_rep_input_wh)

            rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                -1, -1)
            translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
            #                                                config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

            # Need to expand cam_ts for NMR i.e.  from (B, 3)
            # to
            # (B, 1, 3)
            translation_nmr = torch.unsqueeze(translation, dim=1)
            pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                        faces=faces_batch,
                                        t=translation_nmr,
                                        mode='silhouettes')

            if not is_train and save_vis:
                for i in range(len(joints2D_batch)):
                    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[i], 
                                    cam=pred_cam_wp.cpu().detach().numpy()[i], img=image_batch[i])
                    cv2.imwrite(os.path.join(player_vis_batch[i], 'player.png'), best_image)

                    #silhouettes_image = pred_silhouettes.cpu().detach().numpy()[i]*255
                    #for j in range(pred_joints2d_coco.shape[1]):
                    #    cv2.circle(silhouettes_image, (int(pred_joints2d_coco[i, j, 0]), int(pred_joints2d_coco[i, j, 1])), 5, 128, -1)
                    #cv2.imwrite(os.path.join(player_vis_batch[i], 'player_2D.png'), apply_colormap(silhouettes_image))

            pred_dict_for_loss = {'joints2D': pred_joints2d_coco,
                                    'verts': pred_vertices,
                                    'shape_params': pred_shape,
                                    'pose_params_rot_matrices': pred_pose_rotmats,
                                    'joints3D': pred_joints_coco}
            target_dict_for_loss = {'joints2D': torch.from_numpy(np.array(joints2D_batch)[:, :, :2]).float().to(device),
                                    'verts': target_vertices,
                                    'shape_params': betas,
                                    'pose_params_rot_matrices': target_pose_rotmats,
                                    'joints3D': target_joints_coco}

            # ---------------- BACKWARD PASS ----------------
            loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

            # ---------------- TRACK LOSS AND METRICS
            # ----------------
            num_train_inputs_in_batch = len(joints2D_batch)
            if not is_train:
                metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                    pred_dict_for_loss, target_dict_for_loss,
                                                    num_train_inputs_in_batch)
            else:
                metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                    pred_dict_for_loss, target_dict_for_loss,
                                                    num_train_inputs_in_batch)
        #    break
        #break

    #print(is_train)
    # ----------------------- UPDATING LOSS AND METRICS HISTORY
    # -----------------------
    metrics_tracker.update_per_epoch()

    # ----------------------------------- SAVING
    # -----------------------------------
    for metric in save_val_metrics:
        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
            metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                        metric,
                                                        metrics_tracker.history['val_' + metric][-1]))

    endtime = timeit.default_timer()
    print('epoch time: {:.3f}'.format(endtime - starttime))

def evaluate_model_relate_iuv(load_checkpoint=True, folder='Data/relate_iuv', 
                   path=player_recon_train_regressor_checkpoints_folder+'iuvp',
                   path1=player_recon_train_regressor_checkpoints_folder+'relate_iuv0',
                   postfix='', test_pose=False, save_vis=True, game_process='', opt=False):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if save_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints2D', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa', 'pose_mses', 'shape_mses']
    epochs_per_save = 10

    with open('Data/train_set.xml', 'r') as fs:
        train_set = set(json.load(fs))

    regressor = SingleInputRegressor(resnet_in_channels=20,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)

    pose_relation = PoseRelationModule()
    pose_relation.to(device)
    pose_relation.eval()

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # ----------------------- Optimiser -----------------------
    params = list(regressor.parameters()) + list(criterion.parameters())
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    # ----------------------- Resuming -----------------------
    checkpoint_path = os.path.join(path, 'best.tar')
    check_point_loaded = False
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        check_point_loaded = True

        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        #criterion.load_state_dict(checkpoint['criterion_state_dict'])
        print("Regressor loaded. Weights from:", checkpoint_path)

    else:
        checkpoint = torch.load(player_recon_check_points_path, map_location=device)
        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", player_recon_check_points_path)

    checkpoint_path = os.path.join(path1, 'best.tar')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    pose_relation.load_state_dict(checkpoint['best_model_state_dict'])
    print("Pose_relation loaded. Weights from:", checkpoint_path)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    if check_point_loaded:
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = \
            load_training_info_from_checkpoint(checkpoint, save_val_metrics)
        load_logs = False
    else:
        current_epoch = 1
        best_epoch_val_metrics = {}
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(regressor.state_dict())
        load_logs = False
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)
    # Starting training loop
    num_epochs = 10 + current_epoch
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    starttime = timeit.default_timer()
        
    print('eval')
    regressor.eval()

    games = os.listdir(player_recon_broad_view_opt_folder+postfix)
    for game in games:
        if game in train_set:
            is_train = True
        else:
            is_train = False

        if game_process != '' and game_process != game:
            continue
        game_full = os.path.join(player_crop_broad_image_folder+postfix, game)
        game_iuv = os.path.join(player_texture_iuv_folder + 'Broad', game)
        game_dst = os.path.join(player_broad_proxy_folder+postfix, game)
        game_label = os.path.join(player_recon_broad_view_opt_folder+postfix, game)
        game_box = os.path.join(player_crop_broad_folder, game)
        if save_vis and not is_train:
            game_vis = os.path.join(folder, game)
            remake_dir(game_vis)
        if not is_train:
            print('process {}'.format(game_full))
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_iuv = os.path.join(game_iuv, scene)
            scene_dst = os.path.join(game_dst, scene)
            scene_label = os.path.join(game_label, scene)
            if len(scene) == 2:
                scene_box = os.path.join(game_box, str(int(scene)//10))
            else:
                scene_box = os.path.join(game_box, scene)
            boxes_full = os.path.join(scene_box, 'boxes.xml')
            index_full = os.path.join(scene_box, 'index.xml')
            with open(boxes_full, 'r') as fs:
                boxes = json.load(fs)
            with open(index_full, 'r') as fs:
                indexes = json.load(fs)
            if save_vis and not is_train:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_full)
            body_pose_batch = []
            global_orient_batch = []
            betas_batch = []
            translation_batch = []
            joints2D_batch = []
            silhouette_batch = []
            proxy_rep_batch = []
            player_vis_batch = []
            image_batch = []
            boxes_batch = []
            iuv_rep_batch = []
            for player in players:
                player_full = os.path.join(scene_full, player)
                if (player == '1' or os.path.isfile(player_full)):
                    continue
                player_iuv = os.path.join(scene_iuv, player)
                #print(player_full)
                for iii in range(len(indexes)):
                    if indexes[iii] == player:
                        boxes_batch.append(boxes[iii])
                        break
                player_dst = os.path.join(scene_dst, player)
                player_label = os.path.join(scene_label, player)
                if save_vis and not is_train:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)
                    player_vis_batch.append(player_vis)
                        
                view_full = os.path.join(player_full, "player.png")
                iuv_full = os.path.join(player_iuv, "player.png")
                image = cv2.imread(view_full)
                image_batch.append(image)
                j2d_full = os.path.join(player_dst, 'player_j2d.xml')
                sil_full = os.path.join(player_dst, 'player_sil.npy')
                with open(j2d_full, 'r') as fs:
                    joints2D = np.array(json.load(fs))
                silhouette = np.load(sil_full)
                joints2D_batch.append(joints2D[:, :2])
                silhouette_batch.append(silhouette[0])

                label_full = os.path.join(player_label, "data.npz")
                if not os.path.exists(label_full):
                    continue
                target_param = np.load(label_full)
                body_pose = target_param['body_pose']
                global_orient = target_param['global_orient']
                betas = target_param['betas']
                translation = target_param['translation']

                body_pose_batch.append(body_pose[0])
                global_orient_batch.append(global_orient[0])
                betas_batch.append(betas[0])
                translation_batch.append(translation[0])

                proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                in_wh=proxy_rep_input_wh,
                                                out_wh=config.REGRESSOR_IMG_WH)
                proxy_rep_batch.append(proxy_rep)

                iuv_rep = cv2.imread(iuv_full)
                iuv_rep = cv2.resize(iuv_rep, (config.REGRESSOR_IMG_WH, config.REGRESSOR_IMG_WH),interpolation=cv2.INTER_LINEAR)
                iuv_rep = np.transpose(iuv_rep, [2, 0, 1])
                iuv_rep_batch.append(iuv_rep)

            faces_batch = torch.cat(len(joints2D_batch) * [faces[None, :]], dim=0)
            body_pose = torch.from_numpy(np.array(body_pose_batch)).float().to(device)
            betas = torch.from_numpy(np.array(betas_batch)).float().to(device)
            global_orient = torch.from_numpy(np.array(global_orient_batch)).float().to(device)
            translation = torch.from_numpy(np.array(translation_batch)).float().to(device)
            target_pose_rotmats = torch.cat([global_orient[:, :, :, :],
                        body_pose[:, :, :, :]],
                        dim=1)
            target_smpl_output = smpl(body_pose=body_pose,
                                    global_orient=global_orient,
                                    betas=betas,
                                    pose2rot=False)
            target_vertices = target_smpl_output.vertices
            target_joints_all = target_smpl_output.joints
            target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            boxes_batch = torch.from_numpy(np.array(boxes_batch)).float().to(device)

            proxy_rep = torch.from_numpy(np.array(proxy_rep_batch)).float().to(device)
            iuv_rep_batch = torch.from_numpy(np.array(iuv_rep_batch)).float().to(device) / 255
            iuv_rep_batch = torch.cat((proxy_rep[:,1:,:,:], iuv_rep_batch), 1)

            pred_cam_wp, pred_pose, pred_shape = regressor(iuv_rep_batch)

            # Convert pred pose to rotation matrices
            if pred_pose.shape[-1] == 24 * 3:
                pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
            elif pred_pose.shape[-1] == 24 * 6:
                pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

            pred_pose_rotmats = pose_relation([pred_pose_rotmats, boxes_batch])

            if opt:
                pred_pose_rotmats = pred_pose_rotmats.detach()
                pred_shape = pred_shape.detach()
                pred_cam_wp = pred_cam_wp.detach()
                for i in range(len(joints2D_batch)):
                    optimize_camera(pred_pose_rotmats[i, 1:].unsqueeze(0), pred_pose_rotmats[i, 0].unsqueeze(0).unsqueeze(1), pred_shape[i].unsqueeze(0),
                                    joints2D_batch[i], silhouette_batch[i], pred_cam_wp[i].unsqueeze(0))

            if not test_pose:
                pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
            else:
                pred_smpl_output = smpl(body_pose=body_pose,
                                    global_orient=global_orient,
                                    betas=pred_shape,
                                    pose2rot=False)
            pred_vertices = pred_smpl_output.vertices
            pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
            pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                            proxy_rep_input_wh)

            smpl_joints = pred_smpl_output.joints

            pred_joints_all = pred_smpl_output.joints
            pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
            pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
            pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                            proxy_rep_input_wh)

            rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                -1, -1)
            translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
            #                                                config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

            # Need to expand cam_ts for NMR i.e.  from (B, 3)
            # to
            # (B, 1, 3)
            translation_nmr = torch.unsqueeze(translation, dim=1)
            pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                        faces=faces_batch,
                                        t=translation_nmr,
                                        mode='silhouettes')

            if not is_train and save_vis:
                for i in range(len(joints2D_batch)):
                    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[i], 
                                    cam=pred_cam_wp.cpu().detach().numpy()[i], img=image_batch[i])
                    cv2.imwrite(os.path.join(player_vis_batch[i], 'player.png'), best_image)

                    #silhouettes_image = pred_silhouettes.cpu().detach().numpy()[i]*255
                    #for j in range(pred_joints2d_coco.shape[1]):
                    #    cv2.circle(silhouettes_image, (int(pred_joints2d_coco[i, j, 0]), int(pred_joints2d_coco[i, j, 1])), 5, 128, -1)
                    #cv2.imwrite(os.path.join(player_vis_batch[i], 'player_2D.png'), apply_colormap(silhouettes_image))

            pred_dict_for_loss = {'joints2D': pred_joints2d_coco,
                                    'verts': pred_vertices,
                                    'shape_params': pred_shape,
                                    'pose_params_rot_matrices': pred_pose_rotmats,
                                    'joints3D': pred_joints_coco}
            target_dict_for_loss = {'joints2D': torch.from_numpy(np.array(joints2D_batch)[:, :, :2]).float().to(device),
                                    'verts': target_vertices,
                                    'shape_params': betas,
                                    'pose_params_rot_matrices': target_pose_rotmats,
                                    'joints3D': target_joints_coco}

            # ---------------- BACKWARD PASS ----------------
            loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

            # ---------------- TRACK LOSS AND METRICS
            # ----------------
            num_train_inputs_in_batch = len(joints2D_batch)
            if not is_train:
                metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                    pred_dict_for_loss, target_dict_for_loss,
                                                    num_train_inputs_in_batch)
            else:
                metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                    pred_dict_for_loss, target_dict_for_loss,
                                                    num_train_inputs_in_batch)
        #    break
        #break

    #print(is_train)
    # ----------------------- UPDATING LOSS AND METRICS HISTORY
    # -----------------------
    metrics_tracker.update_per_epoch()

    # ----------------------------------- SAVING
    # -----------------------------------
    for metric in save_val_metrics:
        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
            metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                        metric,
                                                        metrics_tracker.history['val_' + metric][-1]))

    endtime = timeit.default_timer()
    print('epoch time: {:.3f}'.format(endtime - starttime))

def evaluate_model_hmr(folder='Data/hmr', 
                   path='Data/PlayerBroadImageHmr', save_vis=True):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if save_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa', 'pose_mses', 'shape_mses']

    with open('Data/train_set.xml', 'r') as fs:
        train_set = set(json.load(fs))

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=False,
                                                      current_epoch=1)
    # Starting training loop
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    starttime = timeit.default_timer()
        
    print('eval')

    games = os.listdir(path)
    for game in games:
        if game in train_set:
            is_train = True
        else:
            is_train = False

        game_full = os.path.join(player_crop_broad_image_folder, game)
        game_dst = os.path.join(path, game)
        game_label = os.path.join(player_recon_broad_view_opt_folder, game)
        if save_vis and not is_train:
            game_vis = os.path.join(folder, game)
            remake_dir(game_vis)
        if not is_train:
            print('process {}'.format(game_full))
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            
            scene_dst = os.path.join(game_dst, scene)
            scene_label = os.path.join(game_label, scene)
            if save_vis and not is_train:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_full)
            
            for player in players:
                player_full = os.path.join(scene_full, player)
                if (player == '1' or os.path.isfile(player_full)):
                    continue
                #print(player_full)
                player_dst = os.path.join(scene_dst, player)
                player_label = os.path.join(scene_label, player)
                if save_vis and not is_train:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)
                        
                view_full = os.path.join(player_full, "player.png")
                image = cv2.imread(view_full)

                label_full = os.path.join(player_label, "data.npz")

                target_param = np.load(label_full)
                body_pose = target_param['body_pose']
                global_orient = target_param['global_orient']
                betas = target_param['betas']
                translation = target_param['translation']

                body_pose = torch.from_numpy(body_pose).float().to(device)
                betas = torch.from_numpy(betas).float().to(device)
                global_orient = torch.from_numpy(global_orient).float().to(device)
                translation = torch.from_numpy(translation).float().to(device)
                target_pose_rotmats = torch.cat([global_orient[:, :, :, :],
                            body_pose[:, :, :, :]],
                            dim=1)
                target_smpl_output = smpl(body_pose=body_pose,
                                        global_orient=global_orient,
                                        betas=betas,
                                        pose2rot=False)
                target_vertices = target_smpl_output.vertices
                target_joints_all = target_smpl_output.joints
                target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]

                dst_full = os.path.join(player_dst, "data.npy")

                pred_param = np.load(dst_full)
                #print(pred_param.shape)
                #print(pred_param)

                pred_cam_wp = torch.from_numpy(pred_param[:,:3]).float().to(device)
                pred_pose = torch.from_numpy(pred_param[:,3:75]).float().to(device)
                pred_shape = torch.from_numpy(pred_param[:,75:]).float().to(device)

                # Convert pred pose to rotation matrices
                if pred_pose.shape[-1] == 24 * 3:
                    pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                    pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                elif pred_pose.shape[-1] == 24 * 6:
                    pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
                pred_vertices = pred_smpl_output.vertices
                pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
                pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                                proxy_rep_input_wh)

                smpl_joints = pred_smpl_output.joints

                pred_joints_all = pred_smpl_output.joints
                pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
                pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
                pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
                pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                                proxy_rep_input_wh)

                rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                    -1, -1)
                translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
                #pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
                #                                                config.FOCAL_LENGTH, proxy_rep_input_wh)
                #pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

                # Need to expand cam_ts for NMR i.e.  from (B, 3)
                # to
                # (B, 1, 3)
                #translation_nmr = torch.unsqueeze(translation, dim=1)
                #pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                #                            faces=faces_batch,
                #                            t=translation_nmr,
                #                            mode='silhouettes')

                if not is_train and save_vis:
                    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                    cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                    cv2.imwrite(os.path.join(player_vis, 'player.png'), best_image)

                pred_dict_for_loss = {
                                        'verts': pred_vertices,
                                        'shape_params': pred_shape,
                                        'pose_params_rot_matrices': pred_pose_rotmats,
                                        'joints3D': pred_joints_coco}
                target_dict_for_loss = {
                                        'verts': target_vertices,
                                        'shape_params': betas,
                                        'pose_params_rot_matrices': target_pose_rotmats,
                                        'joints3D': target_joints_coco}

                # ---------------- BACKWARD PASS ----------------
                loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

                # ---------------- TRACK LOSS AND METRICS
                # ----------------
                num_train_inputs_in_batch = 1
                if not is_train:
                    metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                        pred_dict_for_loss, target_dict_for_loss,
                                                        num_train_inputs_in_batch)
                else:
                    metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                        pred_dict_for_loss, target_dict_for_loss,
                                                        num_train_inputs_in_batch)
        #        break
        #    break
        #break

    #print(is_train)
    # ----------------------- UPDATING LOSS AND METRICS HISTORY
    # -----------------------
    metrics_tracker.update_per_epoch()

    # ----------------------------------- SAVING
    # -----------------------------------
    for metric in save_val_metrics:
        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
            metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                        metric,
                                                        metrics_tracker.history['val_' + metric][-1]))

    endtime = timeit.default_timer()
    print('epoch time: {:.3f}'.format(endtime - starttime))

def evaluate_model_spin(folder='Data/spin', 
                   path='Data/PlayerBroadImageSpin', save_vis=True):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if save_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa', 'pose_mses', 'shape_mses']

    with open('Data/train_set.xml', 'r') as fs:
        train_set = set(json.load(fs))

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=False,
                                                      current_epoch=1)
    # Starting training loop
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    starttime = timeit.default_timer()
        
    print('eval')

    games = os.listdir(path)
    for game in games:
        if game in train_set:
            is_train = True
        else:
            is_train = False

        game_full = os.path.join(player_crop_broad_image_folder, game)
        game_dst = os.path.join(path, game)
        game_label = os.path.join(player_recon_broad_view_opt_folder, game)
        if save_vis and not is_train:
            game_vis = os.path.join(folder, game)
            remake_dir(game_vis)
        if not is_train:
            print('process {}'.format(game_full))
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            
            scene_dst = os.path.join(game_dst, scene)
            scene_label = os.path.join(game_label, scene)
            if save_vis and not is_train:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_full)
            
            for player in players:
                player_full = os.path.join(scene_full, player)
                if (player == '1' or os.path.isfile(player_full)):
                    continue
                #print(player_full)
                player_dst = os.path.join(scene_dst, player)
                player_label = os.path.join(scene_label, player)
                if save_vis and not is_train:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)
                        
                view_full = os.path.join(player_full, "player.png")
                image = cv2.imread(view_full)

                label_full = os.path.join(player_label, "data.npz")

                target_param = np.load(label_full)
                body_pose = target_param['body_pose']
                global_orient = target_param['global_orient']
                betas = target_param['betas']
                translation = target_param['translation']

                body_pose = torch.from_numpy(body_pose).float().to(device)
                betas = torch.from_numpy(betas).float().to(device)
                global_orient = torch.from_numpy(global_orient).float().to(device)
                translation = torch.from_numpy(translation).float().to(device)
                target_pose_rotmats = torch.cat([global_orient[:, :, :, :],
                            body_pose[:, :, :, :]],
                            dim=1)
                target_smpl_output = smpl(body_pose=body_pose,
                                        global_orient=global_orient,
                                        betas=betas,
                                        pose2rot=False)
                target_vertices = target_smpl_output.vertices
                target_joints_all = target_smpl_output.joints
                target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]

                dst_full = os.path.join(player_dst, "data.npz")

                init_param = np.load(dst_full)

                pred_rotmat = init_param['pred_rotmat']
                pred_betas = init_param['pred_betas']
                pred_camera = init_param['pred_camera']

                pred_pose_rotmats = torch.from_numpy(pred_rotmat).float().to(device).unsqueeze(0)
                pred_shape = torch.from_numpy(pred_betas).float().to(device).unsqueeze(0)
                pred_cam_wp = torch.from_numpy(pred_camera).float().to(device).unsqueeze(0)

                pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
                pred_vertices = pred_smpl_output.vertices
                pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
                pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                                proxy_rep_input_wh)

                smpl_joints = pred_smpl_output.joints

                pred_joints_all = pred_smpl_output.joints
                pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
                pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
                pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
                pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                                proxy_rep_input_wh)

                rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                    -1, -1)
                translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
                #pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
                #                                                config.FOCAL_LENGTH, proxy_rep_input_wh)
                #pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

                # Need to expand cam_ts for NMR i.e.  from (B, 3)
                # to
                # (B, 1, 3)
                #translation_nmr = torch.unsqueeze(translation, dim=1)
                #pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                #                            faces=faces_batch,
                #                            t=translation_nmr,
                #                            mode='silhouettes')

                if not is_train and save_vis:
                    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                    cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                    cv2.imwrite(os.path.join(player_vis, 'player.png'), best_image)

                pred_dict_for_loss = {
                                        'verts': pred_vertices,
                                        'shape_params': pred_shape,
                                        'pose_params_rot_matrices': pred_pose_rotmats,
                                        'joints3D': pred_joints_coco}
                target_dict_for_loss = {
                                        'verts': target_vertices,
                                        'shape_params': betas,
                                        'pose_params_rot_matrices': target_pose_rotmats,
                                        'joints3D': target_joints_coco}

                # ---------------- BACKWARD PASS ----------------
                loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

                # ---------------- TRACK LOSS AND METRICS
                # ----------------
                num_train_inputs_in_batch = 1
                if not is_train:
                    metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                        pred_dict_for_loss, target_dict_for_loss,
                                                        num_train_inputs_in_batch)
                else:
                    metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                        pred_dict_for_loss, target_dict_for_loss,
                                                        num_train_inputs_in_batch)
        #        break
        #    break
        #break

    #print(is_train)
    # ----------------------- UPDATING LOSS AND METRICS HISTORY
    # -----------------------
    metrics_tracker.update_per_epoch()

    # ----------------------------------- SAVING
    # -----------------------------------
    for metric in save_val_metrics:
        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
            metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                        metric,
                                                        metrics_tracker.history['val_' + metric][-1]))

    endtime = timeit.default_timer()
    print('epoch time: {:.3f}'.format(endtime - starttime))

def smpl_vis(folder='Data/RealPlayerImageHmrVis', 
                   path='Data/RealHmr', save_vis=True):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if save_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa']

    with open('Data/train_set.xml', 'r') as fs:
        train_set = set(json.load(fs))

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=False,
                                                      current_epoch=1)
    # Starting training loop
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    starttime = timeit.default_timer()
        
    print('eval')

    games = os.listdir(path)
    for game in games:
        is_train = False

        game_full = os.path.join(real_images_player, game)
        game_dst = os.path.join(path, game)
        game_label = os.path.join(player_recon_broad_view_opt_folder, game)
        if save_vis and not is_train:
            game_vis = os.path.join(folder, game)
            remake_dir(game_vis)
        if not is_train:
            print('process {}'.format(game_full))
        scenes = os.listdir(game_dst)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            
            scene_dst = os.path.join(game_dst, scene)
            scene_label = os.path.join(game_label, scene)
            if save_vis and not is_train:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_dst)
            
            for player in players:
                player_full = os.path.join(scene_full, player)
                player_dst = os.path.join(scene_dst, player)
                player_label = os.path.join(scene_label, player)
                if save_vis and not is_train:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)
                        
                view_full = os.path.join(player_full, "player.png")
                image = cv2.imread(view_full)

                dst_full = os.path.join(player_dst, "data.npy")

                pred_param = np.load(dst_full)
                #print(pred_param.shape)
                #print(pred_param)

                pred_cam_wp = torch.from_numpy(pred_param[:,:3]).float().to(device)
                pred_pose = torch.from_numpy(pred_param[:,3:75]).float().to(device)
                pred_shape = torch.from_numpy(pred_param[:,75:]).float().to(device)

                # Convert pred pose to rotation matrices
                if pred_pose.shape[-1] == 24 * 3:
                    pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                    pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                elif pred_pose.shape[-1] == 24 * 6:
                    pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
                pred_vertices = pred_smpl_output.vertices
                pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
                pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                                proxy_rep_input_wh)

                smpl_joints = pred_smpl_output.joints

                pred_joints_all = pred_smpl_output.joints
                pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
                pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
                pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
                pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                                proxy_rep_input_wh)

                rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                    -1, -1)
                translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

                if not is_train and save_vis:
                    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                    cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                    cv2.imwrite(os.path.join(player_vis, 'player.png'), best_image)

    endtime = timeit.default_timer()
    print('epoch time: {:.3f}'.format(endtime - starttime))

def spin_vis(folder='Data/RealPlayerImageSPINVis', 
                   path='Data/RealPlayerImageSPIN', save_vis=True):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if save_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa']

    with open('Data/train_set.xml', 'r') as fs:
        train_set = set(json.load(fs))

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=False,
                                                      current_epoch=1)
    # Starting training loop
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    starttime = timeit.default_timer()
        
    print('eval')

    games = os.listdir(path)
    for game in games:
        is_train = False

        game_full = os.path.join(real_images_player, game)
        game_dst = os.path.join(path, game)
        game_label = os.path.join(player_recon_broad_view_opt_folder, game)
        if save_vis and not is_train:
            game_vis = os.path.join(folder, game)
            remake_dir(game_vis)
        if not is_train:
            print('process {}'.format(game_full))
        scenes = os.listdir(game_dst)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            
            scene_dst = os.path.join(game_dst, scene)
            scene_label = os.path.join(game_label, scene)
            if save_vis and not is_train:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_dst)
            
            for player in players:
                player_full = os.path.join(scene_full, player)
                player_dst = os.path.join(scene_dst, player)
                player_label = os.path.join(scene_label, player)
                if save_vis and not is_train:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)
                        
                view_full = os.path.join(player_full, "player.png")
                image = cv2.imread(view_full)

                dst_full = os.path.join(player_dst, "data.npz")

                init_param = np.load(dst_full)
                pred_rotmat = init_param['pred_rotmat']
                pred_betas = init_param['pred_betas']
                pred_camera = init_param['pred_camera']

                pred_pose_rotmats = torch.from_numpy(pred_rotmat).float().to(device).unsqueeze(0)
                pred_shape = torch.from_numpy(pred_betas).float().to(device).unsqueeze(0)
                pred_cam_wp = torch.from_numpy(pred_camera).float().to(device).unsqueeze(0)

                pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
                pred_vertices = pred_smpl_output.vertices
                pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
                pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                                proxy_rep_input_wh)

                smpl_joints = pred_smpl_output.joints

                pred_joints_all = pred_smpl_output.joints
                pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
                pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
                pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
                pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                                proxy_rep_input_wh)

                rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                    -1, -1)
                translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

                if not is_train and save_vis:
                    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                    cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                    cv2.imwrite(os.path.join(player_vis, 'player.png'), best_image)

    endtime = timeit.default_timer()
    print('epoch time: {:.3f}'.format(endtime - starttime))

def init_loss_and_metric(joints=True, silhoutte=True):
    losses_on = []
    metrics_to_track = []
    if (joints):
        losses_on.append('joints2D')
        metrics_to_track.append('joints2D_l2es')
    if (silhoutte):
        losses_on.append('silhouette')
    metrics_to_track.append('silhouette_iou')
    init_loss_weights = {'joints2D': 1.0, 'silhouette': 100.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    save_val_metrics = metrics_to_track
    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                                init_loss_weights=init_loss_weights,
                                                                reduction='mean')
    criterion.to(device)

    # Instantiate metrics tracker.
    load_logs = False
    current_epoch = 1
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                        metrics_to_track=metrics_to_track,
                                                        img_wh=config.REGRESSOR_IMG_WH,
                                                        log_path=log_path,
                                                        load_logs=load_logs,
                                                        current_epoch=current_epoch)
    metrics_tracker.initialise_loss_metric_sums()
    return criterion, metrics_tracker, save_val_metrics

def optimize_camera(body_pose, global_orient, betas, joints2D, silhouette, cam_wp):
    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)
    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)
    faces = torch.cat(1 * [faces[None, :]], dim=0)
    
    cam_wp.requires_grad = True
    global_orient.requires_grad = True
    params = [cam_wp, global_orient]
    optimiser = optim.Adam(params, lr=0.01)
    criterion, metrics_tracker, save_val_metrics = init_loss_and_metric(True, False)
    translation = convert_weak_perspective_to_camera_translation_torch(cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
    for epoch in range(1, 50 + 1):
        metrics_tracker.initialise_loss_metric_sums()
        pred_smpl_output = smpl(body_pose=body_pose,
                                global_orient=global_orient,
                                betas=betas,
                                pose2rot=False)

        pred_joints_all = pred_smpl_output.joints
        pred_joints2d_coco = orthographic_project_torch(pred_joints_all, cam_wp)
        pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
        pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                        proxy_rep_input_wh)

        # Need to expand cam_ts for NMR i.e.  from (B, 3) to
        # (B, 1, 3)
        translation_nmr = torch.unsqueeze(translation, dim=1)
        pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                    faces=faces,
                                    t=translation_nmr,
                                    mode='silhouettes')

        pred_dict_for_loss = {'joints2D': pred_joints2d_coco[0],
                                'silhouette': pred_silhouettes}
        target_dict_for_loss = {'joints2D': torch.from_numpy(joints2D[:,:2]).float().to(device),
                                'silhouette': torch.from_numpy(silhouette).float().to(device).unsqueeze(0)}

        # ---------------- BACKWARD PASS ----------------
        optimiser.zero_grad()
        loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

        # ---------------- TRACK LOSS AND METRICS
        # ----------------
        num_train_inputs_in_batch = 1
        metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                            pred_dict_for_loss, target_dict_for_loss,
                                            num_train_inputs_in_batch)
        metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                            pred_dict_for_loss, target_dict_for_loss,
                                            num_train_inputs_in_batch)

        loss.backward()
        optimiser.step()
        translation = convert_weak_perspective_to_camera_translation_torch(cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

        metrics_tracker.update_per_epoch()

        #if epoch == 1:
        #    for metric in save_val_metrics:
        #        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
        #            metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
        #                                                        metric,
        #                                                        metrics_tracker.history['val_' + metric][-1]))

    #print('Finished translation optimization.')
    #for metric in save_val_metrics:
    #    print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
    #        metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
    #                                                    metric,
    #                                                    metrics_tracker.history['val_' + metric][-1]))

def evaluate_model_2d_iuv_p(load_checkpoint=True, folder='Data/real_iuvp', blur = '',
                   path=player_recon_train_regressor_checkpoints_folder+'iuvp',
                   save_vis=True, use_relate = False, opt = True,
                   result_folder = '', hmr=False, spin=False):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    path = path + blur
    if save_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints2D', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa']
    epochs_per_save = 10

    regressor = SingleInputRegressor(resnet_in_channels=20,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    print('eval')
    regressor.eval()

    if use_relate:
        pose_relation = PoseRelationModule()
        pose_relation.to(device)
        pose_relation.eval()

        checkpoint_path = os.path.join(player_recon_train_regressor_checkpoints_folder+'iuvp_relate'+blur, 'best.tar')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        pose_relation.load_state_dict(checkpoint['best_model_state_dict'])
        print("Pose_relation loaded. Weights from:", checkpoint_path)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # ----------------------- Optimiser -----------------------
    params = list(regressor.parameters()) + list(criterion.parameters())
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    # ----------------------- Resuming -----------------------
    checkpoint_path = os.path.join(path, 'best.tar')
    check_point_loaded = False
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        check_point_loaded = True

        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", checkpoint_path)

    else:
        checkpoint = torch.load(player_recon_check_points_path, map_location=device)
        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", player_recon_check_points_path)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    if check_point_loaded:
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = \
            load_training_info_from_checkpoint(checkpoint, save_val_metrics)
        load_logs = False
    else:
        current_epoch = 1
        best_epoch_val_metrics = {}
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(regressor.state_dict())
        load_logs = False
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)
    # Starting training loop
    num_epochs = 10 + current_epoch
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    starttime = timeit.default_timer()

    games = os.listdir(real_images_player_proxy)
    is_train = False
    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    for game in games:

        game_full = os.path.join(real_images_player, game)
        game_dst = os.path.join(real_images_player_proxy+'unrefine', game)
        game_dst_refine = os.path.join(real_images_player_proxy, game)
        game_box = os.path.join(real_images_box, game)
        game_result = os.path.join(result_folder, game)
        game_iuv = os.path.join(player_texture_iuv_folder + 'Real', game)
        if save_vis and not is_train:
            game_vis = os.path.join(folder, game)
            remake_dir(game_vis)
        scenes = os.listdir(game_dst_refine)
        for (ii, scene) in enumerate(scenes):
            print('{} / {}'.format(ii, len(scenes)))
            starttime_scene = timeit.default_timer()
            scene_full = os.path.join(game_full, scene)
            
            scene_dst = os.path.join(game_dst, scene)
            scene_dst_refine = os.path.join(game_dst_refine, scene)
            scene_box = os.path.join(game_box, scene)
            scene_iuv = os.path.join(game_iuv, scene)
            scene_result = os.path.join(game_result, scene)
            boxes_full = os.path.join(scene_box, 'boxes.xml')
            with open(boxes_full, 'r') as fs:
                boxes = json.load(fs)
            if save_vis and not is_train:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_dst_refine)
            body_pose_batch = []
            global_orient_batch = []
            betas_batch = []
            translation_batch = []
            joints2D_batch = []
            silhouette_batch = []
            proxy_rep_batch = []
            player_vis_batch = []
            image_batch = []
            boxes_batch = []

            pred_cam_wp_batch = []
            pred_pose_batch = []
            pred_shape_batch = []
            iuv_rep_batch = []
            for player in players:
                player_iuv = os.path.join(scene_iuv, player)
                iuv_full = os.path.join(player_iuv, "player.png")
                if not os.path.exists(iuv_full):
                    continue
                player_full = os.path.join(scene_full, player)
                player_dst = os.path.join(scene_dst, player)
                player_dst_refine = os.path.join(scene_dst_refine, player)
                player_result = os.path.join(scene_result, player)
                
                boxes_batch.append(boxes[int(player)-2])
                if save_vis and not is_train:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)
                    player_vis_batch.append(player_vis)
                        
                view_full = os.path.join(player_full, "player.png")
                
                image = cv2.imread(view_full)
                image_batch.append(image)
                j2d_full = os.path.join(player_dst, 'player_j2d.xml')
                sil_full = os.path.join(player_dst, 'player_sil.npy')
                j2d_full_refine = os.path.join(player_dst_refine, 'player_j2d.xml')
                with open(j2d_full, 'r') as fs:
                    joints2D = np.array(json.load(fs))
                with open(j2d_full_refine, 'r') as fs:
                    joints2D_refine = np.array(json.load(fs))
                silhouette = np.load(sil_full)
                joints2D_batch.append(joints2D_refine[:, :2])
                #print(silhouette.shape)
                silhouette_batch.append(silhouette)

                proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                in_wh=proxy_rep_input_wh,
                                                out_wh=config.REGRESSOR_IMG_WH)
                proxy_rep_batch.append(proxy_rep)

                #print(iuv_full)
                iuv_rep = cv2.imread(iuv_full)
                iuv_rep = cv2.resize(iuv_rep, (config.REGRESSOR_IMG_WH, config.REGRESSOR_IMG_WH),interpolation=cv2.INTER_LINEAR)
                #print(iuv_rep.shape)
                iuv_rep = np.transpose(iuv_rep, [2, 0, 1])
                iuv_rep_batch.append(iuv_rep)

                if hmr:
                    dst_full = os.path.join(player_result, "data.npy")
                    pred_param = np.load(dst_full)

                    pred_cam_wp = torch.from_numpy(pred_param[:,:3]).float().to(device)
                    pred_pose = torch.from_numpy(pred_param[:,3:75]).float().to(device)
                    pred_shape = torch.from_numpy(pred_param[:,75:]).float().to(device)

                    # Convert pred pose to rotation matrices
                    if pred_pose.shape[-1] == 24 * 3:
                        pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                        pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                    elif pred_pose.shape[-1] == 24 * 6:
                        pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                    pred_cam_wp_batch.append(pred_cam_wp[0])
                    pred_pose_batch.append(pred_pose_rotmats[0])
                    pred_shape_batch.append(pred_shape[0])
                if spin:
                    dst_full = os.path.join(player_result, "data.npz")

                    init_param = np.load(dst_full)
                    pred_rotmat = init_param['pred_rotmat']
                    pred_betas = init_param['pred_betas']
                    pred_camera = init_param['pred_camera']

                    pred_pose_rotmats = torch.from_numpy(pred_rotmat).float().to(device)
                    pred_shape = torch.from_numpy(pred_betas).float().to(device)
                    pred_cam_wp = torch.from_numpy(pred_camera).float().to(device)

                    pred_cam_wp_batch.append(pred_cam_wp)
                    pred_pose_batch.append(pred_pose_rotmats)
                    pred_shape_batch.append(pred_shape)

            faces_batch = torch.cat(len(joints2D_batch) * [faces[None, :]], dim=0)
            boxes_batch = torch.from_numpy(np.array(boxes_batch)).float().to(device)

            proxy_rep = torch.from_numpy(np.array(proxy_rep_batch)).float().to(device)
            iuv_rep_batch = torch.from_numpy(np.array(iuv_rep_batch)).float().to(device) / 255
            iuv_rep_batch = torch.cat((proxy_rep[:,1:,:,:], iuv_rep_batch), 1)

            if hmr or spin:
                pred_cam_wp = torch.stack(pred_cam_wp_batch)
                pred_pose_rotmats = torch.stack(pred_pose_batch)
                pred_shape = torch.stack(pred_shape_batch)
            else:
                with torch.no_grad():
                    pred_cam_wp, pred_pose, pred_shape = regressor(iuv_rep_batch)

                # Convert pred pose to rotation matrices
                if pred_pose.shape[-1] == 24 * 3:
                    pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                    pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                elif pred_pose.shape[-1] == 24 * 6:
                    pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

            if use_relate:
                pred_pose_rotmats = pose_relation([pred_pose_rotmats, boxes_batch])

            if opt:
                pred_pose_rotmats = pred_pose_rotmats.detach()
                pred_shape = pred_shape.detach()
                pred_cam_wp = pred_cam_wp.detach()
                for i in range(len(joints2D_batch)):
                    optimize_camera(pred_pose_rotmats[i, 1:].unsqueeze(0), pred_pose_rotmats[i, 0].unsqueeze(0).unsqueeze(1), pred_shape[i].unsqueeze(0),
                                    joints2D_batch[i], silhouette_batch[i], pred_cam_wp[i].unsqueeze(0))

            pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
            pred_vertices = pred_smpl_output.vertices
            pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
            pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                            proxy_rep_input_wh)

            smpl_joints = pred_smpl_output.joints

            pred_joints_all = pred_smpl_output.joints
            pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
            pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
            pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                            proxy_rep_input_wh)

            rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                -1, -1)
            translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
            #                                                config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

            # Need to expand cam_ts for NMR i.e.  from (B, 3)
            # to
            # (B, 1, 3)
            translation_nmr = torch.unsqueeze(translation, dim=1)
            pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                        faces=faces_batch,
                                        t=translation_nmr,
                                        mode='silhouettes')

            for i in range(len(joints2D_batch)):
                keypoints = pred_joints2d_coco.cpu().detach().numpy()[i].astype('int32')
                silh_error, iou_vis = compute_silh_error_metrics(
                    pred_silhouettes.cpu().detach().numpy()[i], silhouette_batch[i], False)
                joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D_batch[i])
                        
                silh_mean_error_init += silh_error['iou']
                joint_mean_error_init += joints_error

                num_counter += 1

                #if i == 0:
                #    cv2.imwrite('Data/1.png', silhouette_batch[i] * 128)

            if not is_train and save_vis:
                for i in range(len(joints2D_batch)):
                    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[i], 
                                    cam=pred_cam_wp.cpu().detach().numpy()[i], img=image_batch[i])
                    cv2.imwrite(os.path.join(player_vis_batch[i], 'player.png'), best_image)
            endtime_scene = timeit.default_timer()
            print('scene time: {:.3f}'.format(endtime_scene - starttime_scene))

        #    break
        #break

    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
    endtime = timeit.default_timer()
    print('epoch time: {:.3f}'.format(endtime - starttime))

def evaluate_model_2d(load_checkpoint=False, folder='Data/real_STR', 
                   path=player_recon_train_regressor_checkpoints_folder,
                   save_vis=True, use_relate = False, opt = True,
                   result_folder = '', hmr=False, spin=False, unrefine=True):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if save_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints2D', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa']
    epochs_per_save = 10

    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    print('eval')
    regressor.eval()

    if use_relate:
        pose_relation = PoseRelationModule()
        pose_relation.to(device)
        pose_relation.eval()

        checkpoint_path = os.path.join(player_recon_train_regressor_checkpoints_folder+'relate_aug1', 'best.tar')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        pose_relation.load_state_dict(checkpoint['best_model_state_dict'])
        print("Pose_relation loaded. Weights from:", checkpoint_path)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # ----------------------- Optimiser -----------------------
    params = list(regressor.parameters()) + list(criterion.parameters())
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    # ----------------------- Resuming -----------------------
    checkpoint_path = os.path.join(path, 'best.tar')
    check_point_loaded = False
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        check_point_loaded = True

        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", checkpoint_path)

    else:
        checkpoint = torch.load(player_recon_check_points_path, map_location=device)
        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", player_recon_check_points_path)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    if check_point_loaded:
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = \
            load_training_info_from_checkpoint(checkpoint, save_val_metrics)
        load_logs = False
    else:
        current_epoch = 1
        best_epoch_val_metrics = {}
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(regressor.state_dict())
        load_logs = False
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)
    # Starting training loop
    num_epochs = 10 + current_epoch
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    starttime = timeit.default_timer()

    games = os.listdir(real_images_player_proxy)
    is_train = False
    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    for game in games:

        game_full = os.path.join(real_images_player, game)
        game_dst = os.path.join(real_images_player_proxy, game)
        game_unrefine = os.path.join(real_images_player_proxy+'unrefine' if unrefine else real_images_player_proxy, game)
        game_box = os.path.join(real_images_box, game)
        game_result = os.path.join(result_folder, game)
        if save_vis and not is_train:
            game_vis = os.path.join(folder, game)
            remake_dir(game_vis)
        scenes = os.listdir(game_dst)
        for scene in scenes:
            print(scene)
            scene_full = os.path.join(game_full, scene)
            
            scene_dst = os.path.join(game_dst, scene)
            scene_unrefine = os.path.join(game_unrefine, scene)
            scene_box = os.path.join(game_box, scene)
            scene_result = os.path.join(game_result, scene)
            boxes_full = os.path.join(scene_box, 'boxes.xml')
            with open(boxes_full, 'r') as fs:
                boxes = json.load(fs)
            if save_vis and not is_train:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_dst)
            body_pose_batch = []
            global_orient_batch = []
            betas_batch = []
            translation_batch = []
            joints2D_batch = []
            silhouette_batch = []
            joints2D_batch_unrefine = []
            silhouette_batch_unrefine = []
            proxy_rep_batch = []
            player_vis_batch = []
            image_batch = []
            boxes_batch = []

            pred_cam_wp_batch = []
            pred_pose_batch = []
            pred_shape_batch = []
            for player in players:
                player_full = os.path.join(scene_full, player)
                player_dst = os.path.join(scene_dst, player)
                player_unrefine = os.path.join(scene_unrefine, player)
                player_result = os.path.join(scene_result, player)
                boxes_batch.append(boxes[int(player)-2])
                if save_vis and not is_train:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)
                    player_vis_batch.append(player_vis)
                        
                view_full = os.path.join(player_full, "player.png")
                image = cv2.imread(view_full)
                image_batch.append(image)

                j2d_full = os.path.join(player_dst, 'player_j2d.xml')
                sil_full = os.path.join(player_dst, 'player_sil.npy')
                with open(j2d_full, 'r') as fs:
                    joints2D = np.array(json.load(fs))
                silhouette = np.load(sil_full)
                joints2D_batch.append(joints2D[:, :2])
                silhouette_batch.append(silhouette)

                j2d_full_unrefine = os.path.join(player_unrefine, 'player_j2d.xml')
                sil_full_unrefine = os.path.join(player_unrefine, 'player_sil.npy')
                with open(j2d_full_unrefine, 'r') as fs:
                    joints2D_unrefine = np.array(json.load(fs))
                silhouette_unrefine = np.load(sil_full_unrefine)
                joints2D_batch_unrefine.append(joints2D_unrefine[:, :2])
                silhouette_batch_unrefine.append(silhouette_unrefine)

                proxy_rep = create_proxy_representation(silhouette_unrefine, joints2D_unrefine,
                                                in_wh=proxy_rep_input_wh,
                                                out_wh=config.REGRESSOR_IMG_WH)
                proxy_rep_batch.append(proxy_rep)

                if hmr:
                    dst_full = os.path.join(player_result, "data.npy")
                    pred_param = np.load(dst_full)

                    pred_cam_wp = torch.from_numpy(pred_param[:,:3]).float().to(device)
                    pred_pose = torch.from_numpy(pred_param[:,3:75]).float().to(device)
                    pred_shape = torch.from_numpy(pred_param[:,75:]).float().to(device)

                    # Convert pred pose to rotation matrices
                    if pred_pose.shape[-1] == 24 * 3:
                        pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                        pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                    elif pred_pose.shape[-1] == 24 * 6:
                        pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                    pred_cam_wp_batch.append(pred_cam_wp[0])
                    pred_pose_batch.append(pred_pose_rotmats[0])
                    pred_shape_batch.append(pred_shape[0])
                if spin:
                    dst_full = os.path.join(player_result, "data.npz")

                    init_param = np.load(dst_full)
                    pred_rotmat = init_param['pred_rotmat']
                    pred_betas = init_param['pred_betas']
                    pred_camera = init_param['pred_camera']

                    pred_pose_rotmats = torch.from_numpy(pred_rotmat).float().to(device)
                    pred_shape = torch.from_numpy(pred_betas).float().to(device)
                    pred_cam_wp = torch.from_numpy(pred_camera).float().to(device)

                    pred_cam_wp_batch.append(pred_cam_wp)
                    pred_pose_batch.append(pred_pose_rotmats)
                    pred_shape_batch.append(pred_shape)

            faces_batch = torch.cat(len(joints2D_batch) * [faces[None, :]], dim=0)
            boxes_batch = torch.from_numpy(np.array(boxes_batch)).float().to(device)

            proxy_rep = torch.from_numpy(np.array(proxy_rep_batch)).float().to(device)

            if hmr or spin:
                pred_cam_wp = torch.stack(pred_cam_wp_batch)
                pred_pose_rotmats = torch.stack(pred_pose_batch)
                pred_shape = torch.stack(pred_shape_batch)
            else:
                with torch.no_grad():
                    pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep)

                # Convert pred pose to rotation matrices
                if pred_pose.shape[-1] == 24 * 3:
                    pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                    pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                elif pred_pose.shape[-1] == 24 * 6:
                    pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

            if use_relate:
                pred_pose_rotmats = pose_relation([pred_pose_rotmats, boxes_batch])

            if opt:
                pred_pose_rotmats = pred_pose_rotmats.detach()
                pred_shape = pred_shape.detach()
                pred_cam_wp = pred_cam_wp.detach()
                for i in range(len(joints2D_batch)):
                    optimize_camera(pred_pose_rotmats[i, 1:].unsqueeze(0), pred_pose_rotmats[i, 0].unsqueeze(0).unsqueeze(1), pred_shape[i].unsqueeze(0),
                                    joints2D_batch[i], silhouette_batch[i], pred_cam_wp[i].unsqueeze(0))

            pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
            pred_vertices = pred_smpl_output.vertices
            pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
            pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                            proxy_rep_input_wh)

            smpl_joints = pred_smpl_output.joints

            pred_joints_all = pred_smpl_output.joints
            pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
            pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
            pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                            proxy_rep_input_wh)

            rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                -1, -1)
            translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
            #                                                config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

            # Need to expand cam_ts for NMR i.e.  from (B, 3)
            # to
            # (B, 1, 3)
            translation_nmr = torch.unsqueeze(translation, dim=1)
            pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                        faces=faces_batch,
                                        t=translation_nmr,
                                        mode='silhouettes')

            for i in range(len(joints2D_batch)):
                keypoints = pred_joints2d_coco.cpu().detach().numpy()[i].astype('int32')
                silh_error, iou_vis = compute_silh_error_metrics(
                    pred_silhouettes.cpu().detach().numpy()[i], silhouette_batch[i], False)
                joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D_batch[i])
                        
                silh_mean_error_init += silh_error['iou']
                joint_mean_error_init += joints_error

                num_counter += 1

                #if i == 0:
                #    cv2.imwrite('Data/1.png', silhouette_batch[i] * 128)

            if not is_train and save_vis:
                for i in range(len(joints2D_batch)):
                    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[i], 
                                    cam=pred_cam_wp.cpu().detach().numpy()[i], img=image_batch[i])
                    cv2.imwrite(os.path.join(player_vis_batch[i], 'player.png'), best_image)

        #    break
        #break

    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
    endtime = timeit.default_timer()
    print('epoch time: {:.3f}'.format(endtime - starttime))

def train_regressor_iuv(load_checkpoint=True, data_argument=False):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if not os.path.exists(player_recon_train_regressor_checkpoints_folder+'iuv'):
        os.makedirs(player_recon_train_regressor_checkpoints_folder, exist_ok=True)
    if not os.path.exists(player_recon_train_regressor_logs_folder+'iuv'):
        os.makedirs(player_recon_train_regressor_logs_folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints2D', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa']
    epochs_per_save = 10

    with open('Data/train_set.xml', 'r') as fs:
        train_set = set(json.load(fs))

    regressor = SingleInputRegressor(resnet_in_channels=3,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # ----------------------- Optimiser -----------------------
    params = list(regressor.parameters()) + list(criterion.parameters())
    optimiser = optim.Adam(params, lr=0.0001)

    # ----------------------- Resuming -----------------------
    checkpoint_path = os.path.join(player_recon_train_regressor_checkpoints_folder+'iuv', 'best.tar')
    check_point_loaded = False
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        check_point_loaded = True

        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        #criterion.load_state_dict(checkpoint['criterion_state_dict'])
        print("Regressor loaded. Weights from:", checkpoint_path)

    #else:
    #    checkpoint = torch.load(player_recon_check_points_path, map_location=device)
    #    regressor.load_state_dict(checkpoint['best_model_state_dict'])
    #    print("Regressor loaded. Weights from:", player_recon_check_points_path)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    if check_point_loaded:
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = \
            load_training_info_from_checkpoint(checkpoint, save_val_metrics)
        load_logs = False
    else:
        current_epoch = 1
        best_epoch_val_metrics = {}
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(regressor.state_dict())
        load_logs = False
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)
    # Starting training loop
    num_epochs = 300 + current_epoch
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    for epoch in range(current_epoch, num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        starttime = timeit.default_timer()
        
        for is_train in [False, True]:
            if (is_train):
                print('Training.')
                regressor.train()
            else:
                print('eval')
                regressor.eval()
                #regressor.fix()

            games = os.listdir(player_recon_broad_view_opt_folder)
            if is_train:
                random.shuffle(games)
            for game in games:
                if is_train:
                    if game not in train_set:
                        continue
                else:
                    if game in train_set:
                        continue

                game_full = os.path.join(player_crop_broad_image_folder, game)
                game_iuv = os.path.join(player_texture_iuv_folder + 'Broad', game)
                game_dst = os.path.join(player_broad_proxy_folder, game)
                game_label = os.path.join(player_recon_broad_view_opt_folder, game)
                scenes = os.listdir(game_full)
                if is_train:
                    random.shuffle(scenes)
                for scene in scenes:
                    scene_full = os.path.join(game_full, scene)
                    #print('process {}'.format(scene_full))
                    scene_iuv = os.path.join(game_iuv, scene)
                    scene_dst = os.path.join(game_dst, scene)
                    scene_label = os.path.join(game_label, scene)
                    players = os.listdir(scene_full)

                    body_pose_batch = []
                    global_orient_batch = []
                    betas_batch = []
                    translation_batch = []
                    joints2D_batch = []
                    silhouette_batch = []
                    proxy_rep_batch = []
                    iuv_rep_batch = []
                    for player in players:
                        player_full = os.path.join(scene_full, player)
                        if (player == '1' or os.path.isfile(player_full)):
                            continue
                        #print(player_full)
                        player_iuv = os.path.join(scene_iuv, player)
                        player_dst = os.path.join(scene_dst, player)
                        player_label = os.path.join(scene_label, player)
                        
                        view_full = os.path.join(player_full, "player.png")
                        iuv_full = os.path.join(player_iuv, "player.png")
                        if not os.path.exists(iuv_full):
                            continue
                        j2d_full = os.path.join(player_dst, 'player_j2d.xml')
                        sil_full = os.path.join(player_dst, 'player_sil.npy')
                        with open(j2d_full, 'r') as fs:
                            joints2D = np.array(json.load(fs))
                        silhouette = np.load(sil_full)
                        joints2D_batch.append(joints2D[:, :2])
                        silhouette_batch.append(silhouette[0])

                        label_full = os.path.join(player_label, "data.npz")
                        if not os.path.exists(label_full):
                            continue
                        target_param = np.load(label_full)
                        body_pose = target_param['body_pose']
                        global_orient = target_param['global_orient']
                        betas = target_param['betas']
                        translation = target_param['translation']

                        body_pose_batch.append(body_pose[0])
                        global_orient_batch.append(global_orient[0])
                        betas_batch.append(betas[0])
                        translation_batch.append(translation[0])

                        proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                        in_wh=proxy_rep_input_wh,
                                                        out_wh=config.REGRESSOR_IMG_WH)
                        proxy_rep_batch.append(proxy_rep)

                        iuv_rep = cv2.imread(iuv_full)
                        iuv_rep = np.transpose(iuv_rep, [2, 0, 1])
                        iuv_rep_batch.append(iuv_rep)

                    faces_batch = torch.cat(len(joints2D_batch) * [faces[None, :]], dim=0)
                    body_pose = torch.from_numpy(np.array(body_pose_batch)).float().to(device)
                    betas = torch.from_numpy(np.array(betas_batch)).float().to(device)
                    global_orient = torch.from_numpy(np.array(global_orient_batch)).float().to(device)
                    translation = torch.from_numpy(np.array(translation_batch)).float().to(device)
                    target_pose_rotmats = torch.cat([global_orient[:, :, :, :],
                                body_pose[:, :, :, :]],
                                dim=1)
                    target_smpl_output = smpl(body_pose=body_pose,
                                            global_orient=global_orient,
                                            betas=betas,
                                            pose2rot=False)
                    target_vertices = target_smpl_output.vertices
                    target_joints_all = target_smpl_output.joints
                    target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]

                    proxy_rep = torch.from_numpy(np.array(proxy_rep_batch)).float().to(device)
                    iuv_rep_batch = torch.from_numpy(np.array(iuv_rep_batch)).float().to(device) / 255

                    pred_cam_wp, pred_pose, pred_shape = regressor(iuv_rep_batch)

                    # Convert pred pose to rotation matrices
                    if pred_pose.shape[-1] == 24 * 3:
                        pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                        pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                    elif pred_pose.shape[-1] == 24 * 6:
                        pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                    pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                            global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                            betas=pred_shape,
                                            pose2rot=False)
                    pred_vertices = pred_smpl_output.vertices
                    pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
                    pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                                    proxy_rep_input_wh)

                    smpl_joints = pred_smpl_output.joints

                    pred_joints_all = pred_smpl_output.joints
                    pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
                    pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
                    pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
                    pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                                    proxy_rep_input_wh)

                    rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                        -1, -1)
                    translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
                    #pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
                    #                                                config.FOCAL_LENGTH, proxy_rep_input_wh)
                    #pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

                    # Need to expand cam_ts for NMR i.e.  from (B, 3)
                    # to
                    # (B, 1, 3)
                    translation_nmr = torch.unsqueeze(translation, dim=1)
                    pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                                faces=faces_batch,
                                                t=translation_nmr,
                                                mode='silhouettes')

                    pred_dict_for_loss = {'joints2D': pred_joints2d_coco,
                                            'verts': pred_vertices,
                                            'shape_params': pred_shape,
                                            'pose_params_rot_matrices': pred_pose_rotmats,
                                            'joints3D': pred_joints_coco}
                    target_dict_for_loss = {'joints2D': torch.from_numpy(np.array(joints2D_batch)[:, :, :2]).float().to(device),
                                            'verts': target_vertices,
                                            'shape_params': betas,
                                            'pose_params_rot_matrices': target_pose_rotmats,
                                            'joints3D': target_joints_coco}

                    # ---------------- BACKWARD PASS ----------------
                    if (is_train):
                        optimiser.zero_grad()
                    loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)
                    if (is_train):
                        loss.backward()
                        optimiser.step()

                    # ---------------- TRACK LOSS AND METRICS
                    # ----------------
                    num_train_inputs_in_batch = len(joints2D_batch)
                    if (is_train):
                        metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                            pred_dict_for_loss, target_dict_for_loss,
                                                            num_train_inputs_in_batch)
                    else:
                        metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                            pred_dict_for_loss, target_dict_for_loss,
                                                            num_train_inputs_in_batch)
                #    break
                #break

            #print(is_train)
            if not is_train:
                # ----------------------- UPDATING LOSS AND METRICS HISTORY
                # -----------------------
                metrics_tracker.update_per_epoch()

                # ----------------------------------- SAVING
                # -----------------------------------
                save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                        best_epoch_val_metrics)

                for metric in save_val_metrics:
                    print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                        metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                    metric,
                                                                    metrics_tracker.history['val_' + metric][-1]))
                if save_model_weights_this_epoch:
                    for metric in save_val_metrics:
                        best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
                    print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                    best_model_wts = copy.deepcopy(regressor.state_dict())
                    best_epoch = epoch
                    print("Best model weights updated!")

                    save_dict = {'epoch': epoch,
                                    'best_epoch': best_epoch,
                                    'best_epoch_val_metrics': best_epoch_val_metrics,
                                    'model_state_dict': regressor.state_dict(),
                                    'best_model_state_dict': best_model_wts,
                                    'optimiser_state_dict': optimiser.state_dict(),
                                    'criterion_state_dict': criterion.state_dict()}
                    model_save_path = os.path.join(player_recon_train_regressor_checkpoints_folder+'iuv', 'best.tar')
                    torch.save(save_dict, model_save_path)

                if epoch % epochs_per_save == 0:
                    # Saving current epoch num, best epoch num, best validation
                    # metrics
                    # (occurred in best
                    # epoch num), current regressor state_dict, best regressor
                    # state_dict, current
                    # optimiser state dict and current criterion state_dict
                    # (i.e.
                    # multi-task loss weights).
                    save_dict = {'epoch': epoch,
                                    'best_epoch': best_epoch,
                                    'best_epoch_val_metrics': best_epoch_val_metrics,
                                    'model_state_dict': regressor.state_dict(),
                                    'best_model_state_dict': best_model_wts,
                                    'optimiser_state_dict': optimiser.state_dict(),
                                    'criterion_state_dict': criterion.state_dict()}
                    model_save_path = os.path.join(player_recon_train_regressor_checkpoints_folder+'iuv', 'model')
                    torch.save(save_dict,
                                model_save_path + '_epoch{}'.format(epoch) + '.tar')
                    print('Model saved! Best Val Metrics:\n',
                            best_epoch_val_metrics,
                            '\nin epoch {}'.format(best_epoch))
                metrics_tracker.initialise_loss_metric_sums()
        endtime = timeit.default_timer()
        print('epoch time: {:.3f}'.format(endtime - starttime))

    print('Training Completed. Best Val Metrics:\n',
          best_epoch_val_metrics)
    print('Best epoch: ', best_epoch)

def evaluate_model_iuv(load_checkpoint=True, folder='Data/STA', 
                   path=player_recon_train_regressor_checkpoints_folder+'iuv',
                   postfix='', test_pose=False, save_vis=True):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if save_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints2D', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa', 'pose_mses', 'shape_mses']
    epochs_per_save = 10

    with open('Data/train_set.xml', 'r') as fs:
        train_set = set(json.load(fs))

    regressor = SingleInputRegressor(resnet_in_channels=3,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # ----------------------- Optimiser -----------------------
    params = list(regressor.parameters()) + list(criterion.parameters())
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    # ----------------------- Resuming -----------------------
    checkpoint_path = os.path.join(path, 'best.tar')
    check_point_loaded = False
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        check_point_loaded = True

        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        #criterion.load_state_dict(checkpoint['criterion_state_dict'])
        print("Regressor loaded. Weights from:", checkpoint_path)

    else:
        checkpoint = torch.load(player_recon_check_points_path, map_location=device)
        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", player_recon_check_points_path)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    if check_point_loaded:
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = \
            load_training_info_from_checkpoint(checkpoint, save_val_metrics)
        load_logs = False
    else:
        current_epoch = 1
        best_epoch_val_metrics = {}
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(regressor.state_dict())
        load_logs = False
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)
    # Starting training loop
    num_epochs = 10 + current_epoch
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    starttime = timeit.default_timer()
        
    print('eval')
    regressor.eval()

    games = os.listdir(player_recon_broad_view_opt_folder+postfix)
    for game in games:
        if game in train_set:
            is_train = True
        else:
            is_train = False

        game_full = os.path.join(player_crop_broad_image_folder+postfix, game)
        game_iuv = os.path.join(player_texture_iuv_folder + 'Broad', game)
        game_dst = os.path.join(player_broad_proxy_folder+postfix, game)
        game_label = os.path.join(player_recon_broad_view_opt_folder+postfix, game)
        if save_vis and not is_train:
            game_vis = os.path.join(folder, game)
            remake_dir(game_vis)
        if not is_train:
            print('process {}'.format(game_full))
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_iuv = os.path.join(game_iuv, scene)
            scene_dst = os.path.join(game_dst, scene)
            scene_label = os.path.join(game_label, scene)
            if save_vis and not is_train:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_full)
            body_pose_batch = []
            global_orient_batch = []
            betas_batch = []
            translation_batch = []
            joints2D_batch = []
            silhouette_batch = []
            proxy_rep_batch = []
            player_vis_batch = []
            image_batch = []
            iuv_rep_batch = []
            for player in players:
                player_full = os.path.join(scene_full, player)
                if (player == '1' or os.path.isfile(player_full)):
                    continue
                #print(player_full)
                player_iuv = os.path.join(scene_iuv, player)
                player_dst = os.path.join(scene_dst, player)
                player_label = os.path.join(scene_label, player)
                if save_vis and not is_train:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)
                    player_vis_batch.append(player_vis)
                        
                view_full = os.path.join(player_full, "player.png")
                iuv_full = os.path.join(player_iuv, "player.png")
                image = cv2.imread(view_full)
                image_batch.append(image)
                j2d_full = os.path.join(player_dst, 'player_j2d.xml')
                sil_full = os.path.join(player_dst, 'player_sil.npy')
                with open(j2d_full, 'r') as fs:
                    joints2D = np.array(json.load(fs))
                silhouette = np.load(sil_full)
                joints2D_batch.append(joints2D[:, :2])
                silhouette_batch.append(silhouette[0])

                label_full = os.path.join(player_label, "data.npz")
                if not os.path.exists(label_full):
                    continue
                target_param = np.load(label_full)
                body_pose = target_param['body_pose']
                global_orient = target_param['global_orient']
                betas = target_param['betas']
                translation = target_param['translation']

                body_pose_batch.append(body_pose[0])
                global_orient_batch.append(global_orient[0])
                betas_batch.append(betas[0])
                translation_batch.append(translation[0])

                proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                in_wh=proxy_rep_input_wh,
                                                out_wh=config.REGRESSOR_IMG_WH)
                proxy_rep_batch.append(proxy_rep)

                iuv_rep = cv2.imread(iuv_full)
                iuv_rep = np.transpose(iuv_rep, [2, 0, 1])
                iuv_rep_batch.append(iuv_rep)

            faces_batch = torch.cat(len(joints2D_batch) * [faces[None, :]], dim=0)
            body_pose = torch.from_numpy(np.array(body_pose_batch)).float().to(device)
            betas = torch.from_numpy(np.array(betas_batch)).float().to(device)
            global_orient = torch.from_numpy(np.array(global_orient_batch)).float().to(device)
            translation = torch.from_numpy(np.array(translation_batch)).float().to(device)
            target_pose_rotmats = torch.cat([global_orient[:, :, :, :],
                        body_pose[:, :, :, :]],
                        dim=1)
            target_smpl_output = smpl(body_pose=body_pose,
                                    global_orient=global_orient,
                                    betas=betas,
                                    pose2rot=False)
            target_vertices = target_smpl_output.vertices
            target_joints_all = target_smpl_output.joints
            target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]

            proxy_rep = torch.from_numpy(np.array(proxy_rep_batch)).float().to(device)
            iuv_rep_batch = torch.from_numpy(np.array(iuv_rep_batch)).float().to(device) / 255

            pred_cam_wp, pred_pose, pred_shape = regressor(iuv_rep_batch)

            # Convert pred pose to rotation matrices
            if pred_pose.shape[-1] == 24 * 3:
                pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
            elif pred_pose.shape[-1] == 24 * 6:
                pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

            if not test_pose:
                pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
            else:
                pred_smpl_output = smpl(body_pose=body_pose,
                                    global_orient=global_orient,
                                    betas=pred_shape,
                                    pose2rot=False)
            pred_vertices = pred_smpl_output.vertices
            pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
            pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                            proxy_rep_input_wh)

            smpl_joints = pred_smpl_output.joints

            pred_joints_all = pred_smpl_output.joints
            pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
            pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
            pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                            proxy_rep_input_wh)

            rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                -1, -1)
            translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
            #                                                config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

            # Need to expand cam_ts for NMR i.e.  from (B, 3)
            # to
            # (B, 1, 3)
            translation_nmr = torch.unsqueeze(translation, dim=1)
            pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                        faces=faces_batch,
                                        t=translation_nmr,
                                        mode='silhouettes')

            if not is_train and save_vis:
                for i in range(len(joints2D_batch)):
                    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[i], 
                                    cam=pred_cam_wp.cpu().detach().numpy()[i], img=image_batch[i])
                    cv2.imwrite(os.path.join(player_vis_batch[i], 'player.png'), best_image)

            pred_dict_for_loss = {'joints2D': pred_joints2d_coco,
                                    'verts': pred_vertices,
                                    'shape_params': pred_shape,
                                    'pose_params_rot_matrices': pred_pose_rotmats,
                                    'joints3D': pred_joints_coco}
            target_dict_for_loss = {'joints2D': torch.from_numpy(np.array(joints2D_batch)[:, :, :2]).float().to(device),
                                    'verts': target_vertices,
                                    'shape_params': betas,
                                    'pose_params_rot_matrices': target_pose_rotmats,
                                    'joints3D': target_joints_coco}

            # ---------------- BACKWARD PASS ----------------
            loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

            # ---------------- TRACK LOSS AND METRICS
            # ----------------
            num_train_inputs_in_batch = len(joints2D_batch)
            if not is_train:
                metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                    pred_dict_for_loss, target_dict_for_loss,
                                                    num_train_inputs_in_batch)
            else:
                metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                    pred_dict_for_loss, target_dict_for_loss,
                                                    num_train_inputs_in_batch)
        #    break
        #break

    #print(is_train)
    # ----------------------- UPDATING LOSS AND METRICS HISTORY
    # -----------------------
    metrics_tracker.update_per_epoch()

    # ----------------------------------- SAVING
    # -----------------------------------
    for metric in save_val_metrics:
        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
            metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                        metric,
                                                        metrics_tracker.history['val_' + metric][-1]))

    endtime = timeit.default_timer()
    print('epoch time: {:.3f}'.format(endtime - starttime))

def evaluate_model_2d_iuv(load_checkpoint=True, folder='Data/real_iuv', 
                   path=player_recon_train_regressor_checkpoints_folder+'iuv',
                   save_vis=True, use_relate = False, opt = True,
                   result_folder = '', hmr=False, spin=False):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if save_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints2D', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa']
    epochs_per_save = 10

    regressor = SingleInputRegressor(resnet_in_channels=3,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    print('eval')
    regressor.eval()

    if use_relate:
        pose_relation = PoseRelationModule()
        pose_relation.to(device)
        pose_relation.eval()

        checkpoint_path = os.path.join(player_recon_train_regressor_checkpoints_folder+'relate_aug1', 'best.tar')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        pose_relation.load_state_dict(checkpoint['best_model_state_dict'])
        print("Pose_relation loaded. Weights from:", checkpoint_path)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # ----------------------- Optimiser -----------------------
    params = list(regressor.parameters()) + list(criterion.parameters())
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    # ----------------------- Resuming -----------------------
    checkpoint_path = os.path.join(path, 'best.tar')
    check_point_loaded = False
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        check_point_loaded = True

        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", checkpoint_path)

    else:
        checkpoint = torch.load(player_recon_check_points_path, map_location=device)
        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", player_recon_check_points_path)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    if check_point_loaded:
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = \
            load_training_info_from_checkpoint(checkpoint, save_val_metrics)
        load_logs = False
    else:
        current_epoch = 1
        best_epoch_val_metrics = {}
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(regressor.state_dict())
        load_logs = False
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)
    # Starting training loop
    num_epochs = 10 + current_epoch
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    starttime = timeit.default_timer()

    games = os.listdir(real_images_player_proxy)
    is_train = False
    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    for game in games:

        game_full = os.path.join(real_images_player, game)
        game_dst = os.path.join(real_images_player_proxy, game)
        game_box = os.path.join(real_images_box, game)
        game_result = os.path.join(result_folder, game)
        game_iuv = os.path.join(player_texture_iuv_folder + 'Real', game)
        if save_vis and not is_train:
            game_vis = os.path.join(folder, game)
            remake_dir(game_vis)
        scenes = os.listdir(game_dst)
        for (ii, scene) in enumerate(scenes):
            print('{} / {}'.format(ii, len(scenes)))
            scene_full = os.path.join(game_full, scene)
            
            scene_dst = os.path.join(game_dst, scene)
            scene_box = os.path.join(game_box, scene)
            scene_iuv = os.path.join(game_iuv, scene)
            scene_result = os.path.join(game_result, scene)
            boxes_full = os.path.join(scene_box, 'boxes.xml')
            with open(boxes_full, 'r') as fs:
                boxes = json.load(fs)
            if save_vis and not is_train:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_dst)
            body_pose_batch = []
            global_orient_batch = []
            betas_batch = []
            translation_batch = []
            joints2D_batch = []
            silhouette_batch = []
            proxy_rep_batch = []
            player_vis_batch = []
            image_batch = []
            boxes_batch = []

            pred_cam_wp_batch = []
            pred_pose_batch = []
            pred_shape_batch = []
            iuv_rep_batch = []
            for player in players:
                player_iuv = os.path.join(scene_iuv, player)
                iuv_full = os.path.join(player_iuv, "player.png")
                if not os.path.exists(iuv_full):
                    continue
                player_full = os.path.join(scene_full, player)
                player_dst = os.path.join(scene_dst, player)
                player_result = os.path.join(scene_result, player)
                
                boxes_batch.append(boxes[int(player)-2])
                if save_vis and not is_train:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)
                    player_vis_batch.append(player_vis)
                        
                view_full = os.path.join(player_full, "player.png")
                
                image = cv2.imread(view_full)
                image_batch.append(image)
                j2d_full = os.path.join(player_dst, 'player_j2d.xml')
                sil_full = os.path.join(player_dst, 'player_sil.npy')
                with open(j2d_full, 'r') as fs:
                    joints2D = np.array(json.load(fs))
                silhouette = np.load(sil_full)
                joints2D_batch.append(joints2D[:, :2])
                #print(silhouette.shape)
                silhouette_batch.append(silhouette)

                proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                in_wh=proxy_rep_input_wh,
                                                out_wh=config.REGRESSOR_IMG_WH)
                proxy_rep_batch.append(proxy_rep)

                #print(iuv_full)
                iuv_rep = cv2.imread(iuv_full)
                #print(iuv_rep.shape)
                iuv_rep = np.transpose(iuv_rep, [2, 0, 1])
                iuv_rep_batch.append(iuv_rep)

                if hmr:
                    dst_full = os.path.join(player_result, "data.npy")
                    pred_param = np.load(dst_full)

                    pred_cam_wp = torch.from_numpy(pred_param[:,:3]).float().to(device)
                    pred_pose = torch.from_numpy(pred_param[:,3:75]).float().to(device)
                    pred_shape = torch.from_numpy(pred_param[:,75:]).float().to(device)

                    # Convert pred pose to rotation matrices
                    if pred_pose.shape[-1] == 24 * 3:
                        pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                        pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                    elif pred_pose.shape[-1] == 24 * 6:
                        pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                    pred_cam_wp_batch.append(pred_cam_wp[0])
                    pred_pose_batch.append(pred_pose_rotmats[0])
                    pred_shape_batch.append(pred_shape[0])
                if spin:
                    dst_full = os.path.join(player_result, "data.npz")

                    init_param = np.load(dst_full)
                    pred_rotmat = init_param['pred_rotmat']
                    pred_betas = init_param['pred_betas']
                    pred_camera = init_param['pred_camera']

                    pred_pose_rotmats = torch.from_numpy(pred_rotmat).float().to(device)
                    pred_shape = torch.from_numpy(pred_betas).float().to(device)
                    pred_cam_wp = torch.from_numpy(pred_camera).float().to(device)

                    pred_cam_wp_batch.append(pred_cam_wp)
                    pred_pose_batch.append(pred_pose_rotmats)
                    pred_shape_batch.append(pred_shape)

            faces_batch = torch.cat(len(joints2D_batch) * [faces[None, :]], dim=0)
            boxes_batch = torch.from_numpy(np.array(boxes_batch)).float().to(device)

            proxy_rep = torch.from_numpy(np.array(proxy_rep_batch)).float().to(device)
            iuv_rep_batch = torch.from_numpy(np.array(iuv_rep_batch)).float().to(device) / 255

            if hmr or spin:
                pred_cam_wp = torch.stack(pred_cam_wp_batch)
                pred_pose_rotmats = torch.stack(pred_pose_batch)
                pred_shape = torch.stack(pred_shape_batch)
            else:
                with torch.no_grad():
                    pred_cam_wp, pred_pose, pred_shape = regressor(iuv_rep_batch)

                # Convert pred pose to rotation matrices
                if pred_pose.shape[-1] == 24 * 3:
                    pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                    pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                elif pred_pose.shape[-1] == 24 * 6:
                    pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

            if use_relate:
                pred_pose_rotmats = pose_relation([pred_pose_rotmats, boxes_batch])

            if opt:
                pred_pose_rotmats = pred_pose_rotmats.detach()
                pred_shape = pred_shape.detach()
                pred_cam_wp = pred_cam_wp.detach()
                for i in range(len(joints2D_batch)):
                    optimize_camera(pred_pose_rotmats[i, 1:].unsqueeze(0), pred_pose_rotmats[i, 0].unsqueeze(0).unsqueeze(1), pred_shape[i].unsqueeze(0),
                                    joints2D_batch[i], silhouette_batch[i], pred_cam_wp[i].unsqueeze(0))

            pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
            pred_vertices = pred_smpl_output.vertices
            pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
            pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                            proxy_rep_input_wh)

            smpl_joints = pred_smpl_output.joints

            pred_joints_all = pred_smpl_output.joints
            pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
            pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
            pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                            proxy_rep_input_wh)

            rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                -1, -1)
            translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
            #                                                config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

            # Need to expand cam_ts for NMR i.e.  from (B, 3)
            # to
            # (B, 1, 3)
            translation_nmr = torch.unsqueeze(translation, dim=1)
            pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                        faces=faces_batch,
                                        t=translation_nmr,
                                        mode='silhouettes')

            for i in range(len(joints2D_batch)):
                keypoints = pred_joints2d_coco.cpu().detach().numpy()[i].astype('int32')
                silh_error, iou_vis = compute_silh_error_metrics(
                    pred_silhouettes.cpu().detach().numpy()[i], silhouette_batch[i], False)
                joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D_batch[i])
                        
                silh_mean_error_init += silh_error['iou']
                joint_mean_error_init += joints_error

                num_counter += 1

                #if i == 0:
                #    cv2.imwrite('Data/1.png', silhouette_batch[i] * 128)

            if not is_train and save_vis:
                for i in range(len(joints2D_batch)):
                    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[i], 
                                    cam=pred_cam_wp.cpu().detach().numpy()[i], img=image_batch[i])
                    cv2.imwrite(os.path.join(player_vis_batch[i], 'player.png'), best_image)

        #    break
        #break

    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
    endtime = timeit.default_timer()
    print('epoch time: {:.3f}'.format(endtime - starttime))

def evaluate_model_2d_s_p(load_checkpoint=True, folder='Data/real_iuvp', 
                   path=player_recon_train_regressor_checkpoints_folder+'sp',
                   save_vis=False, use_relate = False, opt = True, use_s=True,use_p=True,
                   result_folder = '', hmr=False, spin=False):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if save_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    losses_on = ['verts', 'shape_params', 'pose_params', 'joints2D', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
    save_val_metrics = ['pves', 'pves_pa', 'mpjpes', 'mpjpes_pa']
    epochs_per_save = 10

    num_channels = 0
    if (use_s):
        num_channels += 1
    if (use_p):
        num_channels += 17
    regressor = SingleInputRegressor(resnet_in_channels=num_channels,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    print('eval')
    regressor.eval()

    if use_relate:
        pose_relation = PoseRelationModule()
        pose_relation.to(device)
        pose_relation.eval()

        checkpoint_path = os.path.join(player_recon_train_regressor_checkpoints_folder+'iuvp_relate', 'best.tar')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        pose_relation.load_state_dict(checkpoint['best_model_state_dict'])
        print("Pose_relation loaded. Weights from:", checkpoint_path)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    cam_K = get_intrinsics_matrix(proxy_rep_input_wh, proxy_rep_input_wh, config.FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(1, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(1, -1, -1)

    nmr = nr.Renderer(camera_mode='projection',
                            K=cam_K,
                            R=cam_R,
                            image_size=proxy_rep_input_wh,
                            orig_size=proxy_rep_input_wh)
    nmr = nmr.to(device)

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    # ----------------------- Loss -----------------------
    criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                              init_loss_weights=init_loss_weights,
                                                              reduction='mean')
    criterion.to(device)

    # ----------------------- Optimiser -----------------------
    params = list(regressor.parameters()) + list(criterion.parameters())
    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

    # ----------------------- Resuming -----------------------
    checkpoint_path = os.path.join(path, 'best.tar')
    check_point_loaded = False
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        check_point_loaded = True

        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", checkpoint_path)

    else:
        checkpoint = torch.load(player_recon_check_points_path, map_location=device)
        regressor.load_state_dict(checkpoint['best_model_state_dict'])
        print("Regressor loaded. Weights from:", player_recon_check_points_path)

    # Ensure that all metrics used as model save conditions are being tracked
    # (i.e.  that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    if check_point_loaded:
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = \
            load_training_info_from_checkpoint(checkpoint, save_val_metrics)
        load_logs = False
    else:
        current_epoch = 1
        best_epoch_val_metrics = {}
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(regressor.state_dict())
        load_logs = False
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf

    # Instantiate metrics tracker.
    log_path = os.path.join(player_recon_train_regressor_logs_folder, 'logs.pkl')
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=config.REGRESSOR_IMG_WH,
                                                      log_path=log_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)
    # Starting training loop
    num_epochs = 10 + current_epoch
    #num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    starttime = timeit.default_timer()

    games = os.listdir(real_images_player_proxy)
    is_train = False
    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    for game in games:

        game_full = os.path.join(real_images_player, game)
        game_dst = os.path.join(real_images_player_proxy+'unrefine', game)
        game_dst_refine = os.path.join(real_images_player_proxy, game)
        game_box = os.path.join(real_images_box, game)
        game_result = os.path.join(result_folder, game)
        game_iuv = os.path.join(player_texture_iuv_folder + 'Real', game)
        if save_vis and not is_train:
            game_vis = os.path.join(folder, game)
            remake_dir(game_vis)
        scenes = os.listdir(game_dst_refine)
        for (ii, scene) in enumerate(scenes):
            print('{} / {}'.format(ii, len(scenes)))
            starttime_scene = timeit.default_timer()
            scene_full = os.path.join(game_full, scene)
            
            scene_dst = os.path.join(game_dst, scene)
            scene_dst_refine = os.path.join(game_dst_refine, scene)
            scene_box = os.path.join(game_box, scene)
            scene_iuv = os.path.join(game_iuv, scene)
            scene_result = os.path.join(game_result, scene)
            boxes_full = os.path.join(scene_box, 'boxes.xml')
            with open(boxes_full, 'r') as fs:
                boxes = json.load(fs)
            if save_vis and not is_train:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_dst_refine)
            body_pose_batch = []
            global_orient_batch = []
            betas_batch = []
            translation_batch = []
            joints2D_batch = []
            silhouette_batch = []
            proxy_rep_batch = []
            player_vis_batch = []
            image_batch = []
            boxes_batch = []

            pred_cam_wp_batch = []
            pred_pose_batch = []
            pred_shape_batch = []
            iuv_rep_batch = []
            for player in players:
                player_iuv = os.path.join(scene_iuv, player)
                iuv_full = os.path.join(player_iuv, "player.png")
                if not os.path.exists(iuv_full):
                    continue
                player_full = os.path.join(scene_full, player)
                player_dst = os.path.join(scene_dst, player)
                player_dst_refine = os.path.join(scene_dst_refine, player)
                player_result = os.path.join(scene_result, player)
                
                boxes_batch.append(boxes[int(player)-2])
                if save_vis and not is_train:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)
                    player_vis_batch.append(player_vis)
                        
                view_full = os.path.join(player_full, "player.png")
                
                image = cv2.imread(view_full)
                image_batch.append(image)
                j2d_full = os.path.join(player_dst, 'player_j2d.xml')
                sil_full = os.path.join(player_dst, 'player_sil.npy')
                j2d_full_refine = os.path.join(player_dst_refine, 'player_j2d.xml')
                with open(j2d_full, 'r') as fs:
                    joints2D = np.array(json.load(fs))
                with open(j2d_full_refine, 'r') as fs:
                    joints2D_refine = np.array(json.load(fs))
                silhouette = np.load(sil_full)
                joints2D_batch.append(joints2D_refine[:, :2])
                #print(silhouette.shape)
                silhouette_batch.append(silhouette)

                proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                in_wh=proxy_rep_input_wh,
                                                out_wh=config.REGRESSOR_IMG_WH)
                proxy_rep_batch.append(proxy_rep)

                #print(iuv_full)
                iuv_rep = cv2.imread(iuv_full)
                iuv_rep = cv2.resize(iuv_rep, (config.REGRESSOR_IMG_WH, config.REGRESSOR_IMG_WH),interpolation=cv2.INTER_LINEAR)
                #print(iuv_rep.shape)
                iuv_rep = np.transpose(iuv_rep, [2, 0, 1])
                iuv_rep_batch.append(iuv_rep)

                if hmr:
                    dst_full = os.path.join(player_result, "data.npy")
                    pred_param = np.load(dst_full)

                    pred_cam_wp = torch.from_numpy(pred_param[:,:3]).float().to(device)
                    pred_pose = torch.from_numpy(pred_param[:,3:75]).float().to(device)
                    pred_shape = torch.from_numpy(pred_param[:,75:]).float().to(device)

                    # Convert pred pose to rotation matrices
                    if pred_pose.shape[-1] == 24 * 3:
                        pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                        pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                    elif pred_pose.shape[-1] == 24 * 6:
                        pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                    pred_cam_wp_batch.append(pred_cam_wp[0])
                    pred_pose_batch.append(pred_pose_rotmats[0])
                    pred_shape_batch.append(pred_shape[0])
                if spin:
                    dst_full = os.path.join(player_result, "data.npz")

                    init_param = np.load(dst_full)
                    pred_rotmat = init_param['pred_rotmat']
                    pred_betas = init_param['pred_betas']
                    pred_camera = init_param['pred_camera']

                    pred_pose_rotmats = torch.from_numpy(pred_rotmat).float().to(device)
                    pred_shape = torch.from_numpy(pred_betas).float().to(device)
                    pred_cam_wp = torch.from_numpy(pred_camera).float().to(device)

                    pred_cam_wp_batch.append(pred_cam_wp)
                    pred_pose_batch.append(pred_pose_rotmats)
                    pred_shape_batch.append(pred_shape)

            faces_batch = torch.cat(len(joints2D_batch) * [faces[None, :]], dim=0)
            boxes_batch = torch.from_numpy(np.array(boxes_batch)).float().to(device)

            proxy_rep = torch.from_numpy(np.array(proxy_rep_batch)).float().to(device)
            iuv_rep_batch = torch.from_numpy(np.array(iuv_rep_batch)).float().to(device) / 255
            iuv_rep_batch = torch.cat((proxy_rep[:,1:,:,:], iuv_rep_batch), 1)

            if (use_s):
                if (use_p):
                    iuv_rep_batch = proxy_rep
                else:
                    iuv_rep_batch = proxy_rep[:,:1,:,:]
            else:
                iuv_rep_batch = proxy_rep[:,1:,:,:]

            if hmr or spin:
                pred_cam_wp = torch.stack(pred_cam_wp_batch)
                pred_pose_rotmats = torch.stack(pred_pose_batch)
                pred_shape = torch.stack(pred_shape_batch)
            else:
                with torch.no_grad():
                    pred_cam_wp, pred_pose, pred_shape = regressor(iuv_rep_batch)

                # Convert pred pose to rotation matrices
                if pred_pose.shape[-1] == 24 * 3:
                    pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                    pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                elif pred_pose.shape[-1] == 24 * 6:
                    pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

            if use_relate:
                pred_pose_rotmats = pose_relation([pred_pose_rotmats, boxes_batch])

            if opt:
                pred_pose_rotmats = pred_pose_rotmats.detach()
                pred_shape = pred_shape.detach()
                pred_cam_wp = pred_cam_wp.detach()
                for i in range(len(joints2D_batch)):
                    optimize_camera(pred_pose_rotmats[i, 1:].unsqueeze(0), pred_pose_rotmats[i, 0].unsqueeze(0).unsqueeze(1), pred_shape[i].unsqueeze(0),
                                    joints2D_batch[i], silhouette_batch[i], pred_cam_wp[i].unsqueeze(0))

            pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
            pred_vertices = pred_smpl_output.vertices
            pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
            pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                            proxy_rep_input_wh)

            smpl_joints = pred_smpl_output.joints

            pred_joints_all = pred_smpl_output.joints
            pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
            pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
            pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                            proxy_rep_input_wh)

            rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                -1, -1)
            translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
            #                                                config.FOCAL_LENGTH, proxy_rep_input_wh)
            #pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

            # Need to expand cam_ts for NMR i.e.  from (B, 3)
            # to
            # (B, 1, 3)
            translation_nmr = torch.unsqueeze(translation, dim=1)
            pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                        faces=faces_batch,
                                        t=translation_nmr,
                                        mode='silhouettes')

            for i in range(len(joints2D_batch)):
                keypoints = pred_joints2d_coco.cpu().detach().numpy()[i].astype('int32')
                silh_error, iou_vis = compute_silh_error_metrics(
                    pred_silhouettes.cpu().detach().numpy()[i], silhouette_batch[i], False)
                joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D_batch[i])
                        
                silh_mean_error_init += silh_error['iou']
                joint_mean_error_init += joints_error

                num_counter += 1

                #if i == 0:
                #    cv2.imwrite('Data/1.png', silhouette_batch[i] * 128)

            if not is_train and save_vis:
                for i in range(len(joints2D_batch)):
                    best_image = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[i], 
                                    cam=pred_cam_wp.cpu().detach().numpy()[i], img=image_batch[i])
                    cv2.imwrite(os.path.join(player_vis_batch[i], 'player.png'), best_image)
            endtime_scene = timeit.default_timer()
            print('scene time: {:.3f}'.format(endtime_scene - starttime_scene))

        #    break
        #break

    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
    endtime = timeit.default_timer()
    print('epoch time: {:.3f}'.format(endtime - starttime))

#train_regressor(False)
#evaluate_model(False)
#evaluate_model(True, 'Data/baseline',path=player_recon_train_regressor_checkpoints_folder)
#evaluate_model(True, 'Data/aug',path=player_recon_train_regressor_checkpoints_folder+'aug')
#evaluate_model_relate(True)
#evaluate_model_relate(True, 'Data/aug_relate', path=player_recon_train_regressor_checkpoints_folder+'aug',
#                      path1=player_recon_train_regressor_checkpoints_folder+'relate_aug1', opt=False)
#evaluate_model_relate(True, 'Data/aug_relate_opt', path=player_recon_train_regressor_checkpoints_folder+'aug',
#                      path1=player_recon_train_regressor_checkpoints_folder+'relate_aug1', opt=True)
#evaluate_model_hmr()
#smpl_vis()
#evaluate_model_2d()
#evaluate_model_2d(True, 'Data/real_ours_opt_global', player_recon_train_regressor_checkpoints_folder+'aug', use_relate=True, opt = True)
#evaluate_model_2d(False, 'Data/real_STR_opt', player_recon_train_regressor_checkpoints_folder+'aug', use_relate=False, opt = True)
#spin_vis()
#evaluate_model_spin()
#evaluate_model_2d(True, 'Data/real_hmr_opt_global', player_recon_train_regressor_checkpoints_folder+'aug', 
#                  use_relate=False, opt = True, result_folder='Data/RealHmr', hmr=True)
#evaluate_model_2d(True, 'Data/real_spin_opt', player_recon_train_regressor_checkpoints_folder+'aug', 
#                  use_relate=False, opt = True, result_folder='Data/RealPlayerImageSPIN', spin=True)

#train_regressor_iuv(True)
#evaluate_model_iuv(True, 'Data/iuv', path=player_recon_train_regressor_checkpoints_folder+'iuv')
#evaluate_model_2d_iuv()

#evaluate_model(False, 'Data/STR_unrefine', unrefine=True)
#evaluate_model_2d(folder='Data/real_STR_unrefine', unrefine=True)

#evaluate_model_hmr(folder='Data/hmrBlur', path='Data/PlayerBroadImageBlurHmr')
#evaluate_model_spin(folder='Data/spinBlur', path='Data/PlayerBroadImageBlurSpin')
#evaluate_model_2d_iuv_p()
#evaluate_model_2d_iuv_p(use_relate=True)
#evaluate_model_relate_iuv()

#evaluate_model_2d_s_p(path=player_recon_train_regressor_checkpoints_folder+'sp', use_s=True, use_p=True)
#evaluate_model_2d_s_p(path=player_recon_train_regressor_checkpoints_folder+'s', use_s=True, use_p=False)
#evaluate_model_2d_s_p(path=player_recon_train_regressor_checkpoints_folder+'p', use_s=False, use_p=True)

#evaluate_model_2d_iuv_p(blur = '_21_21', save_vis=False)
#evaluate_model_2d_iuv_p(blur = '_0_0', save_vis=False)
#evaluate_model_2d_iuv_p(blur = '_11_11', save_vis=False)
#evaluate_model_2d_iuv_p(blur = '_0_11', save_vis=False)
#evaluate_model_2d_iuv_p(blur = '_11_0', save_vis=False)