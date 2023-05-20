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
    convert_camera_translation_to_weak_perspective, convert_camera_translation_to_weak_perspective_torch
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

def init_loss_and_metric(joints=True, silhoutte=True):
    losses_on = []
    metrics_to_track = []
    save_val_metrics = []
    if (joints):
        losses_on.append('joints2D')
        metrics_to_track.append('joints2D_l2es')
        save_val_metrics.append('joints2D_l2es')
    if (silhoutte):
        losses_on.append('silhouette')
    save_val_metrics.append('silhouette_iou')
    metrics_to_track.append('silhouette_iou')
    init_loss_weights = {'joints2D': 1.0, 'silhouette': 1000000.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    #save_val_metrics = metrics_to_track
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

def evaluate_model_2d(image_folder='Data/PlayerCrop', data_folder='Data/PlayerCrop_hmr', 
                   proxy_folder='Data/PlayerProxy', vis_folder='Data/PlayerCrop_hmr_vis',
                   result_folder='Data/PlayerCrop_hmr_opt',
                   save_vis=True, opt = True, hmr=True, spin=False, pare=False):

    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    if save_vis:
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder, exist_ok=True)

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

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    starttime = timeit.default_timer()

    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    silh_mean_error_opt = 0
    joint_mean_error_opt = 0
    games = os.listdir(image_folder)
    for game in games:
        game_full = os.path.join(image_folder, game)
        game_dst = os.path.join(result_folder, game)
        remake_dir(game_dst)
        game_data = os.path.join(data_folder, game)
        game_proxy = os.path.join(proxy_folder, game)
        if save_vis:
            game_vis = os.path.join(vis_folder, game)
            remake_dir(game_vis)
        scenes = os.listdir(game_full)
        for scene in scenes:
            print(scene)
            scene_full = os.path.join(game_full, scene)
            scene_dst = os.path.join(game_dst, scene)
            remake_dir(scene_dst)
            scene_data = os.path.join(game_data, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            if save_vis:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_full)
            for player in players:
                player_full = os.path.join(scene_full, player)
                player_dst = os.path.join(scene_dst, player)
                remake_dir(player_dst)
                player_data = os.path.join(scene_data, player)
                player_proxy = os.path.join(scene_proxy, player)
                if save_vis:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)

                views = os.listdir(player_full)
                for view in views:
                    view_full = os.path.join(player_full, view)
                    if hmr:
                        view_dst = os.path.join(player_dst, view.replace('.png', '.npz'))
                        view_data = os.path.join(player_data, view.replace('.png', '.npy'))
                    elif spin:
                        view_dst = os.path.join(player_dst, view.replace('.png', '.npz'))
                        view_data = os.path.join(player_data, view.replace('.png', '.npz'))
                    elif pare:
                        view_dst = os.path.join(player_dst, view.replace('.png', '.npz'))
                        view_data = os.path.join(player_data, view.replace('.png', '.npz'))
                    j2d_full = os.path.join(player_proxy, view).replace('.png', '_j2d.xml')
                    sil_full = os.path.join(player_proxy, view).replace('.png', '_sil.npy')
                    if save_vis:
                        view_vis = os.path.join(player_vis, view)

                    image = cv2.imread(view_full)

                    with open(j2d_full, 'r') as fs:
                        joints2D = np.array(json.load(fs))
                    silhouette = np.load(sil_full)

                    if not os.path.exists(view_data):
                        continue

                    if hmr:
                        pred_param = np.load(view_data)

                        pred_cam_wp = torch.from_numpy(pred_param[:,:3]).float().to(device)
                        pred_pose = torch.from_numpy(pred_param[:,3:75]).float().to(device)
                        pred_shape = torch.from_numpy(pred_param[:,75:]).float().to(device)

                        # Convert pred pose to rotation matrices
                        if pred_pose.shape[-1] == 24 * 3:
                            pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                            pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                        elif pred_pose.shape[-1] == 24 * 6:
                            pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)
                    if spin:
                        init_param = np.load(view_data)
                        pred_rotmat = init_param['pred_rotmat']
                        pred_betas = init_param['pred_betas']
                        pred_camera = init_param['pred_camera']

                        pred_pose_rotmats = torch.from_numpy(pred_rotmat).float().to(device).unsqueeze(0)
                        pred_shape = torch.from_numpy(pred_betas).float().to(device).unsqueeze(0)
                        pred_cam_wp = torch.from_numpy(pred_camera).float().to(device).unsqueeze(0)
                    if pare:
                        init_param = np.load(view_data)
                        pred_rotmat = init_param['pred_rotmat']
                        pred_betas = init_param['pred_betas']
                        pred_camera = init_param['pred_camera']

                        pred_pose_rotmats = torch.from_numpy(pred_rotmat).float().to(device)[:1]
                        pred_shape = torch.from_numpy(pred_betas).float().to(device)[:1]
                        pred_cam_wp = torch.from_numpy(pred_camera).float().to(device)[:1, 1:]

                        #print(pred_cam_wp)
                        #print(pred_pose_rotmats)
                        #print(pred_shape)

                    body_pose = pred_pose_rotmats[:, 1:]
                    global_orient = pred_pose_rotmats[:, 0].unsqueeze(1)
                    betas = pred_shape
                    translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

                    if opt:
                        criterion, metrics_tracker, save_val_metrics = init_loss_and_metric(True, False)
                        best_epoch_val_metrics = {}
                        best_smpl = None
                        best_epoch = 1
                        best_model_wts = copy.deepcopy([body_pose.cpu().detach().numpy(), 
                                                        global_orient.cpu().detach().numpy(), 
                                                        betas.cpu().detach().numpy(), 
                                                        translation.cpu().detach().numpy()])
                        for metric in save_val_metrics:
                            best_epoch_val_metrics[metric] = np.inf

                        global_orient.requires_grad = True
                        pred_cam_wp.requires_grad = True
                        params = [global_orient, pred_cam_wp]
                        optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)
                        
                        for epoch in range(1, 50 + 1):
                            metrics_tracker.initialise_loss_metric_sums()
                            pred_smpl_output = smpl(body_pose=body_pose,
                                                    global_orient=global_orient,
                                                    betas=betas,
                                                    pose2rot=False)

                            if (epoch == 1 and save_vis):
                                rend_img = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                              cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                                cv2.imwrite(view_vis.replace('.png', '_0.png'), rend_img)

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
                        
                            keypoints = pred_joints2d_coco.cpu().detach().numpy()[0].astype('int32')
                            silh_error, iou_vis = compute_silh_error_metrics(pred_silhouettes.cpu().detach().numpy()[0], silhouette, False)
                            joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D[:,:2])
                        
                            if epoch == 1:
                                silh_mean_error_init += silh_error['iou']
                                joint_mean_error_init += joints_error

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

                            save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                            best_epoch_val_metrics)
                            if save_model_weights_this_epoch:
                                for metric in save_val_metrics:
                                    best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
                                best_model_wts = copy.deepcopy([body_pose.cpu().detach().numpy(), 
                                                        global_orient.cpu().detach().numpy(), 
                                                        betas.cpu().detach().numpy(), 
                                                        translation.cpu().detach().numpy()])
                                best_epoch = epoch
                                best_smpl = pred_smpl_output
                                best_silh_error = silh_error['iou']
                                best_joints_error = joints_error

                            if epoch == 1:
                                for metric in save_val_metrics:
                                    print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                        metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                                    metric,
                                                                                    metrics_tracker.history['val_' + metric][-1]))

                            loss.backward()
                            optimiser.step()
                            translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

                        print('Finished translation optimization.')
                        print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                        print('Best epoch ', best_epoch)
                    
                        if save_vis:
                            best_image = wp_renderer.render(verts=best_smpl.vertices.cpu().detach().numpy()[0], 
                                                        cam=convert_camera_translation_to_weak_perspective(best_model_wts[3][0], config.FOCAL_LENGTH, proxy_rep_input_wh), img=image)
                            cv2.imwrite(view_vis.replace('.png', '_1.png'), best_image)

                        np.savez(view_dst, body_pose=best_model_wts[0], global_orient=best_model_wts[1],
                                 betas=best_model_wts[2], translation=best_model_wts[3])

                        silh_mean_error_opt += best_silh_error
                        joint_mean_error_opt += best_joints_error
                        num_counter += 1
        #        break
        #    break
        #break
    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
        print('silh_iou_opt: {}, joint_error_opt: {}'.format(silh_mean_error_opt / num_counter, joint_mean_error_opt / num_counter))

def multi_view_optimization(save_vis=True, is_refine=False, score_thresh=10.0, game_continue='',
                            image_folder=player_crop_data_folder, proxy_folder=player_recon_proxy_folder,
                            result_folder=player_recon_multi_view_opt_folder, vis_folder=player_recon_multi_view_opt_result_folder,
                            single_folder=player_recon_single_view_opt_folder, ignore_first=True,
                            interation=player_recon_multi_view_iteration):
    print('multi view opt')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder, exist_ok=True)

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

    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)
    faces = torch.cat(1 * [faces[None, :]], dim=0)

    # Starting training loop
    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    silh_mean_error_opt = 0
    joint_mean_error_opt = 0
    games = os.listdir(image_folder)
    for game in games:
        starttime = timeit.default_timer()
        game_full = os.path.join(image_folder, game)
        game_proxy = os.path.join(proxy_folder, game)
        game_init = os.path.join(single_folder, game)
        game_opt = os.path.join(result_folder, game)
        game_opt_result = os.path.join(vis_folder, game)
        if os.path.exists(game_opt) and game != game_continue and not is_refine:
            continue
        if not is_refine:
            remake_dir(game_opt)
            remake_dir(game_opt_result)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            scene_init = os.path.join(game_init, scene)
            scene_opt = os.path.join(game_opt, scene)
            scene_opt_result = os.path.join(game_opt_result, scene)
            if not is_refine:
                remake_dir(scene_opt)
                remake_dir(scene_opt_result)
            players = os.listdir(scene_full)
            for player in players:
                starttime_player = timeit.default_timer()
                player_full = os.path.join(scene_full, player)
                if (os.path.isfile(player_full)):
                    continue
                if (ignore_first and player == '1'):
                    continue
                player_proxy = os.path.join(scene_proxy, player)
                player_init = os.path.join(scene_init, player)
                player_opt = os.path.join(scene_opt, player)
                player_opt_result = os.path.join(scene_opt_result, player)
                if not is_refine:
                    remake_dir(player_opt)
                    remake_dir(player_opt_result)
                if is_refine:
                    with open(os.path.join(player_opt_result, 'metrics.xml'), 'r') as fs:
                        before = json.load(fs)
                    if before[1] < score_thresh:
                        continue
                    print('Before ', before[1])
                print('process {}'.format(player_full))

                joints2D_mult = []
                silhouette_mult = []
                body_pose_mult = []
                global_orient_mult = []
                betas_mult = []
                translation_mult = []
                view_mult = []
                image_mult = []
                cam_wp_mult = []
                
                opt_full = os.path.join(player_opt, 'data.npz')
                views = os.listdir(player_full)
                for view in views:
                    view_full = os.path.join(player_full, view)
                    
                    image = cv2.imread(view_full)
                    j2d_full = os.path.join(player_proxy, view).replace('.png', '_j2d.xml')
                    sil_full = os.path.join(player_proxy, view).replace('.png', '_sil.npy')
                    with open(j2d_full, 'r') as fs:
                        joints2D = np.array(json.load(fs))
                    silhouette = np.load(sil_full)

                    init_full = os.path.join(player_init, view).replace('.png', '.npz')
                    if not os.path.exists(init_full):
                        continue
                    init_param = np.load(init_full)
                    body_pose = init_param['body_pose']
                    global_orient = init_param['global_orient']
                    betas = init_param['betas']
                    translation = init_param['translation']

                    joints2D_mult.append(joints2D)
                    silhouette_mult.append(silhouette)
                    body_pose_mult.append(torch.from_numpy(body_pose).float().to(device))
                    global_orient_mult.append(torch.from_numpy(global_orient).float().to(device))
                    betas_mult.append(torch.from_numpy(betas).float().to(device))
                    translation = torch.from_numpy(translation).float().to(device)
                    translation_mult.append(translation)
                    view_mult.append(view)
                    image_mult.append(image)
                    cam_wp_mult.append(convert_camera_translation_to_weak_perspective_torch(translation, config.FOCAL_LENGTH, proxy_rep_input_wh))

                body_pose_mean = torch.stack(body_pose_mult)
                betas_mean = torch.stack(betas_mult)
                body_pose_mean = body_pose_mean.mean(axis = 0)
                betas_mean = betas_mean.mean(axis = 0)

                #body_pose_mean += (torch.rand_like(body_pose_mean).float().to(device) - 0.5) * 0.1
                #betas_mean += ((torch.rand_like(betas_mean).float().to(device) - 0.5) * 2.5) * 0.1

                pred_smpl_output_mult = [None] * len(joints2D_mult)
                global_orient_mult = torch.stack(global_orient_mult)
                #translation_mult = torch.stack(translation_mult)
                cam_wp_mult = torch.stack(cam_wp_mult)

                criterion, metrics_tracker, save_val_metrics = init_loss_and_metric(True, False)

                best_pose = torch.cat([body_pose_mean[:, :6, :, :],
                                        body_pose_mean[:, 8:21, :, :]],
                                        dim=1)
                best_epoch_val_metrics = {}
                best_epoch = 1
                best_model_wts = copy.deepcopy([body_pose_mean.cpu().detach().numpy(), 
                                                betas_mean.cpu().detach().numpy()])
                best_model_view = []
                for j in range(len(view_mult)):
                    best_model_view.append(copy.deepcopy([translation_mult[j].cpu().detach().numpy(), 
                                            global_orient_mult[j].cpu().detach().numpy()]))
                for metric in save_val_metrics:
                    best_epoch_val_metrics[metric] = np.inf
                current_epoch = 1
                for i in range(3):

                    # ----------------------- Optimiser -----------------------
                    body_poses_without_hands_feet = best_pose
                    body_poses_hands_feet = torch.cat([body_pose_mean[:, 6:8, :, :],
                                                            body_pose_mean[:, 21:, :, :]],
                                                            dim=1)

                    body_poses_with_hand_feet = torch.cat([body_poses_without_hands_feet[:, :6, :, :],
                                            body_poses_hands_feet[:, 0:2, :, :],
                                            body_poses_without_hands_feet[:, 6:19, :, :],
                                            body_poses_hands_feet[:, 2:, :, :]],
                                            dim=1)

                    # optimize camera
                    betas_mean.requires_grad = False
                    body_poses_without_hands_feet.requires_grad = False
                    global_orient_mult.requires_grad = True
                    cam_wp_mult.requires_grad = True
                    params = [cam_wp_mult, global_orient_mult]
                    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

                    best_global_orient_mult = global_orient_mult.clone().detach()
                    best_cam_wp_mult = cam_wp_mult.clone().detach()

                    earlier_stop = False
                    for epoch in range(current_epoch, interation + current_epoch):
                        silh_error_mult = 0
                        joints_error_mult = 0
                        metrics_tracker.initialise_loss_metric_sums()
                        for is_train in [True, False]:
                            indexes = list(range(len(joints2D_mult)))
                            random.shuffle(indexes)
                            for index in indexes:
                                pred_smpl_output = smpl(body_pose=body_poses_with_hand_feet,
                                                        global_orient=global_orient_mult[index],
                                                        betas=betas_mean,
                                                        pose2rot=False)
                                pred_smpl_output_mult[index] = pred_smpl_output

                                if (epoch == 1 and i == 0 and save_vis):
                                    rend_img = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                                    cam=cam_wp_mult[index].cpu().detach().numpy()[0], img=image_mult[index])
                                    opt_full_result = os.path.join(player_opt_result, view_mult[index])
                                    cv2.imwrite(opt_full_result.replace('.png', '_{}_0.png'.format(i)), rend_img)

                                pred_joints_all = pred_smpl_output.joints
                                pred_joints2d_coco = orthographic_project_torch(pred_joints_all, cam_wp_mult[index])
                                pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
                                pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                                                proxy_rep_input_wh)

                                # Need to expand cam_ts for NMR i.e.  from (B, 3)
                                # to
                                # (B, 1, 3)
                                translation_nmr = torch.unsqueeze(translation_mult[index], dim=1)
                                pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                                            faces=faces,
                                                            t=translation_nmr,
                                                            mode='silhouettes')

                                if not is_train:
                                    keypoints = pred_joints2d_coco.cpu().detach().numpy()[0].astype('int32')
                                    silh_error, iou_vis = compute_silh_error_metrics(pred_silhouettes.cpu().detach().numpy()[0], silhouette_mult[index], False)
                                    joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D_mult[index][:,:2])

                                    silh_error_mult += silh_error['iou']
                                    joints_error_mult += joints_error

                                if epoch == 1 and i == 0 and not is_train:
                                    silh_mean_error_init += silh_error['iou']
                                    joint_mean_error_init += joints_error
                                    num_counter += 1

                                pred_dict_for_loss = {'joints2D': pred_joints2d_coco[0],
                                                        'silhouette': pred_silhouettes}
                                target_dict_for_loss = {'joints2D': torch.from_numpy(joints2D_mult[index][:,:2]).float().to(device),
                                                        'silhouette': torch.from_numpy(silhouette_mult[index]).float().to(device).unsqueeze(0)}

                                # ---------------- BACKWARD PASS ----------------
                                if is_train:
                                    optimiser.zero_grad()
                                loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

                                # ---------------- TRACK LOSS AND METRICS
                                # ----------------
                                num_train_inputs_in_batch = 1
                                if is_train:
                                    metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                                        pred_dict_for_loss, target_dict_for_loss,
                                                                        num_train_inputs_in_batch)
                                else:
                                    metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                                    pred_dict_for_loss, target_dict_for_loss,
                                                                    num_train_inputs_in_batch)

                                if is_train:
                                    loss.backward()
                                    optimiser.step()
                                    translation_mult[index] = convert_weak_perspective_to_camera_translation_torch(cam_wp_mult[index], config.FOCAL_LENGTH, proxy_rep_input_wh)

                            if not is_train:
                                metrics_tracker.update_per_epoch()

                                save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                                    best_epoch_val_metrics)
                                if save_model_weights_this_epoch:
                                    for metric in save_val_metrics:
                                        best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
                                    best_model_wts = copy.deepcopy([body_poses_with_hand_feet.cpu().detach().numpy(), 
                                                            betas_mean.cpu().detach().numpy()])
                                    for j in range(len(view_mult)):
                                        best_model_view[j] = copy.deepcopy([translation_mult[j].cpu().detach().numpy(), 
                                                                global_orient_mult[j].cpu().detach().numpy()])
                                    best_epoch = epoch
                                    best_smpl = pred_smpl_output_mult.copy()
                                    best_silh_error = silh_error_mult
                                    best_joints_error = joints_error_mult

                                    best_global_orient_mult = global_orient_mult.clone().detach()
                                    best_cam_wp_mult = cam_wp_mult.clone().detach()
                                else:
                                    if epoch - max(current_epoch, best_epoch) > 10:
                                        earlier_stop = True
                                        break

                            if epoch == 1 and i == 0 and not is_train:
                                for metric in save_val_metrics:
                                    print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                        metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                                    metric,
                                                                                    metrics_tracker.history['val_' + metric][-1]))
                        if earlier_stop:
                            break

                    current_epoch = epoch
                    cam_wp_mult = best_cam_wp_mult
                    global_orient_mult = best_global_orient_mult
                    print('Finished translation optimization.')
                    print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                    print('Best epoch ', best_epoch)

                    if save_vis:
                        for index in range(len(joints2D_mult)):
                            best_image = wp_renderer.render(verts=best_smpl[index].vertices.cpu().detach().numpy()[0], 
                                                        cam=cam_wp_mult[index].cpu().detach().numpy()[0], img=image_mult[index])
                            opt_full_result = os.path.join(player_opt_result, view_mult[index])
                            cv2.imwrite(opt_full_result.replace('.png', '_{}_1.png'.format(i)), best_image)

                    for j in range(len(translation_mult)):
                        translation_mult[j] = translation_mult[j].detach()

                    criterion, metrics_tracker, save_val_metrics = init_loss_and_metric(True, False)
                    # optimize global orient and pose
                    betas_mean.requires_grad = True
                    body_poses_without_hands_feet.requires_grad = True
                    global_orient_mult.requires_grad = False
                    cam_wp_mult.requires_grad = False
                    params_pose = [body_poses_without_hands_feet, betas_mean]
                    optimiser_pose = optim.Adam(params_pose, lr=player_recon_train_regressor_learning_rate)
                    best_pose = body_poses_without_hands_feet.clone().detach()
                    best_betas_mean = betas_mean.clone().detach()

                    earlier_stop = False
                    for epoch in range(current_epoch, interation + current_epoch):
                        silh_error_mult = 0
                        joints_error_mult = 0
                        for is_train in [True, False]:
                            indexes = list(range(len(joints2D_mult)))
                            random.shuffle(indexes)
                            for index in indexes:
                                pred_smpl_output = smpl(body_pose=body_poses_with_hand_feet,
                                                        global_orient=global_orient_mult[index],
                                                        betas=betas_mean,
                                                        pose2rot=False)
                                pred_smpl_output_mult[index] = pred_smpl_output

                                pred_joints_all = pred_smpl_output.joints
                                pred_joints2d_coco = orthographic_project_torch(pred_joints_all, cam_wp_mult[index])
                                pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
                                pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                                                proxy_rep_input_wh)

                                # Need to expand cam_ts for NMR i.e.  from (B,
                                # 3)
                                # to
                                # (B, 1, 3)
                                translation_nmr = torch.unsqueeze(translation_mult[index], dim=1)
                                pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                                            faces=faces,
                                                            t=translation_nmr,
                                                            mode='silhouettes')

                                pred_dict_for_loss = {'joints2D': pred_joints2d_coco[0],
                                                        'silhouette': pred_silhouettes}
                                target_dict_for_loss = {'joints2D': torch.from_numpy(joints2D_mult[index][:,:2]).float().to(device),
                                                        'silhouette': torch.from_numpy(silhouette_mult[index]).float().to(device).unsqueeze(0)}

                                if not is_train:
                                    keypoints = pred_joints2d_coco.cpu().detach().numpy()[0].astype('int32')
                                    silh_error, iou_vis = compute_silh_error_metrics(pred_silhouettes.cpu().detach().numpy()[0], silhouette_mult[index], False)
                                    joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D_mult[index][:,:2])

                                    silh_error_mult += silh_error['iou']
                                    joints_error_mult += joints_error

                                # ---------------- BACKWARD PASS
                                # ----------------
                                if is_train:
                                    optimiser_pose.zero_grad()
                                loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

                                # ---------------- TRACK LOSS AND METRICS
                                # ----------------
                                num_train_inputs_in_batch = 1
                                if is_train:
                                    metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                                    pred_dict_for_loss, target_dict_for_loss,
                                                                    num_train_inputs_in_batch)
                                else:
                                    metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                                    pred_dict_for_loss, target_dict_for_loss,
                                                                    num_train_inputs_in_batch)

                                if is_train:
                                    loss.backward()
                                    optimiser_pose.step()
                        
                            if not is_train:
                                metrics_tracker.update_per_epoch()
                                save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                                    best_epoch_val_metrics)
                                if save_model_weights_this_epoch:
                                    for metric in save_val_metrics:
                                        best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
                                    best_model_wts = copy.deepcopy([body_poses_with_hand_feet.cpu().detach().numpy(), 
                                                            betas_mean.cpu().detach().numpy()])
                                    for j in range(len(view_mult)):
                                        best_model_view[j] = copy.deepcopy([translation_mult[j].cpu().detach().numpy(), 
                                                                global_orient_mult[j].cpu().detach().numpy()])
                                    best_epoch = epoch
                                    best_smpl = pred_smpl_output_mult.copy()
                                    best_silh_error = silh_error_mult
                                    best_joints_error = joints_error_mult

                                    best_pose = body_poses_without_hands_feet.clone().detach()
                                    best_betas_mean = betas_mean.clone().detach()
                                else:
                                    if epoch - max(current_epoch, best_epoch) > 20:
                                        earlier_stop = True
                                        break

                                if epoch == 1 and i == 0:
                                    for metric in save_val_metrics:
                                        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                            metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                                        metric,
                                                                                        metrics_tracker.history['val_' + metric][-1]))
                                metrics_tracker.initialise_loss_metric_sums()
                        if earlier_stop:
                            break

                    current_epoch = epoch
                    betas_mean = best_betas_mean
                    body_poses_without_hands_feet = best_pose
                    print('Finished pose optimization.')
                    print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                    print('Best epoch ', best_epoch)

                    if save_vis:
                        for index in range(len(joints2D_mult)):
                            best_image = wp_renderer.render(verts=best_smpl[index].vertices.cpu().detach().numpy()[0], 
                                                        cam=cam_wp_mult[index].cpu().detach().numpy()[0], img=image_mult[index])
                            opt_full_result = os.path.join(player_opt_result, view_mult[index])
                            cv2.imwrite(opt_full_result.replace('.png', '_{}_2.png'.format(i)), best_image)

                np.savez(opt_full, body_pose=best_model_wts[0], betas=best_model_wts[1])
                for j in range(len(view_mult)):
                    opt_view = os.path.join(player_opt, view_mult[j]).replace('.png', '.npz')
                    np.savez(opt_view, translation=best_model_view[j][0], global_orient=best_model_view[j][1])

                silh_mean_error_opt += best_silh_error
                joint_mean_error_opt += best_joints_error
                if is_refine:
                    print('After ', best_joints_error / len(joints2D_mult))
                with open(os.path.join(player_opt_result, 'metrics.xml'), 'w') as fs:
                    fs.write(json.dumps([best_silh_error / len(joints2D_mult), best_joints_error / len(joints2D_mult)]))

                endtime_player = timeit.default_timer()
                print('time player: {:.3f}'.format(endtime_player - starttime_player))
        #        break
        #    break
        #break
        endtime = timeit.default_timer()
        print('time: {:.3f}'.format(endtime - starttime))

    if (num_counter != 0):
        print("num_counter: {}".format(num_counter))
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
        print('silh_iou_opt: {}, joint_error_opt: {}'.format(silh_mean_error_opt / num_counter, joint_mean_error_opt / num_counter))

def calc_result(folder = 'Data/PlayerCrop_strap_multi_vis'):
    count = 0
    sil = 0
    score = 0

    games = os.listdir(folder)
    for game in games:
        game_full = os.path.join(folder, game)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            players = os.listdir(scene_full)
            for player in players:
                player_full = os.path.join(scene_full, player)
                score_file = os.path.join(player_full, 'metrics.xml')
                if (not os.path.exists(score_file)):
                    continue
                with open(score_file, 'r') as fs:
                    before = json.load(fs)
                count += 1
                sil += before[0]
                score += before[1]
    print("count: {}, silh_iou_opt: {}, joint_error_opt: {}".format(count, sil / count, score / count))

def evaluate_model_2d_separate(image_folder='Data/PlayerCrop', data_folder='Data/PlayerCrop_hmr', 
                   proxy_folder='Data/PlayerProxy', vis_folder='Data/PlayerCrop_hmr_vis',
                   result_folder='Data/PlayerCrop_hmr_opt', camera_folder='',
                   save_vis=True, opt = True):

    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    if save_vis:
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder, exist_ok=True)

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

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    starttime = timeit.default_timer()

    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    silh_mean_error_opt = 0
    joint_mean_error_opt = 0
    games = os.listdir(image_folder)
    for game in games:
        game_full = os.path.join(image_folder, game)
        game_dst = os.path.join(result_folder, game)
        remake_dir(game_dst)
        game_data = os.path.join(data_folder, game)
        game_camera = os.path.join(camera_folder, game)
        game_proxy = os.path.join(proxy_folder, game)
        if save_vis:
            game_vis = os.path.join(vis_folder, game)
            remake_dir(game_vis)
        scenes = os.listdir(game_full)
        for scene in scenes:
            print(scene)
            scene_full = os.path.join(game_full, scene)
            scene_dst = os.path.join(game_dst, scene)
            remake_dir(scene_dst)
            scene_data = os.path.join(game_data, scene)
            scene_camera = os.path.join(game_camera, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            if save_vis:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_full)
            for player in players:
                player_full = os.path.join(scene_full, player)
                player_dst = os.path.join(scene_dst, player)
                remake_dir(player_dst)
                player_data = os.path.join(scene_data, player)
                player_camera = os.path.join(scene_camera, player)
                player_proxy = os.path.join(scene_proxy, player)
                if save_vis:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)

                views = os.listdir(player_full)
                for view in views:
                    view_full = os.path.join(player_full, view)
                    #if hmr:
                    #    view_dst = os.path.join(player_dst, view.replace('.png', '.npz'))
                    #    view_data = os.path.join(player_data, view.replace('.png', '.npy'))
                    #elif spin:
                    view_dst = os.path.join(player_dst, view.replace('.png', '.npz'))
                    view_data = os.path.join(player_data, 'data.npz')
                    if not os.path.exists(view_data):
                        continue
                    view_camera = os.path.join(player_camera, view.replace('.png', '.npz'))
                    if not os.path.exists(view_camera):
                        continue
                    j2d_full = os.path.join(player_proxy, view).replace('.png', '_j2d.xml')
                    sil_full = os.path.join(player_proxy, view).replace('.png', '_sil.npy')
                    if save_vis:
                        view_vis = os.path.join(player_vis, view)

                    image = cv2.imread(view_full)

                    with open(j2d_full, 'r') as fs:
                        joints2D = np.array(json.load(fs))
                    silhouette = np.load(sil_full)

                    #if hmr:
                    #    pred_param = np.load(view_data)

                    #    pred_cam_wp = torch.from_numpy(pred_param[:,:3]).float().to(device)
                    #    pred_pose = torch.from_numpy(pred_param[:,3:75]).float().to(device)
                    #    pred_shape = torch.from_numpy(pred_param[:,75:]).float().to(device)

                    #    # Convert pred pose to rotation matrices
                    #    if pred_pose.shape[-1] == 24 * 3:
                    #        pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                    #        pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                    #    elif pred_pose.shape[-1] == 24 * 6:
                    #        pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)
                    #if spin:
                    
                    init_param = np.load(view_data)
                    pred_rotmat = init_param['body_pose']
                    pred_betas = init_param['betas']

                    camera_param = np.load(view_camera)
                    translation = camera_param['translation']
                    global_orient = camera_param['global_orient']

                    translation = torch.from_numpy(translation).float().to(device)
                    pred_cam_wp = convert_camera_translation_to_weak_perspective_torch(translation, config.FOCAL_LENGTH, proxy_rep_input_wh)

                    pred_pose_rotmats = torch.from_numpy(pred_rotmat).float().to(device)
                    pred_shape = torch.from_numpy(pred_betas).float().to(device)

                    body_pose = pred_pose_rotmats
                    global_orient = torch.from_numpy(global_orient).float().to(device)
                    betas = pred_shape
                    #print(body_pose.shape)
                    #print(betas.shape)
                    #print(global_orient.shape)
                    translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

                    if opt:
                        criterion, metrics_tracker, save_val_metrics = init_loss_and_metric(True, False)
                        best_epoch_val_metrics = {}
                        best_smpl = None
                        best_epoch = 1
                        best_model_wts = copy.deepcopy([body_pose.cpu().detach().numpy(), 
                                                        global_orient.cpu().detach().numpy(), 
                                                        betas.cpu().detach().numpy(), 
                                                        translation.cpu().detach().numpy()])
                        for metric in save_val_metrics:
                            best_epoch_val_metrics[metric] = np.inf

                        global_orient.requires_grad = True
                        pred_cam_wp.requires_grad = True
                        params = [global_orient, pred_cam_wp]
                        optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)
                        
                        for epoch in range(1, 100 + 1):
                            metrics_tracker.initialise_loss_metric_sums()
                            pred_smpl_output = smpl(body_pose=body_pose,
                                                    global_orient=global_orient,
                                                    betas=betas,
                                                    pose2rot=False)

                            if (epoch == 1 and save_vis):
                                rend_img = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                              cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                                cv2.imwrite(view_vis.replace('.png', '_0.png'), rend_img)

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
                        
                            keypoints = pred_joints2d_coco.cpu().detach().numpy()[0].astype('int32')
                            silh_error, iou_vis = compute_silh_error_metrics(pred_silhouettes.cpu().detach().numpy()[0], silhouette, False)
                            joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D[:,:2])
                        
                            if epoch == 1:
                                silh_mean_error_init += silh_error['iou']
                                joint_mean_error_init += joints_error

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

                            save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                            best_epoch_val_metrics)
                            if save_model_weights_this_epoch:
                                for metric in save_val_metrics:
                                    best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
                                best_model_wts = copy.deepcopy([body_pose.cpu().detach().numpy(), 
                                                        global_orient.cpu().detach().numpy(), 
                                                        betas.cpu().detach().numpy(), 
                                                        translation.cpu().detach().numpy()])
                                best_epoch = epoch
                                best_smpl = pred_smpl_output
                                best_silh_error = silh_error['iou']
                                best_joints_error = joints_error

                            if epoch == 1:
                                for metric in save_val_metrics:
                                    print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                        metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                                    metric,
                                                                                    metrics_tracker.history['val_' + metric][-1]))

                            loss.backward()
                            optimiser.step()
                            translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

                        print('Finished translation optimization.')
                        print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                        print('Best epoch ', best_epoch)
                    
                        if save_vis:
                            best_image = wp_renderer.render(verts=best_smpl.vertices.cpu().detach().numpy()[0], 
                                                        cam=convert_camera_translation_to_weak_perspective(best_model_wts[3][0], config.FOCAL_LENGTH, proxy_rep_input_wh), img=image)
                            cv2.imwrite(view_vis.replace('.png', '_1.png'), best_image)

                        np.savez(view_dst, body_pose=best_model_wts[0], global_orient=best_model_wts[1],
                                 betas=best_model_wts[2], translation=best_model_wts[3])

                        silh_mean_error_opt += best_silh_error
                        joint_mean_error_opt += best_joints_error
                        num_counter += 1
        #        break
        #    break
        #break
    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
        print('silh_iou_opt: {}, joint_error_opt: {}'.format(silh_mean_error_opt / num_counter, joint_mean_error_opt / num_counter))

def single_view_optimization(save_vis=True,
                             image_folder=player_crop_data_folder, proxy_folder=player_recon_proxy_folder,
                             data_folder=player_crop_data_folder+'_spin_opt',
                             result_folder=player_recon_single_view_opt_folder,
                             vis_folder=player_recon_single_view_opt_result_folder,
                             ignore_first=True,
                             interation=player_recon_single_view_iteration):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder, exist_ok=True)

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

    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)
    faces = torch.cat(1 * [faces[None, :]], dim=0)

    # Starting training loop
    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    silh_mean_error_opt = 0
    joint_mean_error_opt = 0
    games = os.listdir(image_folder)
    for game in games:
        game_full = os.path.join(image_folder, game)
        game_proxy = os.path.join(proxy_folder, game)
        game_opt = os.path.join(result_folder, game)
        game_opt_result = os.path.join(vis_folder, game)
        game_data = os.path.join(data_folder, game)

        remake_dir(game_opt)
        remake_dir(game_opt_result)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            scene_opt = os.path.join(game_opt, scene)
            scene_opt_result = os.path.join(game_opt_result, scene)
            scene_data = os.path.join(game_data, scene)
            remake_dir(scene_opt)
            remake_dir(scene_opt_result)

            players = os.listdir(scene_full)
            for player in players:
                starttime = timeit.default_timer()
                player_full = os.path.join(scene_full, player)
                if (os.path.isfile(player_full)):
                    continue
                if (ignore_first and player == '1'):
                    continue
                player_proxy = os.path.join(scene_proxy, player)
                player_opt = os.path.join(scene_opt, player)
                player_opt_result = os.path.join(scene_opt_result, player)
                player_data = os.path.join(scene_data, player)
                remake_dir(player_opt)
                remake_dir(player_opt_result)

                views = os.listdir(player_full)
                for view in views:
                    view_full = os.path.join(player_full, view)
                    print('process {}'.format(view_full))
                    image = cv2.imread(view_full)
                    j2d_full = os.path.join(player_proxy, view).replace('.png', '_j2d.xml')
                    sil_full = os.path.join(player_proxy, view).replace('.png', '_sil.npy')
                    with open(j2d_full, 'r') as fs:
                        joints2D = np.array(json.load(fs))
                    silhouette = np.load(sil_full)

                    opt_full = os.path.join(player_opt, view).replace('.png', '.npz')
                    opt_full_result = os.path.join(player_opt_result, view)

                    init_full = os.path.join(player_data, view).replace('.png', '.npz')
                    if not os.path.exists(init_full):
                        continue
                    init_param = np.load(init_full)
                    body_pose = init_param['body_pose']
                    global_orient = init_param['global_orient']
                    betas = init_param['betas']
                    translation = init_param['translation']

                    body_pose = torch.from_numpy(body_pose).float().to(device)
                    global_orient = torch.from_numpy(global_orient).float().to(device)
                    betas = torch.from_numpy(betas).float().to(device)
                    translation = torch.from_numpy(translation).float().to(device)
                    pred_cam_wp = convert_camera_translation_to_weak_perspective_torch(translation, config.FOCAL_LENGTH, proxy_rep_input_wh)

                    criterion, metrics_tracker, save_val_metrics = init_loss_and_metric(True, False)

                    # ----------------------- Optimiser -----------------------
                    body_poses_without_hands_feet = torch.cat([body_pose[:, :6, :, :],
                                                           body_pose[:, 8:21, :, :]],
                                                          dim=1)

                    best_epoch_val_metrics = {}
                    best_smpl = None
                    best_epoch = 1
                    best_model_wts = copy.deepcopy([body_pose.cpu().detach().numpy(), 
                                                    global_orient.cpu().detach().numpy(), 
                                                    betas.cpu().detach().numpy(), 
                                                    translation.cpu().detach().numpy()])
                    for metric in save_val_metrics:
                        best_epoch_val_metrics[metric] = np.inf

                    #print('translation before')
                    #print(translation)
                    # optimize global orientation and translation
                    global_orient.requires_grad = True
                    betas.requires_grad = True
                    #translation.requires_grad = False
                    body_poses_without_hands_feet.requires_grad = True
                    pred_cam_wp.requires_grad = True
                    params = [global_orient, body_poses_without_hands_feet, pred_cam_wp, betas] # + list(criterion.parameters())
                    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

                    for epoch in range(1, interation + 1):
                        metrics_tracker.initialise_loss_metric_sums()
                        body_poses_with_hand_feet = torch.cat([body_poses_without_hands_feet[:, :6, :, :],
                                        body_pose[:, 6:8, :, :],
                                        body_poses_without_hands_feet[:, 6:19, :, :],
                                        body_pose[:, 21:, :, :]],
                                       dim=1)
                        pred_smpl_output = smpl(body_pose=body_poses_with_hand_feet,
                                                global_orient=global_orient,
                                                betas=betas,
                                                pose2rot=False)

                        if (epoch == 1 and save_vis):
                            rend_img = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                          cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                            cv2.imwrite(opt_full_result.replace('.png', '_0.png'), rend_img)

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
                        
                        keypoints = pred_joints2d_coco.cpu().detach().numpy()[0].astype('int32')
                        silh_error, iou_vis = compute_silh_error_metrics(pred_silhouettes.cpu().detach().numpy()[0], silhouette, False)
                        joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D[:,:2])
                        
                        if epoch == 1:
                            silh_mean_error_init += silh_error['iou']
                            joint_mean_error_init += joints_error

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

                        save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                        best_epoch_val_metrics)
                        if save_model_weights_this_epoch:
                            for metric in save_val_metrics:
                                best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
                            best_model_wts = copy.deepcopy([body_pose.cpu().detach().numpy(), 
                                                    global_orient.cpu().detach().numpy(), 
                                                    betas.cpu().detach().numpy(), 
                                                    translation.cpu().detach().numpy()])
                            best_epoch = epoch
                            best_smpl = pred_smpl_output
                            best_silh_error = silh_error['iou']
                            best_joints_error = joints_error

                        if epoch == 1:
                            for metric in save_val_metrics:
                                print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                    metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                                metric,
                                                                                metrics_tracker.history['val_' + metric][-1]))

                        loss.backward()
                        optimiser.step()
                        translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

                    print('Finished translation optimization.')
                    print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                    print('Best epoch ', best_epoch)
                    
                    if save_vis:
                        best_image = wp_renderer.render(verts=best_smpl.vertices.cpu().detach().numpy()[0], 
                                                    cam=convert_camera_translation_to_weak_perspective(best_model_wts[3][0], config.FOCAL_LENGTH, proxy_rep_input_wh), img=image)
                        cv2.imwrite(opt_full_result.replace('.png', '_1.png'), best_image)

                    np.savez(opt_full, body_pose=best_model_wts[0], global_orient=best_model_wts[1],
                             betas=best_model_wts[2], translation=best_model_wts[3])

                    silh_mean_error_opt += best_silh_error
                    joint_mean_error_opt += best_joints_error
                    num_counter += 1
                endtime = timeit.default_timer()
                print('time: {:.3f}'.format(endtime - starttime))
        #        break
        #    break
        #break
        
    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
        print('silh_iou_opt: {}, joint_error_opt: {}'.format(silh_mean_error_opt / num_counter, joint_mean_error_opt / num_counter))

def multi_view_optimization_multi(save_vis=True, game_continue='',
                            image_folder=player_crop_data_folder, proxy_folder=player_recon_proxy_folder,
                            result_folder=player_crop_data_folder+'_spin_multi_multi', vis_folder=player_crop_data_folder+'_spin_multi_multi_vis',
                            single_folder=player_crop_data_folder+'_spin_single', ignore_first=True,
                            image_folder1=player_crop_broad_image_folder, proxy_folder1=player_broad_proxy_folder,
                            result_folder1=player_crop_broad_image_folder+'_spin_multi_multi', vis_folder1=player_crop_broad_image_folder+'_spin_multi_multi_vis',
                            single_folder1=player_crop_broad_image_folder+'_spin_single',
                            interation=player_recon_multi_view_iteration):
    print('multi view opt')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder, exist_ok=True)
    if not os.path.exists(result_folder1):
        os.makedirs(result_folder1, exist_ok=True)
    if not os.path.exists(vis_folder1):
        os.makedirs(vis_folder1, exist_ok=True)

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

    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    faces = torch.from_numpy(smpl.faces.astype(np.int32)).float().to(device)
    faces = torch.cat(1 * [faces[None, :]], dim=0)

    # Starting training loop
    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    silh_mean_error_opt = 0
    joint_mean_error_opt = 0
    games = os.listdir(image_folder)
    for game in games:
        starttime = timeit.default_timer()
        game_full = os.path.join(image_folder, game)
        game_proxy = os.path.join(proxy_folder, game)
        game_init = os.path.join(single_folder, game)
        game_opt = os.path.join(result_folder, game)
        game_opt_result = os.path.join(vis_folder, game)

        game_full1 = os.path.join(image_folder1, game)
        game_proxy1 = os.path.join(proxy_folder1, game)
        game_init1 = os.path.join(single_folder1, game)
        game_opt1 = os.path.join(result_folder1, game)
        game_opt_result1 = os.path.join(vis_folder1, game)
        if os.path.exists(game_opt) and game != game_continue:
            continue

        remake_dir(game_opt)
        remake_dir(game_opt_result)
        remake_dir(game_opt1)
        remake_dir(game_opt_result1)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            scene_init = os.path.join(game_init, scene)
            scene_opt = os.path.join(game_opt, scene)
            scene_opt_result = os.path.join(game_opt_result, scene)

            scene_full1 = os.path.join(game_full1, scene)
            scene_proxy1 = os.path.join(game_proxy1, scene)
            scene_init1 = os.path.join(game_init1, scene)
            scene_opt1 = os.path.join(game_opt1, scene)
            scene_opt_result1 = os.path.join(game_opt_result1, scene)

            remake_dir(scene_opt)
            remake_dir(scene_opt_result)
            remake_dir(scene_opt1)
            remake_dir(scene_opt_result1)
            players = os.listdir(scene_full)
            for player in players:
                starttime_player = timeit.default_timer()
                player_full = os.path.join(scene_full, player)
                player_full1 = os.path.join(scene_full1, player)
                if (os.path.isfile(player_full)):
                    continue
                if (ignore_first and player == '1'):
                    continue
                player_proxy = os.path.join(scene_proxy, player)
                player_init = os.path.join(scene_init, player)
                player_opt = os.path.join(scene_opt, player)
                player_opt_result = os.path.join(scene_opt_result, player)

                player_proxy1 = os.path.join(scene_proxy1, player)
                player_init1 = os.path.join(scene_init1, player)
                player_opt1 = os.path.join(scene_opt1, player)
                player_opt_result1 = os.path.join(scene_opt_result1, player)

                remake_dir(player_opt)
                remake_dir(player_opt_result)
                remake_dir(player_opt1)
                remake_dir(player_opt_result1)

                print('process {}'.format(player_full))

                joints2D_mult = []
                silhouette_mult = []
                body_pose_mult = []
                global_orient_mult = []
                betas_mult = []
                translation_mult = []
                view_mult = []
                image_mult = []
                cam_wp_mult = []
                
                opt_full = os.path.join(player_opt, 'data.npz')
                opt_full1 = os.path.join(player_opt1, 'data.npz')
                views = os.listdir(player_full)
                views1 = os.listdir(player_full1)
                for view in views:
                    view_full = os.path.join(player_full, view)
                    
                    image = cv2.imread(view_full)
                    j2d_full = os.path.join(player_proxy, view).replace('.png', '_j2d.xml')
                    sil_full = os.path.join(player_proxy, view).replace('.png', '_sil.npy')
                    with open(j2d_full, 'r') as fs:
                        joints2D = np.array(json.load(fs))
                    silhouette = np.load(sil_full)

                    init_full = os.path.join(player_init, view).replace('.png', '.npz')
                    if not os.path.exists(init_full):
                        continue
                    init_param = np.load(init_full)
                    body_pose = init_param['body_pose']
                    global_orient = init_param['global_orient']
                    betas = init_param['betas']
                    translation = init_param['translation']

                    joints2D_mult.append(joints2D)
                    silhouette_mult.append(silhouette)
                    body_pose_mult.append(torch.from_numpy(body_pose).float().to(device))
                    global_orient_mult.append(torch.from_numpy(global_orient).float().to(device))
                    betas_mult.append(torch.from_numpy(betas).float().to(device))
                    translation = torch.from_numpy(translation).float().to(device)
                    translation_mult.append(translation)
                    view_mult.append(view)
                    image_mult.append(image)
                    cam_wp_mult.append(convert_camera_translation_to_weak_perspective_torch(translation, config.FOCAL_LENGTH, proxy_rep_input_wh))

                for view1 in views1:
                    view_full1 = os.path.join(player_full1, view1)
                    
                    image = cv2.imread(view_full1)
                    j2d_full = os.path.join(player_proxy1, view1).replace('.png', '_j2d.xml')
                    sil_full = os.path.join(player_proxy1, view1).replace('.png', '_sil.npy')
                    with open(j2d_full, 'r') as fs:
                        joints2D = np.array(json.load(fs))
                    silhouette = np.load(sil_full)

                    init_full = os.path.join(player_init1, view1).replace('.png', '.npz')
                    if not os.path.exists(init_full):
                        continue
                    init_param = np.load(init_full)
                    body_pose = init_param['body_pose']
                    global_orient = init_param['global_orient']
                    betas = init_param['betas']
                    translation = init_param['translation']

                    joints2D_mult.append(joints2D)
                    silhouette_mult.append(silhouette)
                    body_pose_mult.append(torch.from_numpy(body_pose).float().to(device))
                    global_orient_mult.append(torch.from_numpy(global_orient).float().to(device))
                    betas_mult.append(torch.from_numpy(betas).float().to(device))
                    translation = torch.from_numpy(translation).float().to(device)
                    translation_mult.append(translation)
                    view_mult.append(view1)
                    image_mult.append(image)
                    cam_wp_mult.append(convert_camera_translation_to_weak_perspective_torch(translation, config.FOCAL_LENGTH, proxy_rep_input_wh))

                body_pose_mean = torch.stack(body_pose_mult)
                betas_mean = torch.stack(betas_mult)
                body_pose_mean = body_pose_mean.mean(dim=0)
                betas_mean = betas_mean.mean(dim = 0)

                #body_pose_mean += (torch.rand_like(body_pose_mean).float().to(device) - 0.5) * 0.1
                #betas_mean += ((torch.rand_like(betas_mean).float().to(device) - 0.5) * 2.5) * 0.1

                pred_smpl_output_mult = [None] * len(joints2D_mult)
                global_orient_mult = torch.stack(global_orient_mult)
                #translation_mult = torch.stack(translation_mult)
                cam_wp_mult = torch.stack(cam_wp_mult)

                criterion, metrics_tracker, save_val_metrics = init_loss_and_metric(True, False)

                best_pose = torch.cat([body_pose_mean[:, :6, :, :],
                                        body_pose_mean[:, 8:21, :, :]],
                                        dim=1)
                best_epoch_val_metrics = {}
                best_epoch = 1
                best_model_wts = copy.deepcopy([body_pose_mean.cpu().detach().numpy(), 
                                                betas_mean.cpu().detach().numpy()])
                best_model_view = []
                for j in range(len(view_mult)):
                    best_model_view.append(copy.deepcopy([translation_mult[j].cpu().detach().numpy(), 
                                            global_orient_mult[j].cpu().detach().numpy()]))
                for metric in save_val_metrics:
                    best_epoch_val_metrics[metric] = np.inf
                current_epoch = 1
                for i in range(3):

                    # ----------------------- Optimiser -----------------------
                    body_poses_without_hands_feet = best_pose
                    body_poses_hands_feet = torch.cat([body_pose_mean[:, 6:8, :, :],
                                                            body_pose_mean[:, 21:, :, :]],
                                                            dim=1)

                    body_poses_with_hand_feet = torch.cat([body_poses_without_hands_feet[:, :6, :, :],
                                            body_poses_hands_feet[:, 0:2, :, :],
                                            body_poses_without_hands_feet[:, 6:19, :, :],
                                            body_poses_hands_feet[:, 2:, :, :]],
                                            dim=1)

                    # optimize camera
                    betas_mean.requires_grad = True
                    body_poses_without_hands_feet.requires_grad = False
                    global_orient_mult.requires_grad = False
                    cam_wp_mult.requires_grad = True
                    params = [cam_wp_mult, betas_mean]
                    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

                    best_cam_wp_mult = cam_wp_mult.clone().detach()
                    best_betas_mean = betas_mean.clone().detach()

                    earlier_stop = False
                    for epoch in range(current_epoch, interation + current_epoch):
                        silh_error_mult = 0
                        joints_error_mult = 0
                        metrics_tracker.initialise_loss_metric_sums()
                        for is_train in [True, False]:
                            indexes = list(range(len(joints2D_mult)))
                            if is_train:
                                random.shuffle(indexes)
                            for index in indexes:
                                pred_smpl_output = smpl(body_pose=body_poses_with_hand_feet,
                                                        global_orient=global_orient_mult[index],
                                                        betas=betas_mean,
                                                        pose2rot=False)
                                pred_smpl_output_mult[index] = pred_smpl_output

                                if (epoch == 1 and i == 0 and save_vis and not is_train):
                                    rend_img = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                                    cam=cam_wp_mult[index].cpu().detach().numpy()[0], img=image_mult[index])
                                    if index == len(indexes) - 1:
                                        opt_full_result = os.path.join(player_opt_result1, view_mult[index])
                                    else:
                                        opt_full_result = os.path.join(player_opt_result, view_mult[index])
                                    cv2.imwrite(opt_full_result.replace('.png', '_{}_0.png'.format(i)), rend_img)

                                pred_joints_all = pred_smpl_output.joints
                                pred_joints2d_coco = orthographic_project_torch(pred_joints_all, cam_wp_mult[index])
                                pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
                                pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                                                proxy_rep_input_wh)

                                # Need to expand cam_ts for NMR i.e.  from (B, 3)
                                # to
                                # (B, 1, 3)
                                translation_nmr = torch.unsqueeze(translation_mult[index], dim=1)
                                pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                                            faces=faces,
                                                            t=translation_nmr,
                                                            mode='silhouettes')

                                if not is_train:
                                    keypoints = pred_joints2d_coco.cpu().detach().numpy()[0].astype('int32')
                                    silh_error, iou_vis = compute_silh_error_metrics(pred_silhouettes.cpu().detach().numpy()[0], silhouette_mult[index], False)
                                    joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D_mult[index][:,:2])

                                    silh_error_mult += silh_error['iou']
                                    joints_error_mult += joints_error

                                if epoch == 1 and i == 0 and not is_train:
                                    silh_mean_error_init += silh_error['iou']
                                    joint_mean_error_init += joints_error
                                    num_counter += 1

                                pred_dict_for_loss = {'joints2D': pred_joints2d_coco[0],
                                                        'silhouette': pred_silhouettes}
                                target_dict_for_loss = {'joints2D': torch.from_numpy(joints2D_mult[index][:,:2]).float().to(device),
                                                        'silhouette': torch.from_numpy(silhouette_mult[index]).float().to(device).unsqueeze(0)}

                                # ---------------- BACKWARD PASS ----------------
                                if is_train:
                                    optimiser.zero_grad()
                                loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

                                # ---------------- TRACK LOSS AND METRICS
                                # ----------------
                                num_train_inputs_in_batch = 1
                                if is_train:
                                    metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                                        pred_dict_for_loss, target_dict_for_loss,
                                                                        num_train_inputs_in_batch)
                                else:
                                    metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                                    pred_dict_for_loss, target_dict_for_loss,
                                                                    num_train_inputs_in_batch)

                                if is_train:
                                    loss.backward()
                                    optimiser.step()
                                    translation_mult[index] = convert_weak_perspective_to_camera_translation_torch(cam_wp_mult[index], config.FOCAL_LENGTH, proxy_rep_input_wh)

                            if not is_train:
                                metrics_tracker.update_per_epoch()

                                save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                                    best_epoch_val_metrics)
                                if save_model_weights_this_epoch:
                                    for metric in save_val_metrics:
                                        best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
                                    best_model_wts = copy.deepcopy([body_poses_with_hand_feet.cpu().detach().numpy(), 
                                                            betas_mean.cpu().detach().numpy()])
                                    for j in range(len(view_mult)):
                                        best_model_view[j] = copy.deepcopy([translation_mult[j].cpu().detach().numpy(), 
                                                                global_orient_mult[j].cpu().detach().numpy()])
                                    best_epoch = epoch
                                    best_smpl = pred_smpl_output_mult.copy()
                                    best_silh_error = silh_error_mult
                                    best_joints_error = joints_error_mult

                                    best_cam_wp_mult = cam_wp_mult.clone().detach()
                                    best_betas_mean = betas_mean.clone().detach()
                                    #print(best_cam_wp_mult)
                                    #print(best_betas_mean)
                                else:
                                    if epoch - max(current_epoch, best_epoch) > 10:
                                        earlier_stop = True
                                        break

                            if epoch == 1 and i == 0 and not is_train:
                                for metric in save_val_metrics:
                                    print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                        metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                                    metric,
                                                                                    metrics_tracker.history['val_' + metric][-1]))
                        if earlier_stop:
                            break

                    current_epoch = epoch
                    cam_wp_mult = best_cam_wp_mult
                    betas_mean = best_betas_mean
                    #print(best_cam_wp_mult)
                    #print(best_betas_mean)
                    print('Finished translation optimization.')
                    print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                    print('Best epoch ', best_epoch)

                    if save_vis:
                        for index in range(len(joints2D_mult)):
                            best_image = wp_renderer.render(verts=best_smpl[index].vertices.cpu().detach().numpy()[0], 
                                                        cam=cam_wp_mult[index].cpu().detach().numpy()[0], img=image_mult[index])
                            if index == len(joints2D_mult) - 1:
                                opt_full_result = os.path.join(player_opt_result1, view_mult[index])
                            else:
                                opt_full_result = os.path.join(player_opt_result, view_mult[index])
                            cv2.imwrite(opt_full_result.replace('.png', '_{}_1.png'.format(i)), best_image)

                    for j in range(len(translation_mult)):
                        translation_mult[j] = translation_mult[j].detach()

                    criterion, metrics_tracker, save_val_metrics = init_loss_and_metric(True, False)
                    # optimize global orient and pose
                    betas_mean.requires_grad = False
                    body_poses_without_hands_feet.requires_grad = True
                    global_orient_mult.requires_grad = True
                    cam_wp_mult.requires_grad = False
                    params_pose = [body_poses_without_hands_feet, global_orient_mult]
                    optimiser_pose = optim.Adam(params_pose, lr=player_recon_train_regressor_learning_rate)

                    best_pose = body_poses_without_hands_feet.clone().detach()
                    best_global_orient_mult = global_orient_mult.clone().detach()

                    earlier_stop = False
                    for epoch in range(current_epoch, interation + current_epoch):
                        silh_error_mult = 0
                        joints_error_mult = 0
                        for is_train in [True, False]:
                            indexes = list(range(len(joints2D_mult)))
                            if is_train:
                                indexes.extend(indexes)
                                random.shuffle(indexes)
                                indexes = indexes[:-5]
                            for index in indexes:
                                pred_smpl_output = smpl(body_pose=body_poses_with_hand_feet,
                                                        global_orient=global_orient_mult[index],
                                                        betas=betas_mean,
                                                        pose2rot=False)
                                pred_smpl_output_mult[index] = pred_smpl_output

                                pred_joints_all = pred_smpl_output.joints
                                pred_joints2d_coco = orthographic_project_torch(pred_joints_all, cam_wp_mult[index])
                                pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
                                pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                                                proxy_rep_input_wh)

                                # Need to expand cam_ts for NMR i.e.  from (B,
                                # 3)
                                # to
                                # (B, 1, 3)
                                translation_nmr = torch.unsqueeze(translation_mult[index], dim=1)
                                pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                                            faces=faces,
                                                            t=translation_nmr,
                                                            mode='silhouettes')

                                pred_dict_for_loss = {'joints2D': pred_joints2d_coco[0],
                                                        'silhouette': pred_silhouettes}
                                target_dict_for_loss = {'joints2D': torch.from_numpy(joints2D_mult[index][:,:2]).float().to(device),
                                                        'silhouette': torch.from_numpy(silhouette_mult[index]).float().to(device).unsqueeze(0)}

                                if not is_train:
                                    keypoints = pred_joints2d_coco.cpu().detach().numpy()[0].astype('int32')
                                    silh_error, iou_vis = compute_silh_error_metrics(pred_silhouettes.cpu().detach().numpy()[0], silhouette_mult[index], False)
                                    joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D_mult[index][:,:2])

                                    silh_error_mult += silh_error['iou']
                                    joints_error_mult += joints_error

                                # ---------------- BACKWARD PASS
                                # ----------------
                                if is_train:
                                    optimiser_pose.zero_grad()
                                loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)

                                # ---------------- TRACK LOSS AND METRICS
                                # ----------------
                                num_train_inputs_in_batch = 1
                                if is_train:
                                    metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                                    pred_dict_for_loss, target_dict_for_loss,
                                                                    num_train_inputs_in_batch)
                                else:
                                    metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                                    pred_dict_for_loss, target_dict_for_loss,
                                                                    num_train_inputs_in_batch)

                                if is_train:
                                    loss.backward()
                                    optimiser_pose.step()
                        
                            if not is_train:
                                metrics_tracker.update_per_epoch()
                                save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                                    best_epoch_val_metrics)
                                if save_model_weights_this_epoch:
                                    for metric in save_val_metrics:
                                        best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
                                    best_model_wts = copy.deepcopy([body_poses_with_hand_feet.cpu().detach().numpy(), 
                                                            betas_mean.cpu().detach().numpy()])
                                    for j in range(len(view_mult)):
                                        best_model_view[j] = copy.deepcopy([translation_mult[j].cpu().detach().numpy(), 
                                                                global_orient_mult[j].cpu().detach().numpy()])
                                    best_epoch = epoch
                                    best_smpl = pred_smpl_output_mult.copy()
                                    best_silh_error = silh_error_mult
                                    best_joints_error = joints_error_mult

                                    best_pose = body_poses_without_hands_feet.clone().detach()
                                    best_global_orient_mult = global_orient_mult.clone().detach()
                                    #print(best_pose)
                                    #print(best_global_orient_mult)
                                else:
                                    if epoch - max(current_epoch, best_epoch) > 20:
                                        earlier_stop = True
                                        break

                                if epoch == 1 and i == 0:
                                    for metric in save_val_metrics:
                                        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                            metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                                        metric,
                                                                                        metrics_tracker.history['val_' + metric][-1]))
                                metrics_tracker.initialise_loss_metric_sums()
                        if earlier_stop:
                            break

                    current_epoch = epoch
                    global_orient_mult = best_global_orient_mult
                    body_poses_without_hands_feet = best_pose
                    #print(best_pose)
                    #print(best_global_orient_mult)
                    print('Finished pose optimization.')
                    print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                    print('Best epoch ', best_epoch)

                    if save_vis:
                        for index in range(len(joints2D_mult)):
                            best_image = wp_renderer.render(verts=best_smpl[index].vertices.cpu().detach().numpy()[0], 
                                                        cam=cam_wp_mult[index].cpu().detach().numpy()[0], img=image_mult[index])
                            if (index == len(joints2D_mult) - 1):
                                opt_full_result = os.path.join(player_opt_result1, view_mult[index])
                            else:
                                opt_full_result = os.path.join(player_opt_result, view_mult[index])
                            cv2.imwrite(opt_full_result.replace('.png', '_{}_2.png'.format(i)), best_image)

                np.savez(opt_full, body_pose=best_model_wts[0], betas=best_model_wts[1])
                np.savez(opt_full1, body_pose=best_model_wts[0], betas=best_model_wts[1])
                for j in range(len(view_mult)):
                    if j == len(view_mult) - 1:
                        opt_view = os.path.join(player_opt1, view_mult[j]).replace('.png', '.npz')
                    else:
                        opt_view = os.path.join(player_opt, view_mult[j]).replace('.png', '.npz')
                    np.savez(opt_view, translation=best_model_view[j][0], global_orient=best_model_view[j][1])

                silh_mean_error_opt += best_silh_error
                joint_mean_error_opt += best_joints_error

                with open(os.path.join(player_opt_result, 'metrics.xml'), 'w') as fs:
                    fs.write(json.dumps([best_silh_error / len(joints2D_mult), best_joints_error / len(joints2D_mult)]))
                with open(os.path.join(player_opt_result1, 'metrics.xml'), 'w') as fs:
                    fs.write(json.dumps([best_silh_error / len(joints2D_mult), best_joints_error / len(joints2D_mult)]))

                endtime_player = timeit.default_timer()
                print('time player: {:.3f}'.format(endtime_player - starttime_player))
        #        break
        #    break
        #break
        endtime = timeit.default_timer()
        print('time: {:.3f}'.format(endtime - starttime))

    if (num_counter != 0):
        print("num_counter: {}".format(num_counter))
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
        print('silh_iou_opt: {}, joint_error_opt: {}'.format(silh_mean_error_opt / num_counter, joint_mean_error_opt / num_counter))

def evaluate_model_2d_oneview(image_folder='Data/PlayerCrop/F - Lazio - Dortmund', data_folder='Data/PlayerCrop_spin_single/F - Lazio - Dortmund', 
                   proxy_folder='Data/PlayerProxy/F - Lazio - Dortmund', vis_folder='Data/PlayerCrop_oneview_vis/F - Lazio - Dortmund',
                   result_folder='Data/PlayerCrop_oneview/F - Lazio - Dortmund',
                   save_vis=True, opt = True):

    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    if save_vis:
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder, exist_ok=True)

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

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    starttime = timeit.default_timer()

    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    silh_mean_error_opt = 0
    joint_mean_error_opt = 0
    games = ['1']
    for game in games:
        game_full = image_folder
        game_dst = result_folder
        game_data = data_folder
        game_proxy = proxy_folder
        if save_vis:
            game_vis = vis_folder
        scenes = os.listdir(game_full)
        for scene in scenes:
            print(scene)
            scene_full = os.path.join(game_full, scene)
            scene_dst = os.path.join(game_dst, scene)
            remake_dir(scene_dst)
            scene_data = os.path.join(game_data, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            if save_vis:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)
            players = os.listdir(scene_full)
            for player in players:
                player_full = os.path.join(scene_full, player)
                player_dst = os.path.join(scene_dst, player)
                remake_dir(player_dst)
                player_data = os.path.join(scene_data, player)
                player_proxy = os.path.join(scene_proxy, player)
                if save_vis:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)

                views = os.listdir(player_full)
                for view1 in views:
                    view_data = os.path.join(player_data, view1.replace('.png', '.npz'))
                    init_param = np.load(view_data)
                    body_pose = init_param['body_pose']
                    betas = init_param['betas']

                    body_pose = torch.from_numpy(body_pose).float().to(device)
                    betas = torch.from_numpy(betas).float().to(device)

                    for view in views:
                        view_full = os.path.join(player_full, view)
                        view_dst = os.path.join(player_dst, view.replace('.png', '.npz'))
                    
                        j2d_full = os.path.join(player_proxy, view).replace('.png', '_j2d.xml')
                        sil_full = os.path.join(player_proxy, view).replace('.png', '_sil.npy')
                        if save_vis:
                            view_vis = os.path.join(player_vis, view)

                        view_data = os.path.join(player_data, view.replace('.png', '.npz'))
                        init_param = np.load(view_data)
                        global_orient = init_param['global_orient']
                        translation = init_param['translation']

                        global_orient = torch.from_numpy(global_orient).float().to(device)
                        translation = torch.from_numpy(translation).float().to(device)
                        pred_cam_wp = convert_camera_translation_to_weak_perspective_torch(translation, config.FOCAL_LENGTH, proxy_rep_input_wh)

                        image = cv2.imread(view_full)

                        with open(j2d_full, 'r') as fs:
                            joints2D = np.array(json.load(fs))
                        silhouette = np.load(sil_full)

                        if opt:
                            criterion, metrics_tracker, save_val_metrics = init_loss_and_metric(True, False)
                            best_epoch_val_metrics = {}
                            best_smpl = None
                            best_epoch = 1
                            best_model_wts = copy.deepcopy([body_pose.cpu().detach().numpy(), 
                                                            global_orient.cpu().detach().numpy(), 
                                                            betas.cpu().detach().numpy(), 
                                                            translation.cpu().detach().numpy()])
                            for metric in save_val_metrics:
                                best_epoch_val_metrics[metric] = np.inf

                            global_orient.requires_grad = True
                            pred_cam_wp.requires_grad = True
                            params = [global_orient, pred_cam_wp]
                            optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)
                        
                            for epoch in range(1, 50 + 1):
                                metrics_tracker.initialise_loss_metric_sums()
                                pred_smpl_output = smpl(body_pose=body_pose,
                                                        global_orient=global_orient,
                                                        betas=betas,
                                                        pose2rot=False)

                                if (epoch == 1 and save_vis):
                                    rend_img = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                                  cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                                    cv2.imwrite(view_vis.replace('.png', '_{}_0.png'.format(view1)), rend_img)

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
                        
                                keypoints = pred_joints2d_coco.cpu().detach().numpy()[0].astype('int32')
                                silh_error, iou_vis = compute_silh_error_metrics(pred_silhouettes.cpu().detach().numpy()[0], silhouette, False)
                                joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D[:,:2])
                        
                                if epoch == 1:
                                    silh_mean_error_init += silh_error['iou']
                                    joint_mean_error_init += joints_error

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

                                save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                                best_epoch_val_metrics)
                                if save_model_weights_this_epoch:
                                    for metric in save_val_metrics:
                                        best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
                                    best_model_wts = copy.deepcopy([body_pose.cpu().detach().numpy(), 
                                                            global_orient.cpu().detach().numpy(), 
                                                            betas.cpu().detach().numpy(), 
                                                            translation.cpu().detach().numpy()])
                                    best_epoch = epoch
                                    best_smpl = pred_smpl_output
                                    best_silh_error = silh_error['iou']
                                    best_joints_error = joints_error

                                if epoch == 1:
                                    for metric in save_val_metrics:
                                        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                            metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                                        metric,
                                                                                        metrics_tracker.history['val_' + metric][-1]))

                                loss.backward()
                                optimiser.step()
                                translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

                            print('Finished translation optimization.')
                            print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                            print('Best epoch ', best_epoch)
                    
                            if save_vis:
                                best_image = wp_renderer.render(verts=best_smpl.vertices.cpu().detach().numpy()[0], 
                                                            cam=convert_camera_translation_to_weak_perspective(best_model_wts[3][0], config.FOCAL_LENGTH, proxy_rep_input_wh), img=image)
                                cv2.imwrite(view_vis.replace('.png', '_{}_1.png'.format(view1)), best_image)

                            np.savez(view_dst, body_pose=best_model_wts[0], global_orient=best_model_wts[1],
                                     betas=best_model_wts[2], translation=best_model_wts[3])

                            silh_mean_error_opt += best_silh_error
                            joint_mean_error_opt += best_joints_error
                            num_counter += 1
        #        break
        #    break
        #break
    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
        print('silh_iou_opt: {}, joint_error_opt: {}'.format(silh_mean_error_opt / num_counter, joint_mean_error_opt / num_counter))

def evaluate_model_2d_cross(image_folder=player_crop_data_folder, data_folder=player_crop_data_folder + '_hmr_opt', 
                   proxy_folder=player_recon_proxy_folder, vis_folder= player_crop_data_folder + '_hmr_cross_vis',
                   result_folder=player_crop_data_folder +'_hmr_cross',
                   image_folder1=player_crop_broad_image_folder, data_folder1=player_crop_broad_image_folder + '_hmr_opt', 
                   proxy_folder1=player_broad_proxy_folder, vis_folder1= player_crop_broad_image_folder + '_hmr_cross_vis',
                   result_folder1=player_crop_broad_image_folder +'_hmr_cross',
                   save_vis=True, opt = True):

    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    if save_vis:
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder, exist_ok=True)

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

    #if render_vis:
        # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    starttime = timeit.default_timer()

    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    silh_mean_error_opt = 0
    joint_mean_error_opt = 0
    games = os.listdir(image_folder)
    for game in games:
        game_full = os.path.join(image_folder, game)
        game_dst = os.path.join(result_folder, game)
        if os.path.exists(game_dst):
            print('skip ' + game_full)
            continue

        remake_dir(game_dst)
        game_data = os.path.join(data_folder, game)
        game_proxy = os.path.join(proxy_folder, game)
        if save_vis:
            game_vis = os.path.join(vis_folder, game)
            remake_dir(game_vis)

        game_full1 = os.path.join(image_folder1, game)
        game_dst1 = os.path.join(result_folder1, game)
        remake_dir(game_dst1)
        game_data1 = os.path.join(data_folder1, game)
        game_proxy1 = os.path.join(proxy_folder1, game)
        if save_vis:
            game_vis1 = os.path.join(vis_folder1, game)
            remake_dir(game_vis1)
        scenes = os.listdir(game_full)
        for scene in scenes:
            print(scene)
            scene_full = os.path.join(game_full, scene)
            scene_dst = os.path.join(game_dst, scene)
            remake_dir(scene_dst)
            scene_data = os.path.join(game_data, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            if save_vis:
                scene_vis = os.path.join(game_vis, scene)
                remake_dir(scene_vis)

            scene_full1 = os.path.join(game_full1, scene)
            scene_dst1 = os.path.join(game_dst1, scene)
            remake_dir(scene_dst1)
            scene_data1 = os.path.join(game_data1, scene)
            scene_proxy1 = os.path.join(game_proxy1, scene)
            if save_vis:
                scene_vis1 = os.path.join(game_vis1, scene)
                remake_dir(scene_vis1)
            players = os.listdir(scene_full)
            for player in players:
                player_full = os.path.join(scene_full, player)
                player_dst = os.path.join(scene_dst, player)
                remake_dir(player_dst)
                player_data = os.path.join(scene_data, player)
                player_proxy = os.path.join(scene_proxy, player)
                if save_vis:
                    player_vis = os.path.join(scene_vis, player)
                    remake_dir(player_vis)

                player_full1 = os.path.join(scene_full1, player)
                player_dst1 = os.path.join(scene_dst1, player)
                remake_dir(player_dst1)
                player_data1 = os.path.join(scene_data1, player)
                player_proxy1 = os.path.join(scene_proxy1, player)
                if save_vis:
                    player_vis1 = os.path.join(scene_vis1, player)
                    remake_dir(player_vis1)

                view_full_multi = []
                j2d_full_multi = []
                sil_full_multi = []
                view_vis_multi = []
                init_full_multi = []
                view_result_multi = []

                views = os.listdir(player_full)
                for view in views:
                    view_full = os.path.join(player_full, view)
                    j2d_full = os.path.join(player_proxy, view).replace('.png', '_j2d.xml')
                    sil_full = os.path.join(player_proxy, view).replace('.png', '_sil.npy')
                    if save_vis:
                        view_vis = os.path.join(player_vis, view)
                    init_full = os.path.join(player_data, view).replace('.png', '.npz')
                    if not os.path.exists(init_full):
                        continue
                    view_result = os.path.join(player_dst, view).replace('.png', '.npz')

                    view_full_multi.append(view_full)
                    j2d_full_multi.append(j2d_full)
                    sil_full_multi.append(sil_full)
                    view_vis_multi.append(view_vis)
                    init_full_multi.append(init_full)
                    view_result_multi.append(view_result)

                views = os.listdir(player_full1)
                for view in views:
                    view_full = os.path.join(player_full1, view)
                    j2d_full = os.path.join(player_proxy1, view).replace('.png', '_j2d.xml')
                    sil_full = os.path.join(player_proxy1, view).replace('.png', '_sil.npy')
                    if save_vis:
                        view_vis = os.path.join(player_vis1, view)
                    init_full = os.path.join(player_data1, view).replace('.png', '.npz')
                    if not os.path.exists(init_full):
                        continue
                    view_result = os.path.join(player_dst1, view).replace('.png', '.npz')

                    view_full_multi.append(view_full)
                    j2d_full_multi.append(j2d_full)
                    sil_full_multi.append(sil_full)
                    view_vis_multi.append(view_vis)
                    init_full_multi.append(init_full)
                    view_result_multi.append(view_result)

                for i in range(len(view_full_multi)):
                    init_full = init_full_multi[i]
                    init_param = np.load(init_full)
                    body_pose = init_param['body_pose']
                    betas = init_param['betas']

                    body_pose = torch.from_numpy(body_pose).float().to(device)
                    betas = torch.from_numpy(betas).float().to(device)

                    for j in range(len(view_full_multi)):
                        view_full = view_full_multi[j]
                        j2d_full = j2d_full_multi[j]
                        sil_full = sil_full_multi[j]
                        if save_vis:
                            view_vis = view_vis_multi[j].replace(".png", "_{}.png".format(i))
                        view_result = view_result_multi[j].replace(".npz", "_{}.npz".format(i))

                        image = cv2.imread(view_full)

                        with open(j2d_full, 'r') as fs:
                            joints2D = np.array(json.load(fs))
                        silhouette = np.load(sil_full)

                        init_full = init_full_multi[j]
                        init_param = np.load(init_full)
                        global_orient = init_param['global_orient']
                        translation = init_param['translation']

                        global_orient = torch.from_numpy(global_orient).float().to(device)
                        translation = torch.from_numpy(translation).float().to(device)
                        pred_cam_wp = convert_camera_translation_to_weak_perspective_torch(translation, config.FOCAL_LENGTH, proxy_rep_input_wh)

                        if opt:
                            criterion, metrics_tracker, save_val_metrics = init_loss_and_metric(True, False)
                            best_epoch_val_metrics = {}
                            best_smpl = None
                            best_epoch = 1
                            best_model_wts = copy.deepcopy([body_pose.cpu().detach().numpy(), 
                                                            global_orient.cpu().detach().numpy(), 
                                                            betas.cpu().detach().numpy(), 
                                                            translation.cpu().detach().numpy()])
                            for metric in save_val_metrics:
                                best_epoch_val_metrics[metric] = np.inf

                            global_orient.requires_grad = True
                            pred_cam_wp.requires_grad = True
                            params = [global_orient, pred_cam_wp]
                            optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)
                        
                            for epoch in range(1, 20 + 1):
                                metrics_tracker.initialise_loss_metric_sums()
                                pred_smpl_output = smpl(body_pose=body_pose,
                                                        global_orient=global_orient,
                                                        betas=betas,
                                                        pose2rot=False)

                                if (epoch == 1 and save_vis):
                                    rend_img = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                                  cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                                    cv2.imwrite(view_vis.replace('.png', '_0.png'), rend_img)

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
                        
                                keypoints = pred_joints2d_coco.cpu().detach().numpy()[0].astype('int32')
                                silh_error, iou_vis = compute_silh_error_metrics(pred_silhouettes.cpu().detach().numpy()[0], silhouette, False)
                                joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D[:,:2])
                        
                                if epoch == 1:
                                    silh_mean_error_init += silh_error['iou']
                                    joint_mean_error_init += joints_error

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

                                save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                                best_epoch_val_metrics)
                                if save_model_weights_this_epoch:
                                    for metric in save_val_metrics:
                                        best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
                                    best_model_wts = copy.deepcopy([body_pose.cpu().detach().numpy(), 
                                                            global_orient.cpu().detach().numpy(), 
                                                            betas.cpu().detach().numpy(), 
                                                            translation.cpu().detach().numpy()])
                                    best_epoch = epoch
                                    best_smpl = pred_smpl_output
                                    best_silh_error = silh_error['iou']
                                    best_joints_error = joints_error

                                if epoch == 1:
                                    for metric in save_val_metrics:
                                        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                            metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                                        metric,
                                                                                        metrics_tracker.history['val_' + metric][-1]))

                                loss.backward()
                                optimiser.step()
                                translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

                            print('Finished translation optimization.')
                            print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                            print('Best epoch ', best_epoch)
                    
                            if save_vis:
                                best_image = wp_renderer.render(verts=best_smpl.vertices.cpu().detach().numpy()[0], 
                                                            cam=convert_camera_translation_to_weak_perspective(best_model_wts[3][0], config.FOCAL_LENGTH, proxy_rep_input_wh), img=image)
                                cv2.imwrite(view_vis.replace('.png', '_1.png'), best_image)

                            np.savez(view_result, body_pose=best_model_wts[0], global_orient=best_model_wts[1],
                                     betas=best_model_wts[2], translation=best_model_wts[3])

                            silh_mean_error_opt += best_silh_error
                            joint_mean_error_opt += best_joints_error
                            num_counter += 1

                            with open(view_vis.replace('.png', '.xml'), 'w') as fs:
                                fs.write(json.dumps([best_silh_error, best_joints_error]))
        #        break
        #    break
        #break
    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
        print('silh_iou_opt: {}, joint_error_opt: {}'.format(silh_mean_error_opt / num_counter, joint_mean_error_opt / num_counter))

#evaluate_model_2d()
#evaluate_model_2d(data_folder='Data/PlayerCrop_spin',vis_folder='Data/PlayerCrop_spin_vis',result_folder='Data/PlayerCrop_spin_opt',hmr=False,spin=True)
#multi_view_optimization(single_folder=player_recon_strap_result_folder, result_folder=player_crop_data_folder+'_strap_multi',vis_folder=player_crop_data_folder+'_strap_multi_vis')
#multi_view_optimization(single_folder='Data/PlayerCrop_spin_opt', result_folder=player_crop_data_folder+'_spin_multi',vis_folder=player_crop_data_folder+'_spin_multi_vis')
#calc_result()

#evaluate_model_2d(image_folder=player_crop_broad_image_folder, data_folder='Data/PlayerBroadImage_hmr', vis_folder='Data/PlayerBroadImage_hmr_vis',
#                  result_folder='Data/PlayerBroadImage_hmr_opt', proxy_folder=player_broad_proxy_folder)
#evaluate_model_2d(image_folder=player_crop_broad_image_folder, data_folder='Data/PlayerBroadImage_spin', vis_folder='Data/PlayerBroadImage_spin_vis',
#                  result_folder='Data/PlayerBroadImage_spin_opt', proxy_folder=player_broad_proxy_folder, hmr=False, spin=True)
#evaluate_model_2d_separate(image_folder=player_crop_broad_image_folder, data_folder='Data/PlayerCrop_spin_multi', vis_folder='Data/PlayerBroadImage_spin_orbit_vis',
#                  result_folder='Data/PlayerBroadImage_spin_orbit', proxy_folder=player_broad_proxy_folder, camera_folder='Data/PlayerBroadImage_spin_opt')
#single_view_optimization(result_folder=player_crop_data_folder+'_spin_single', vis_folder=player_crop_data_folder+'_spin_single_vis')
#single_view_optimization(result_folder=player_crop_broad_image_folder+'_spin_single', vis_folder=player_crop_broad_image_folder+'_spin_single_vis',
#                         image_folder=player_crop_broad_image_folder, proxy_folder=player_broad_proxy_folder,data_folder='Data/PlayerBroadImage_spin_opt')
#multi_view_optimization_multi()

#calc_result('Data/PlayerCrop_spin_multi_multi_vis')

#multi_view_optimization_multi(result_folder=player_crop_data_folder+'_spin_multi_no_single', vis_folder=player_crop_data_folder+'_spin_multi_no_single_vis',
#                            single_folder=player_crop_data_folder+'_spin_opt',
#                            result_folder1=player_crop_broad_image_folder+'_spin_multi_no_single', vis_folder1=player_crop_broad_image_folder+'_spin_multi_no_single_vis',
#                            single_folder1=player_crop_broad_image_folder+'_spin_opt')

#evaluate_model_2d_oneview()

#evaluate_model_2d_cross()
#evaluate_model_2d_cross(data_folder = player_crop_data_folder + '_spin_opt', vis_folder = player_crop_data_folder + '_spin_cross_vis',
#                        result_folder = player_crop_data_folder + '_spin_cross', 
#                        data_folder1 = player_crop_broad_image_folder + '_spin_opt', vis_folder1 = player_crop_broad_image_folder + '_spin_cross_vis',
#                        result_folder1 = player_crop_broad_image_folder + '_spin_cross')

#evaluate_model_2d_cross(data_folder = 'Data/strap', vis_folder = player_crop_data_folder + '_strap_cross_vis',
#                        result_folder = player_crop_data_folder + '_strap_cross', 
#                        data_folder1 = player_crop_broad_image_folder + '_strap', vis_folder1 = player_crop_broad_image_folder + '_strap_cross_vis',
#                        result_folder1 = player_crop_broad_image_folder + '_strap_cross')

#evaluate_model_2d(image_folder=player_crop_broad_image_folder, data_folder='Data/PlayerBroadImage_pare', vis_folder='Data/PlayerBroadImage_pare_vis',
#                  result_folder='Data/PlayerBroadImage_pare_opt', proxy_folder=player_broad_proxy_folder, hmr=False, pare=True)

#evaluate_model_2d(image_folder=player_crop_data_folder, data_folder='Data/PlayerCrop_pare', vis_folder='Data/PlayerCrop_pare_vis',
#                  result_folder='Data/PlayerCrop_pare_opt', proxy_folder=player_recon_proxy_folder, hmr=False, pare=True)

#evaluate_model_2d_cross(data_folder = player_crop_data_folder + '_pare_opt', vis_folder = player_crop_data_folder + '_pare_cross_vis',
#                        result_folder = player_crop_data_folder + '_pare_cross', 
#                        data_folder1 = player_crop_broad_image_folder + '_pare_opt', vis_folder1 = player_crop_broad_image_folder + '_pare_cross_vis',
#                        result_folder1 = player_crop_broad_image_folder + '_pare_cross')

#multi_view_optimization(single_folder='Data/PlayerCrop_pare_opt', result_folder=player_crop_data_folder+'_pare_multi',vis_folder=player_crop_data_folder+'_pare_multi_vis')

#calc_result('Data/PlayerCrop_pare_multi_vis')

#evaluate_model_2d_separate(image_folder=player_crop_broad_image_folder, data_folder='Data/PlayerCrop_pare_multi', vis_folder='Data/PlayerBroadImage_pare_orbit_vis',
#                  result_folder='Data/PlayerBroadImage_pare_orbit', proxy_folder=player_broad_proxy_folder, camera_folder='Data/PlayerBroadImage_pare_opt')

#multi_view_optimization_multi(result_folder=player_crop_data_folder+'_pare_multi_no_single', vis_folder=player_crop_data_folder+'_pare_multi_no_single_vis',
#                            single_folder=player_crop_data_folder+'_pare_opt',
#                            result_folder1=player_crop_broad_image_folder+'_pare_multi_no_single', vis_folder1=player_crop_broad_image_folder+'_pare_multi_no_single_vis',
#                            single_folder1=player_crop_broad_image_folder+'_pare_opt')

#single_view_optimization(result_folder=player_crop_broad_image_folder+'_pare_single', vis_folder=player_crop_broad_image_folder+'_pare_single_vis',
#                         image_folder=player_crop_broad_image_folder, proxy_folder=player_broad_proxy_folder,data_folder='Data/PlayerBroadImage_pare_opt')
#single_view_optimization(result_folder=player_crop_data_folder+'_pare_single', vis_folder=player_crop_data_folder+'_pare_single_vis',
#                         image_folder=player_crop_data_folder, proxy_folder=player_recon_proxy_folder,data_folder=player_crop_data_folder + '_pare_opt')

multi_view_optimization_multi(result_folder=player_crop_data_folder+'_pare_multi_multi', vis_folder=player_crop_data_folder+'_pare_multi_multi_vis',
                            single_folder=player_crop_data_folder+'_pare_single',
                            result_folder1=player_crop_broad_image_folder+'_pare_multi_multi', vis_folder1=player_crop_broad_image_folder+'_pare_multi_multi_vis',
                            single_folder1=player_crop_broad_image_folder+'_pare_single')