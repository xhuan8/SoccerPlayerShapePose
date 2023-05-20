import os
from pickle import FALSE
import sys
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

def train_regressor_iuv(load_checkpoint=True, data_argument=False):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    device_text = 'cuda:0'
    gpu_index = 0
    device = torch.device(device_text)

    if not os.path.exists(player_recon_train_regressor_checkpoints_folder+'iuvsp'):
        os.makedirs(player_recon_train_regressor_checkpoints_folder, exist_ok=True)
    if not os.path.exists(player_recon_train_regressor_logs_folder+'iuvsp'):
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

    regressor = SingleInputRegressor(resnet_in_channels=21,
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
    checkpoint_path = os.path.join(player_recon_train_regressor_checkpoints_folder+'iuvsp', 'best.tar')
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
                game_dst = os.path.join(player_broad_proxy_folder+'unrefine', game)
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

                    proxy_rep = torch.from_numpy(np.array(proxy_rep_batch)).float().to(device)
                    iuv_rep_batch = torch.from_numpy(np.array(iuv_rep_batch)).float().to(device) / 255
                    iuv_rep_batch = torch.cat((proxy_rep, iuv_rep_batch), 1)

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
                    model_save_path = os.path.join(player_recon_train_regressor_checkpoints_folder+'iuvsp', 'best.tar')
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
                    model_save_path = os.path.join(player_recon_train_regressor_checkpoints_folder+'iuvsp', 'model')
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

train_regressor_iuv(True)
