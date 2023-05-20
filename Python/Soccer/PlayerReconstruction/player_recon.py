import os
from pickle import TRUE
import sys
from weakref import proxy
import torch
import cv2
import numpy as np
import torch.optim as optim
import neural_renderer as nr
import copy
import timeit
import json
import random

from models.regressor import SingleInputRegressor
from predict.predict_3D import *

from predict.predict_joints2D import predict_joints2D
from predict.predict_silhouette_pointrend import predict_silhouette_pointrend
from predict.predict_densepose import predict_densepose, apply_colormap

from utils.label_conversions import convert_multiclass_to_binary_labels, \
    convert_2Djoints_to_gaussian_heatmaps

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

def create_proxy(silhouettes_from='densepose', folder=player_crop_data_folder,
                 data_folder=player_recon_proxy_folder, vis_folder=player_recon_proxy_vis_folder, ignore_first=True):
    # Set-up proxy representation predictors.
    joints2D_predictor, silhouette_predictor = setup_detectron2_predictors(silhouettes_from=silhouettes_from)

    games = os.listdir(folder)
    remake_dir(data_folder)
    remake_dir(vis_folder)
    for ii,game in enumerate(games):
        print('{}/{}'.format(ii, len(games)))
        game_full = os.path.join(folder, game)
        game_dst = os.path.join(data_folder, game)
        game_vis = os.path.join(vis_folder, game)
        remake_dir(game_dst)
        remake_dir(game_vis)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_dst = os.path.join(game_dst, scene)
            scene_vis = os.path.join(game_vis, scene)
            remake_dir(scene_dst)
            remake_dir(scene_vis)
            players = os.listdir(scene_full)
            for player in players:
                player_full = os.path.join(scene_full, player)
                print('process {}'.format(player_full))
                if (os.path.isfile(player_full)):
                    continue
                if (ignore_first and player == '1'):
                    continue
                player_dst = os.path.join(scene_dst, player)
                player_vis = os.path.join(scene_vis, player)
                remake_dir(player_dst)
                remake_dir(player_vis)
                views = os.listdir(player_full)
                for view in views:
                    view_full = os.path.join(player_full, view)
                    image = cv2.imread(view_full)
                    image = cv2.resize(image, (proxy_rep_input_wh, proxy_rep_input_wh),
                               interpolation=cv2.INTER_LINEAR)
                    joints2D, joints2D_vis = predict_joints2D(image, joints2D_predictor)
                    if silhouettes_from == 'pointrend':
                        silhouette, silhouette_vis = predict_silhouette_pointrend(image,
                                                                                  silhouette_predictor)
                    elif silhouettes_from == 'densepose':
                        silhouette, silhouette_vis = predict_densepose(image, silhouette_predictor)
                        if silhouette is not None:
                            silhouette = convert_multiclass_to_binary_labels(silhouette)
                    if silhouette is not None and joints2D is not None:
                        for j in range(joints2D.shape[0]):
                            cv2.circle(silhouette_vis, (int(joints2D[j, 0]), int(joints2D[j, 1])), 5, (0, 255, 0), -1)
                        cv2.imwrite(os.path.join(player_vis, view).replace('.png','_silhouette.png'), silhouette_vis)
                        #cv2.imwrite(os.path.join(player_vis,
                        #view).replace('.png','_joints2D.png'), joints2D_vis)
                        with open(os.path.join(player_dst, view).replace('.png', '_j2d.xml'), 'w') as fs:
                            fs.write(json.dumps(joints2D.tolist()))
                        np.save(os.path.join(player_dst, view).replace('.png', '_sil.npy'), silhouette)
                    else:
                        shutil.rmtree(player_dst)
                        shutil.rmtree(player_vis)
                        break
        #        break
        #    break
        #break

#create_proxy('densepose', player_crop_data_folder, player_recon_proxy_folder, player_recon_proxy_vis_folder)
#create_proxy('densepose', player_crop_broad_image_folder,
#player_broad_proxy_folder, player_broad_proxy_vis_folder)
#create_proxy('pointrend', real_images_player, real_images_player_proxy, real_images_player_proxy_vis)

#recreate_proxy_vis(real_images_player, real_images_player_proxy,
#real_images_player_proxy_vis)
def predict(silhouettes_from='densepose', save_proxy_vis=True, render_vis=True):
    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    #print(os.getcwd())
    checkpoint = torch.load(player_recon_check_points_path, map_location=device)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])
    print("Regressor loaded. Weights from:", player_recon_check_points_path)

    # Set-up proxy representation predictors.
    joints2D_predictor, silhouette_predictor = setup_detectron2_predictors(silhouettes_from=silhouettes_from)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    if render_vis:
        # Set-up renderer for visualisation.
        wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    games = os.listdir(player_crop_data_folder)
    remake_dir(player_recon_result_folder)
    for game in games:
        game_full = os.path.join(player_crop_data_folder, game)
        game_dst = os.path.join(player_recon_result_folder, game)
        remake_dir(game_dst)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_dst = os.path.join(game_dst, scene)
            remake_dir(scene_dst)
            players = os.listdir(scene_full)
            for player in players:
                player_full = os.path.join(scene_full, player)
                print('process {}'.format(player_full))
                if (player == '1' or os.path.isfile(player_full)):
                    continue
                player_dst = os.path.join(scene_dst, player)
                remake_dir(player_dst)
                views = os.listdir(player_full)
                for view in views:
                    view_full = os.path.join(player_full, view)
                    image = cv2.imread(view_full)
                    image = cv2.resize(image, (proxy_rep_input_wh, proxy_rep_input_wh),
                               interpolation=cv2.INTER_LINEAR)

                    # Predict 2D
                    joints2D, joints2D_vis = predict_joints2D(image, joints2D_predictor)
                    if silhouettes_from == 'pointrend':
                        silhouette, silhouette_vis = predict_silhouette_pointrend(image,
                                                                                  silhouette_predictor)
                    elif silhouettes_from == 'densepose':
                        silhouette, silhouette_vis = predict_densepose(image, silhouette_predictor)
                        silhouette = convert_multiclass_to_binary_labels(silhouette)

                    # Create proxy representation
                    proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                            in_wh=proxy_rep_input_wh,
                                                            out_wh=config.REGRESSOR_IMG_WH)
                    proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
                    proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

                    # Predict 3D
                    regressor.eval()
                    with torch.no_grad():
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

                        pred_reposed_smpl_output = smpl(betas=pred_shape)
                        pred_reposed_vertices = pred_reposed_smpl_output.vertices

                    # Numpy-fying
                    pred_vertices = pred_vertices.cpu().detach().numpy()[0]
                    pred_vertices2d = pred_vertices2d.cpu().detach().numpy()[0]
                    pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()[0]
                    pred_cam_wp = pred_cam_wp.cpu().detach().numpy()[0]

                    plt.figure()
                    plt.imshow(image[:,:,::-1])
                    plt.scatter(pred_vertices2d[:, 0], pred_vertices2d[:, 1], s=0.3)
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    #plt.savefig(os.path.join(player_dst, view).replace('.png',
                    #'_verts.png'))
                    plt.close()

                    if render_vis:
                        rend_img = wp_renderer.render(verts=pred_vertices, cam=pred_cam_wp, img=image)
                        rend_reposed_img = wp_renderer.render(verts=pred_reposed_vertices,
                                                              cam=np.array([0.8, 0., -0.2]),
                                                              angle=180,
                                                              axis=[1, 0, 0])

                        cv2.imwrite(os.path.join(player_dst, view).replace('.png', '_rend.png'), rend_img)
                        #cv2.imwrite(os.path.join(player_dst,
                        #view).replace('.png', '_reposed.png'),
                        #rend_reposed_img)
                    if save_proxy_vis:
                        cv2.imwrite(os.path.join(player_dst, view).replace('.png', '_silhouette.png'), silhouette_vis)
                        cv2.imwrite(os.path.join(player_dst, view).replace('.png', '_joints2D.png'), joints2D_vis)

#predict()
# evaluate metrics
def eval_metrics(load_checkpoint=False, silhouettes_from='densepose', save_proxy_vis=True, render_vis=True):
    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    regressor.eval()

    #print(os.getcwd())
    if not load_checkpoint or not os.path.exists(os.path.join(player_recon_train_regressor_checkpoints_folder, 'best.tar')):
        checkpoint = torch.load(player_recon_check_points_path, map_location=device)
        print("Regressor loaded. Weights from:", player_recon_check_points_path)
    else:
        checkpoint = torch.load(os.path.join(player_recon_train_regressor_checkpoints_folder, 'best.tar'), map_location=device)
        print("Regressor loaded. Weights from:", os.path.join(player_recon_train_regressor_checkpoints_folder, 'best.tar'))
    regressor.load_state_dict(checkpoint['best_model_state_dict'])

    # Set-up proxy representation predictors.
    joints2D_predictor, silhouette_predictor = setup_detectron2_predictors(silhouettes_from=silhouettes_from)

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

    if render_vis:
        # Set-up renderer for visualisation.
        wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    silh_mean_error = 0
    num_counter = 0
    joint_mean_error = 0
    games = os.listdir(player_crop_data_folder)
    remake_dir(player_recon_result_folder)
    for game in games:
        game_full = os.path.join(player_crop_data_folder, game)
        game_dst = os.path.join(player_recon_result_folder, game)
        remake_dir(game_dst)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            print('process {}'.format(scene_full))
            scene_dst = os.path.join(game_dst, scene)
            remake_dir(scene_dst)
            players = os.listdir(scene_full)
            for player in players:
                player_full = os.path.join(scene_full, player)
                
                if (player == '1' or os.path.isfile(player_full)):
                    continue
                player_dst = os.path.join(scene_dst, player)
                remake_dir(player_dst)
                views = os.listdir(player_full)
                for view in views:
                    view_full = os.path.join(player_full, view)
                    image = cv2.imread(view_full)
                    image = cv2.resize(image, (proxy_rep_input_wh, proxy_rep_input_wh),
                               interpolation=cv2.INTER_LINEAR)

                    # Predict 2D
                    joints2D, joints2D_vis = predict_joints2D(image, joints2D_predictor)
                    if silhouettes_from == 'pointrend':
                        silhouette, silhouette_vis = predict_silhouette_pointrend(image,
                                                                                  silhouette_predictor)
                    elif silhouettes_from == 'densepose':
                        silhouette, silhouette_vis = predict_densepose(image, silhouette_predictor)
                        silhouette = convert_multiclass_to_binary_labels(silhouette)

                    # Create proxy representation
                    #for ii in range(silhouette.shape[0]):
                    #    print(silhouette[ii])
                    proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                            in_wh=proxy_rep_input_wh,
                                                            out_wh=config.REGRESSOR_IMG_WH)
                    proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
                    proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

                    # Predict 3D
                    with torch.no_grad():
                        pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep)
                        #print(pred_cam_wp)
                        #print(pred_pose)
                        #print(pred_shape)
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
                        #pred_joints2d =
                        #orthographic_project_torch(smpl_joints, pred_cam_wp)
                        #pred_joints2d =
                        #undo_keypoint_normalisation(pred_joints2d,
                        #                                              proxy_rep_input_wh)

                        pred_reposed_smpl_output = smpl(betas=pred_shape)
                        pred_reposed_vertices = pred_reposed_smpl_output.vertices

                    # Numpy-fying
                    pred_vertices = pred_vertices.cpu().detach().numpy()[0]
                    pred_vertices2d = pred_vertices2d.cpu().detach().numpy()[0]
                    pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()[0]
                    pred_cam_wp = pred_cam_wp.cpu().detach().numpy()[0]

                    plt.figure()
                    plt.imshow(image[:,:,::-1])
                    plt.scatter(pred_vertices2d[:, 0], pred_vertices2d[:, 1], s=0.3)
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    #plt.savefig(os.path.join(player_dst, view).replace('.png',
                    #'_verts.png'))
                    plt.close()

                    if render_vis:
                        rend_img = wp_renderer.render(verts=pred_vertices, cam=pred_cam_wp, img=image)
                        rend_reposed_img = wp_renderer.render(verts=pred_reposed_vertices,
                                                              cam=np.array([0.8, 0., -0.2]),
                                                              angle=180,
                                                              axis=[1, 0, 0])

                        cv2.imwrite(os.path.join(player_dst, view).replace('.png', '_rend.png'), rend_img)
                        #cv2.imwrite(os.path.join(player_dst,
                        #view).replace('.png', '_reposed.png'),
                        #rend_reposed_img)
                    if save_proxy_vis:
                        a = 0
                        #cv2.imwrite(os.path.join(player_dst,
                        #view).replace('.png', '_silhouette.png'),
                        #silhouette_vis)
                        #cv2.imwrite(os.path.join(player_dst,
                        #view).replace('.png', '_joints2D.png'), joints2D_vis)
                    
                    rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                          -1, -1)
                    translation = convert_weak_perspective_to_camera_translation(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
                    translation = torch.Tensor(translation).to(smpl_joints.device).unsqueeze(0)
                    pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
                                                                 config.FOCAL_LENGTH, proxy_rep_input_wh)
                    
                    pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

                    keypoints = pred_joints2d.cpu().detach().numpy()[0].astype('int32')
                    image = np.copy(image)
                    for j in range(keypoints.shape[0]):
                        cv2.circle(image, (keypoints[j, 0], keypoints[j, 1]), 5, (0, 255, 0), -1)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.5
                        fontColor = (0, 0, 255)
                        #cv2.putText(image, str(j), (keypoints[j, 0],
                        #keypoints[j, 1]),
                        #                         font, fontScale, fontColor,
                        #                         lineType=2)
                    for j in range(joints2D.shape[0]):
                        cv2.circle(silhouette_vis, (joints2D[j, 0], joints2D[j, 1]), 5, (0, 255, 0), -1)
                    #cv2.imwrite(os.path.join(player_dst, view).replace('.png',
                    #'_project_joints.png'), image)
                    #cv2.imwrite(os.path.join(player_dst, view).replace('.png',
                    #'_both.png'), silhouette_vis)

                    rend_img = wp_renderer.render(verts=pred_vertices, cam=pred_cam_wp)
                    rend_img = rend_img[:,:,0]

                    silh_error, iou_vis = compute_silh_error_metrics(rend_img, silhouette, True)
                    #cv2.imwrite(os.path.join(player_dst, view).replace('.png',
                    #'_silh_iou.png'), iou_vis)

                    rend_img_vis = np.copy(rend_img)
                    rend_img_vis[rend_img_vis != 0] = 1
                    rend_img_vis = apply_colormap(rend_img_vis, vmin=0, vmax=1)
                    rend_img_vis = rend_img_vis[:, :, :3]
                    #cv2.imwrite(os.path.join(player_dst, view).replace('.png',
                    #'_project_silhouette.png'), rend_img_vis)
                    
                    joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D[:,:2])

                    silh_mean_error += silh_error['iou']
                    num_counter += 1
                    joint_mean_error += joints_error

                    #cv2.imwrite('1.png', silhouette)
                    #print(keypoints)
                    #print(joints2D)
                    # Need to expand cam_ts for NMR i.e.  from (B, 3) to (B, 1,
                    # 3)
                    #translation_nmr = torch.unsqueeze(translation, dim=1)
                    #pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                    #                            faces=faces,
                    #                            t=translation_nmr,
                    #                            mode='silhouettes')
                    
                    #pred_silhouettes, _ =
                    #nmr_parts_renderer(pred_smpl_output.vertices, translation)
                    #pred_silhouettes =
                    #pred_silhouettes.cpu().detach().numpy()[0]
                    #for ii in range(pred_silhouettes.shape[0]):
                    #    print(pred_silhouettes[ii])
                    #pred_silhouettes[pred_silhouettes != 0] = 255
                    #cv2.imwrite(os.path.join(player_dst, view).replace('.png',
                    #'_project_silhouette1.png'), rend_img)
                    break
                break
            break
        break

    if (num_counter != 0):
        print('silh_iou: {}, joint_error: {}'.format(silh_mean_error / num_counter, joint_mean_error / num_counter))

#eval_metrics()
#eval_metrics(True)
def train_regressor(item='pose', load_checkpoint=True):
    if not os.path.exists(player_recon_train_regressor_checkpoints_folder):
        os.makedirs(player_recon_train_regressor_checkpoints_folder, exist_ok=True)
    if not os.path.exists(player_recon_train_regressor_logs_folder):
        os.makedirs(player_recon_train_regressor_logs_folder, exist_ok=True)

    if (item == 'pose'):
        losses_on = ['joints2D']
    elif (item == 'shape'):
        losses_on = ['silhouette']
    else:
        losses_on = ['joints2D', 'silhouette']
    init_loss_weights = {'joints2D': 1.0, 'silhouette': 1.0}
    losses_to_track = losses_on
    normalise_joints_before_loss = True
    if (item == 'pose'):
        metrics_to_track = ['joints2D_l2es']
    elif (item == 'shape'):
        metrics_to_track = ['silhouette_iou']
    else:
        metrics_to_track = ['joints2D_l2es', 'silhouette_iou']
    save_val_metrics = metrics_to_track
    epochs_per_save = 10

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
    faces = torch.cat(1 * [faces[None, :]], dim=0)

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
    num_epochs = player_recon_train_regressor_epoch + current_epoch
    metrics_tracker.initialise_loss_metric_sums()
    for epoch in range(current_epoch, num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for is_train in [False, True]:
            if (is_train):
                print('Training.')
                regressor.train()
                regressor.fix()
            else:
                print('eval')
                regressor.eval()

            games = os.listdir(player_crop_data_folder)
            if is_train:
                random.shuffle(games)
            for game in games:
                game_full = os.path.join(player_crop_data_folder, game)
                game_dst = os.path.join(player_recon_proxy_folder, game)
                scenes = os.listdir(game_full)
                if is_train:
                    random.shuffle(scenes)
                for scene in scenes:
                    scene_full = os.path.join(game_full, scene)
                    print('process {}'.format(scene_full))
                    scene_dst = os.path.join(game_dst, scene)
                    players = os.listdir(scene_full)
                    for player in players:
                        player_full = os.path.join(scene_full, player)
                        if (player == '1' or os.path.isfile(player_full)):
                            continue
                        player_dst = os.path.join(scene_dst, player)
                        views = os.listdir(player_full)
                        for view in views:
                            view_full = os.path.join(player_full, view)
                            j2d_full = os.path.join(player_dst, view).replace('.png', '_j2d.xml')
                            sil_full = os.path.join(player_dst, view).replace('.png', '_sil.npy')
                            with open(j2d_full, 'r') as fs:
                                joints2D = np.array(json.load(fs))
                            silhouette = np.load(sil_full)

                            proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                                in_wh=proxy_rep_input_wh,
                                                                out_wh=config.REGRESSOR_IMG_WH)
                            proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
                            proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

                            pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep)
                            #print(pred_cam_wp)
                            #print(pred_pose)
                            #print(pred_shape)
                            # Convert pred pose to rotation matrices
                            if pred_pose.shape[-1] == 24 * 3:
                                pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                                pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                            elif pred_pose.shape[-1] == 24 * 6:
                                pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                            #print(pred_pose_rotmats[:, 1:])
                            #print(pred_pose_rotmats[:, 1:].shape)
                            #print(pred_pose_rotmats[:, 0].unsqueeze(1))
                            #print(pred_pose_rotmats[:, 0].unsqueeze(1).shape)
                            #print(pred_shape)
                            #print(pred_shape.shape)
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
                            pred_joints2d_coco = orthographic_project_torch(pred_joints_all, pred_cam_wp)
                            pred_joints2d_coco = pred_joints2d_coco[:, config.SMPL_TO_KPRCNN_MAP, :]
                            pred_joints2d_coco = undo_keypoint_normalisation(pred_joints2d_coco,
                                                                            proxy_rep_input_wh)

                            rotation = torch.eye(3, device=smpl_joints.device).unsqueeze(0).expand(1,
                                                                                                -1, -1)
                            translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)
                            pred_joints2d = perspective_project_torch(smpl_joints, rotation, translation, None,
                                                                            config.FOCAL_LENGTH, proxy_rep_input_wh)
                            pred_joints2d = pred_joints2d[:, config.SMPL_TO_KPRCNN_MAP, :]

                            # Need to expand cam_ts for NMR i.e.  from (B, 3)
                            # to
                            # (B, 1, 3)
                            translation_nmr = torch.unsqueeze(translation, dim=1)
                            pred_silhouettes = nmr(vertices=pred_smpl_output.vertices,
                                                        faces=faces,
                                                        t=translation_nmr,
                                                        mode='silhouettes')
                            #print(pred_silhouettes)

                            teapot_mesh = Meshes(verts=pred_smpl_output.vertices, faces=faces)
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
                            #print(np.log(1.  / 1e-4 - 1.) *
                            #blend_params.sigma)
                            silhouette_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, 
                                    raster_settings=raster_settings),
                                shader=SoftSilhouetteShader(blend_params=blend_params))
        
                            sil_image = silhouette_renderer(meshes_world=teapot_mesh, T=translation, R=cam_R)
                            sil_image = sil_image[...,3]

                            #silhouette = np.flipud(silhouette)
                            #silhouette = np.fliplr(silhouette).copy()
                            #if not is_train:
                            #    cv2.imwrite('PlayerReconstruction/1{}{}.png'.format(epoch,
                            #    is_train),
                            #    pred_silhouettes.cpu().detach().numpy()[0]*255)
                            #    cv2.imwrite('PlayerReconstruction/2.png',
                            #    silhouette*255)
                            #    print(np.sum(silhouette))
                            #    print(np.sum(np.abs(silhouette -
                            #    pred_silhouettes.cpu().detach().numpy()[0])))
                            #    cv2.imwrite('PlayerReconstruction/3.png',
                            #    np.abs(silhouette -
                            #    pred_silhouettes.cpu().detach().numpy()[0])*255)

                            #for ii in range(sil_image[0].shape[0]):
                            #    print(sil_image[0][ii])

                            pred_dict_for_loss = {'joints2D': pred_joints2d_coco[0],
                                                    'silhouette': pred_silhouettes}
                            target_dict_for_loss = {'joints2D': torch.from_numpy(joints2D[:, :2]).float().to(device),
                                                    'silhouette': torch.from_numpy(silhouette).float().to(device).unsqueeze(0)}

                            #print(pred_joints2d_coco)
                            #print(pred_joints2d)
                            #print(joints2D)

                            # ---------------- BACKWARD PASS ----------------
                            if (is_train):
                                optimiser.zero_grad()
                            loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)
                            if (is_train):
                                #print(loss.item())
                                #print(pred_pose_rotmats[:, 0].unsqueeze(1))
                                #print(pred_shape)
                                loss.backward()
                                optimiser.step()
                                #pred_cam_wp_, pred_pose_, pred_shape_ =
                                #regressor(proxy_rep)
                                #print(pred_pose_rotmats[:, 0].unsqueeze(1))
                                #print(pred_shape_)

                            # ---------------- TRACK LOSS AND METRICS
                            # ----------------
                            num_train_inputs_in_batch = 1
                            if (is_train):
                                metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                                                    pred_dict_for_loss, target_dict_for_loss,
                                                                    num_train_inputs_in_batch)
                            else:
                                metrics_tracker.update_per_batch('val', loss, task_losses_dict,
                                                                    pred_dict_for_loss, target_dict_for_loss,
                                                                    num_train_inputs_in_batch)

                            #cv2.imwrite('PlayerReconstruction/1.png',
                            #silhouette*255)
                            #cv2.imwrite('PlayerReconstruction/2.png',
                            #pred_silhouettes.cpu().detach().numpy()[0]*255)
                            break
                        break
                    break
                break

            #print(is_train)
            if not is_train:
                # ----------------------- UPDATING LOSS AND METRICS HISTORY
                # -----------------------
                metrics_tracker.update_per_epoch()

                # ----------------------------------- SAVING
                # -----------------------------------
                save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                        best_epoch_val_metrics)

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
                #metrics_tracker.initialise_loss_metric_sums()

    print('Training Completed. Best Val Metrics:\n',
          best_epoch_val_metrics)

#train_regressor('shape', False)
#eval_metrics(True)
def init_loss_and_metric(joints=True, silhoutte=True):
    losses_on = []
    metrics_to_track = []
    if (joints):
        losses_on.append('joints2D')
        metrics_to_track.append('joints2D_l2es')
    if (silhoutte):
        losses_on.append('silhouette')
    metrics_to_track.append('silhouette_iou')
    init_loss_weights = {'joints2D': 1.0, 'silhouette': 1000000.0}
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

def eval_metric_strap(save_vis=True, is_refine=False, score_thresh=10.0, game_continue='',
                             image_folder=player_crop_data_folder, proxy_folder=player_recon_proxy_folder,
                             result_folder=player_recon_strap_result_folder,
                             vis_folder=player_recon_strap_result_vis_folder):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder, exist_ok=True)

    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    checkpoint = torch.load(player_recon_check_points_path, map_location=device)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])
    regressor.eval()
    print("Regressor loaded. Weights from:", player_recon_check_points_path)

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

        remake_dir(game_opt)
        remake_dir(game_opt_result)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            scene_opt = os.path.join(game_opt, scene)
            scene_opt_result = os.path.join(game_opt_result, scene)
            remake_dir(scene_opt)
            remake_dir(scene_opt_result)
            players = os.listdir(scene_full)
            for player in players:
                starttime = timeit.default_timer()
                player_full = os.path.join(scene_full, player)
                if (os.path.isfile(player_full)):
                    continue
                player_proxy = os.path.join(scene_proxy, player)
                player_opt = os.path.join(scene_opt, player)
                player_opt_result = os.path.join(scene_opt_result, player)
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

                    proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                        in_wh=proxy_rep_input_wh,
                                                        out_wh=config.REGRESSOR_IMG_WH)
                    proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
                    proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

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
                    body_poses_with_hand_feet = torch.cat([body_poses_without_hands_feet[:, :6, :, :],
                                        body_pose[:, 6:8, :, :],
                                        body_poses_without_hands_feet[:, 6:19, :, :],
                                        body_pose[:, 21:, :, :]],
                                       dim=1)
                    pred_smpl_output = smpl(body_pose=body_poses_with_hand_feet,
                                            global_orient=global_orient,
                                            betas=betas,
                                            pose2rot=False)

                    if (save_vis):
                        rend_img = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                        cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                        cv2.imwrite(opt_full_result, rend_img)

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
                        
                    silh_mean_error_init += silh_error['iou']
                    joint_mean_error_init += joints_error

                    #for metric in save_val_metrics:
                    #    print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                    #        metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                    #                                                    metric,
                    #                                                    metrics_tracker.history['val_' + metric][-1]))

                    np.savez(opt_full, body_pose=best_model_wts[0], global_orient=best_model_wts[1],
                             betas=best_model_wts[2], translation=best_model_wts[3])

                    num_counter += 1
                endtime = timeit.default_timer()
                print('time: {:.3f}'.format(endtime - starttime))
        
    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))

def single_view_optimization(save_vis=True, is_refine=False, score_thresh=10.0, game_continue='',
                             image_folder=player_crop_data_folder, proxy_folder=player_recon_proxy_folder,
                             result_folder=player_recon_single_view_opt_folder,
                             vis_folder=player_recon_single_view_opt_result_folder,
                             mul_folder=player_recon_multi_view_opt_result_folder, ignore_first=True,
                             interation=player_recon_single_view_iteration):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder, exist_ok=True)

    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    checkpoint = torch.load(player_recon_check_points_path, map_location=device)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])
    regressor.eval()
    print("Regressor loaded. Weights from:", player_recon_check_points_path)

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
        game_mult = os.path.join(mul_folder, game)

        if os.path.exists(game_opt) and game != game_continue and not is_refine:
            continue
        if not is_refine:
            remake_dir(game_opt)
            remake_dir(game_opt_result)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            scene_opt = os.path.join(game_opt, scene)
            scene_opt_result = os.path.join(game_opt_result, scene)
            scene_mult = os.path.join(game_mult, scene)
            if not is_refine:
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
                player_mult = os.path.join(scene_mult, player)
                if not is_refine:
                    remake_dir(player_opt)
                    remake_dir(player_opt_result)
                views = os.listdir(player_full)
                if is_refine:
                    with open(os.path.join(player_mult, 'metrics.xml'), 'r') as fs:
                        before = json.load(fs)
                    if before[1] < score_thresh:
                        continue
                for view in views:
                    view_full = os.path.join(player_full, view)
                    print('process {}'.format(view_full))
                    image = cv2.imread(view_full)
                    j2d_full = os.path.join(player_proxy, view).replace('.png', '_j2d.xml')
                    sil_full = os.path.join(player_proxy, view).replace('.png', '_sil.npy')
                    with open(j2d_full, 'r') as fs:
                        joints2D = np.array(json.load(fs))
                    silhouette = np.load(sil_full)
                    #print(joints2D)

                    opt_full = os.path.join(player_opt, view).replace('.png', '.npz')
                    opt_full_result = os.path.join(player_opt_result, view)

                    proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                        in_wh=proxy_rep_input_wh,
                                                        out_wh=config.REGRESSOR_IMG_WH)
                    proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
                    proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

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

def single_view_optimization_test(save_vis=True, is_refine=False, score_thresh=10.0, game_continue='',
                                  postfix='epoch', number=50, save_every=False):
    if not os.path.exists(player_recon_single_view_opt_folder + postfix):
        os.makedirs(player_recon_single_view_opt_folder + postfix, exist_ok=True)
    if not os.path.exists(player_recon_single_view_opt_result_folder + postfix):
        os.makedirs(player_recon_single_view_opt_result_folder + postfix, exist_ok=True)

    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    checkpoint = torch.load(player_recon_check_points_path, map_location=device)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])
    regressor.eval()
    print("Regressor loaded. Weights from:", player_recon_check_points_path)

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
    games = os.listdir(player_crop_data_folder)
    for game in games:
        game_full = os.path.join(player_crop_data_folder, game)
        game_proxy = os.path.join(player_recon_proxy_folder, game)
        game_opt = os.path.join(player_recon_single_view_opt_folder + postfix, game)
        game_opt_result = os.path.join(player_recon_single_view_opt_result_folder + postfix, game)
        game_mult = os.path.join(player_recon_multi_view_opt_result_folder, game)

        if os.path.exists(game_opt) and game != game_continue and not is_refine:
            continue
        if not is_refine:
            remake_dir(game_opt)
            remake_dir(game_opt_result)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            scene_opt = os.path.join(game_opt, scene)
            scene_opt_result = os.path.join(game_opt_result, scene)
            scene_mult = os.path.join(game_mult, scene)
            if not is_refine:
                remake_dir(scene_opt)
                remake_dir(scene_opt_result)
            players = os.listdir(scene_full)
            for player in players:
                starttime = timeit.default_timer()
                player_full = os.path.join(scene_full, player)
                if (player == '1' or os.path.isfile(player_full)):
                    continue
                player_proxy = os.path.join(scene_proxy, player)
                player_opt = os.path.join(scene_opt, player)
                player_opt_result = os.path.join(scene_opt_result, player)
                player_mult = os.path.join(scene_mult, player)
                if not is_refine:
                    remake_dir(player_opt)
                    remake_dir(player_opt_result)
                views = os.listdir(player_full)
                if is_refine:
                    with open(os.path.join(player_mult, 'metrics.xml'), 'r') as fs:
                        before = json.load(fs)
                    if before[1] < score_thresh:
                        continue
                for view in views:
                    view_full = os.path.join(player_full, view)
                    print('process {}'.format(view_full))
                    image = cv2.imread(view_full)
                    j2d_full = os.path.join(player_proxy, view).replace('.png', '_j2d.xml')
                    sil_full = os.path.join(player_proxy, view).replace('.png', '_sil.npy')
                    with open(j2d_full, 'r') as fs:
                        joints2D = np.array(json.load(fs))
                    silhouette = np.load(sil_full)
                    #print(joints2D)

                    opt_full = os.path.join(player_opt, view).replace('.png', '.npz')
                    opt_full_result = os.path.join(player_opt_result, view)

                    proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                        in_wh=proxy_rep_input_wh,
                                                        out_wh=config.REGRESSOR_IMG_WH)
                    proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
                    proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

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

                    criterion, metrics_tracker, save_val_metrics = init_loss_and_metric(True, True)

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
                    params = [global_orient, body_poses_without_hands_feet, pred_cam_wp, betas] + list(criterion.parameters())
                    optimiser = optim.Adam(params, lr=player_recon_train_regressor_learning_rate)

                    for epoch in range(1, 50 + 1):
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
                        if save_model_weights_this_epoch or save_every:
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
                    #break
        #        break
                endtime = timeit.default_timer()
                print('time: {:.3f}'.format(endtime - starttime))
        #    break
        break
        
    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
        print('silh_iou_opt: {}, joint_error_opt: {}'.format(silh_mean_error_opt / num_counter, joint_mean_error_opt / num_counter))

#single_view_optimization()
def multi_view_optimization(save_vis=True, is_refine=False, score_thresh=10.0, game_continue='',
                            image_folder=player_crop_data_folder, proxy_folder=player_recon_proxy_folder,
                            result_folder=player_recon_multi_view_opt_folder, vis_folder=player_recon_multi_view_opt_result_folder,
                            single_folder=player_recon_single_view_opt_folder, ignore_first=True,
                            interation=player_recon_multi_view_iteration):
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

                            if epoch == 1 and i == 0 and not is_train:
                                for metric in save_val_metrics:
                                    print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                        metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                                    metric,
                                                                                    metrics_tracker.history['val_' + metric][-1]))

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
                                if epoch == 1 and i == 0:
                                    for metric in save_val_metrics:
                                        print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                            metrics_tracker.history['train_' + metric][-1] if len(metrics_tracker.history['train_' + metric]) > 0 else 0,
                                                                                        metric,
                                                                                        metrics_tracker.history['val_' + metric][-1]))
                                metrics_tracker.initialise_loss_metric_sums()
                    
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

#single_view_optimization()
#multi_view_optimization()
def broad_view_optimization(save_vis=True, is_refine=False, score_thresh=10.0, game_continue=''):
    if not os.path.exists(player_recon_broad_view_opt_folder):
        os.makedirs(player_recon_broad_view_opt_folder, exist_ok=True)
    if not os.path.exists(player_recon_broad_view_opt_result_folder):
        os.makedirs(player_recon_broad_view_opt_result_folder, exist_ok=True)

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

    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    checkpoint = torch.load(player_recon_check_points_path, map_location=device)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])

    # Starting training loop
    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    silh_mean_error_opt = 0
    joint_mean_error_opt = 0
    games = os.listdir(player_crop_broad_image_folder)
    for game in games:
        starttime = timeit.default_timer()
        game_full = os.path.join(player_crop_broad_image_folder, game)
        game_proxy = os.path.join(player_broad_proxy_folder, game)
        game_init = os.path.join(player_recon_multi_view_opt_folder, game)
        game_opt = os.path.join(player_recon_broad_view_opt_folder, game)
        game_opt_result = os.path.join(player_recon_broad_view_opt_result_folder, game)
        if os.path.exists(game_opt) and game != game_continue and not is_refine:
            continue
        if not is_refine:
            remake_dir(game_opt)
            remake_dir(game_opt_result)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            scene_opt = os.path.join(game_opt, scene)
            scene_init = os.path.join(game_init, scene)
            scene_opt_result = os.path.join(game_opt_result, scene)
            if not is_refine:
                remake_dir(scene_opt)
                remake_dir(scene_opt_result)
            players = os.listdir(scene_full)
            for player in players:
                starttime_player = timeit.default_timer()
                player_full = os.path.join(scene_full, player, 'player.png')
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

                opt_full = os.path.join(player_opt, 'data.npz')
                j2d_full = os.path.join(player_proxy, 'player_j2d.xml')
                sil_full = os.path.join(player_proxy, 'player_sil.npy')
                init_full = os.path.join(player_init, 'data.npz')
                opt_full_result = os.path.join(player_opt_result, 'player.png')
                with open(j2d_full, 'r') as fs:
                    joints2D = np.array(json.load(fs))
                silhouette = np.load(sil_full)
                image = cv2.imread(player_full)

                init_param = np.load(init_full)
                body_pose = init_param['body_pose']
                betas = init_param['betas']
                body_pose = torch.from_numpy(body_pose).float().to(device)
                betas = torch.from_numpy(betas).float().to(device)

                proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                    in_wh=proxy_rep_input_wh,
                                                    out_wh=config.REGRESSOR_IMG_WH)
                proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
                proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

                with torch.no_grad():
                    pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep)

                    if pred_pose.shape[-1] == 24 * 3:
                        pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                        pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                    elif pred_pose.shape[-1] == 24 * 6:
                        pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                    global_orient = pred_pose_rotmats[:, 0].unsqueeze(1)
                    translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

                    pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                            global_orient=global_orient,
                                            betas=pred_shape,
                                            pose2rot=False)

                    rend_img = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                    cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                    cv2.imwrite(opt_full_result.replace('.png', '_0.png'), rend_img)

                criterion, metrics_tracker, save_val_metrics = init_loss_and_metric(True, False)

                # ----------------------- Optimiser -----------------------
                best_epoch_val_metrics = {}
                best_epoch = 1
                best_model_wts = copy.deepcopy([body_pose.cpu().detach().numpy(), 
                                                global_orient.cpu().detach().numpy(), 
                                                betas.cpu().detach().numpy(), 
                                                translation.cpu().detach().numpy()])
                for metric in save_val_metrics:
                    best_epoch_val_metrics[metric] = np.inf

                # optimize camera
                global_orient.requires_grad = True
                pred_cam_wp.requires_grad = True
                params = [pred_cam_wp, global_orient]
                optimiser = optim.Adam(params, lr=player_recon_broad_view_learning_rate)

                #print(pred_cam_wp)
                #print(global_orient)

                for epoch in range(1, player_recon_broad_view_iteration + 1):
                    for is_train in [True, False]:
                        pred_smpl_output = smpl(body_pose=body_pose,
                                                global_orient=global_orient,
                                                betas=betas,
                                                pose2rot=False)

                        if (epoch == 1 and save_vis and is_train):
                            rend_img = wp_renderer.render(verts=pred_smpl_output.vertices.cpu().detach().numpy()[0], 
                                                            cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                            cv2.imwrite(opt_full_result.replace('.png', '_1.png'), rend_img)

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
                        if epoch == 1 and is_train:
                            silh_mean_error_init += silh_error['iou']
                            joint_mean_error_init += joints_error
                            num_counter += 1

                        pred_dict_for_loss = {'joints2D': pred_joints2d_coco[0],
                                                'silhouette': pred_silhouettes}
                        target_dict_for_loss = {'joints2D': torch.from_numpy(joints2D[:,:2]).float().to(device),
                                                'silhouette': torch.from_numpy(silhouette).float().to(device).unsqueeze(0)}

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
                            translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

                        if not is_train:
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
                            metrics_tracker.initialise_loss_metric_sums()
                print('Finished translation optimization.')
                print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                print('Best epoch ', best_epoch)
                #print(pred_cam_wp)
                #print(global_orient)

                if save_vis:
                    best_image = wp_renderer.render(verts=best_smpl.vertices.cpu().detach().numpy()[0], 
                                                cam=pred_cam_wp.cpu().detach().numpy()[0], img=image)
                    cv2.imwrite(opt_full_result.replace('.png', '_2.png'), best_image)

                np.savez(opt_full, body_pose=best_model_wts[0], global_orient=best_model_wts[1],
                             betas=best_model_wts[2], translation=best_model_wts[3])
                silh_mean_error_opt += best_silh_error
                joint_mean_error_opt += best_joints_error
                with open(os.path.join(player_opt_result, 'metrics.xml'), 'w') as fs:
                    fs.write(json.dumps([best_silh_error, best_joints_error]))
                if is_refine:
                    print('After ', best_joints_error)

                endtime_player = timeit.default_timer()
                print('time player: {:.3f}'.format(endtime_player - starttime_player))
        #        break
        #    break
        #break
        endtime = timeit.default_timer()
        print('time: {:.3f}'.format(endtime - starttime))

    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))
        print('silh_iou_opt: {}, joint_error_opt: {}'.format(silh_mean_error_opt / num_counter, joint_mean_error_opt / num_counter))

def calc_initial_metrics(crop_folder=player_crop_data_folder, proxy_folder=player_recon_proxy_folder):
    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    checkpoint = torch.load(player_recon_check_points_path, map_location=device)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])
    regressor.eval()
    print("Regressor loaded. Weights from:", player_recon_check_points_path)

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
    games = os.listdir(crop_folder)
    for game in games:
        game_full = os.path.join(crop_folder, game)
        game_proxy = os.path.join(proxy_folder, game)
        scenes = os.listdir(game_full)
        print('process {}'.format(game_full))
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            players = os.listdir(scene_full)
            for player in players:
                starttime = timeit.default_timer()
                player_full = os.path.join(scene_full, player)
                if (player == '1' or os.path.isfile(player_full)):
                    continue
                player_proxy = os.path.join(scene_proxy, player)
                views = os.listdir(player_full)
                for view in views:
                    view_full = os.path.join(player_full, view)
                    
                    image = cv2.imread(view_full)
                    j2d_full = os.path.join(player_proxy, view).replace('.png', '_j2d.xml')
                    sil_full = os.path.join(player_proxy, view).replace('.png', '_sil.npy')
                    with open(j2d_full, 'r') as fs:
                        joints2D = np.array(json.load(fs))
                    silhouette = np.load(sil_full)
                    #print(joints2D)

                    proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                        in_wh=proxy_rep_input_wh,
                                                        out_wh=config.REGRESSOR_IMG_WH)
                    proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
                    proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

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

                    # ----------------------- Optimiser -----------------------
                    body_poses_without_hands_feet = torch.cat([body_pose[:, :6, :, :],
                                                           body_pose[:, 8:21, :, :]],
                                                          dim=1)

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
                        
                    keypoints = pred_joints2d_coco.cpu().detach().numpy()[0].astype('int32')
                    silh_error, iou_vis = compute_silh_error_metrics(pred_silhouettes.cpu().detach().numpy()[0], silhouette, False)
                    joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D[:,:2])
                        
                    silh_mean_error_init += silh_error['iou']
                    joint_mean_error_init += joints_error

                    num_counter += 1
                endtime = timeit.default_timer()
                print('time: {:.3f}'.format(endtime - starttime))
        
    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))

def calc_initial_metrics_broad():
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

    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)
    regressor.to(device)
    checkpoint = torch.load(player_recon_check_points_path, map_location=device)
    regressor.load_state_dict(checkpoint['best_model_state_dict'])

    # Starting training loop
    silh_mean_error_init = 0
    num_counter = 0
    joint_mean_error_init = 0
    games = os.listdir(player_crop_broad_image_folder)
    for game in games:
        starttime = timeit.default_timer()
        game_full = os.path.join(player_crop_broad_image_folder, game)
        game_proxy = os.path.join(player_broad_proxy_folder, game)
        scenes = os.listdir(game_full)
        print('process {}'.format(game_full))
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            players = os.listdir(scene_full)
            for player in players:
                starttime_player = timeit.default_timer()
                player_full = os.path.join(scene_full, player, 'player.png')
                player_proxy = os.path.join(scene_proxy, player)

                j2d_full = os.path.join(player_proxy, 'player_j2d.xml')
                sil_full = os.path.join(player_proxy, 'player_sil.npy')
                with open(j2d_full, 'r') as fs:
                    joints2D = np.array(json.load(fs))
                silhouette = np.load(sil_full)
                image = cv2.imread(player_full)

                proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                    in_wh=proxy_rep_input_wh,
                                                    out_wh=config.REGRESSOR_IMG_WH)
                proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
                proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

                with torch.no_grad():
                    pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep)

                    if pred_pose.shape[-1] == 24 * 3:
                        pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                        pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                    elif pred_pose.shape[-1] == 24 * 6:
                        pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                    global_orient = pred_pose_rotmats[:, 0].unsqueeze(1)
                    translation = convert_weak_perspective_to_camera_translation_torch(pred_cam_wp, config.FOCAL_LENGTH, proxy_rep_input_wh)

                    pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                            global_orient=global_orient,
                                            betas=pred_shape,
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

                keypoints = pred_joints2d_coco.cpu().detach().numpy()[0].astype('int32')
                silh_error, iou_vis = compute_silh_error_metrics(pred_silhouettes.cpu().detach().numpy()[0], silhouette, False)
                joints_error = compute_j2d_mean_l2_pixel_error(keypoints, joints2D[:,:2])
                silh_mean_error_init += silh_error['iou']
                joint_mean_error_init += joints_error
                num_counter += 1

                endtime_player = timeit.default_timer()
                #print('time player: {:.3f}'.format(endtime_player -
                #starttime_player))
        endtime = timeit.default_timer()
        print('time: {:.3f}'.format(endtime - starttime))

    if (num_counter != 0):
        print('silh_iou_init: {}, joint_error_init: {}'.format(silh_mean_error_init / num_counter, joint_mean_error_init / num_counter))

#create_proxy('densepose', real_images_player, 'Data/Temp1', 'Data/Temp2')
#create_proxy()
#single_view_optimization(is_refine=True)
#multi_view_optimization(is_refine=True)
#create_proxy('densepose',
#player_crop_broad_image_folder,player_broad_proxy_folder,
#player_broad_proxy_vis_folder)
#broad_view_optimization(is_refine=True)
#calc_initial_metrics()
#calc_initial_metrics_broad()
#evaluate_model(False)
#single_view_optimization_test(game_continue='G - Juventus - Santa Clara',
#postfix='epoch')
#single_view_optimization_test(game_continue='G - Juventus - Santa Clara',
#postfix='sil', save_every=True)
#create_proxy('densepose', texture_crop_data_folder, texture_proxy, texture_proxy_vis, ignore_first=False)
#single_view_optimization(image_folder=texture_crop_data_folder,
#proxy_folder=texture_proxy,
#                         result_folder=texture_smpl_single,vis_folder=texture_smpl_single_vis,
#                         ignore_first=False,
#                         interation=texture_single_opt_interation)
#multi_view_optimization(image_folder=texture_crop_data_folder, proxy_folder=texture_proxy, result_folder=texture_smpl_mult,
#                        vis_folder=texture_smpl_mult_vis, single_folder=texture_smpl_single, ignore_first=False,
#                        interation=texture_mult_opt_interation)
#create_proxy('densepose', player_crop_data_folder, player_recon_proxy_folder+'unrefine', player_recon_proxy_vis_folder+'unrefine')
#create_proxy('pointrend', player_crop_broad_image_folder+'Blur', player_broad_proxy_folder+'unrefine', player_broad_proxy_vis_folder+'unrefine')
#create_proxy('pointrend', real_images_player, real_images_player_proxy+'unrefine', real_images_player_proxy_vis+'unrefine')
#create_proxy('pointrend', player_crop_broad_image_folder+'Blur'+'_21_21', player_broad_proxy_folder+'unrefine'+'_21_21', player_broad_proxy_vis_folder+'unrefine'+'_21_21')
#create_proxy('pointrend', player_crop_broad_image_folder, player_broad_proxy_folder+'unrefine'+'_0_0', player_broad_proxy_vis_folder+'unrefine'+'_0_0')
#create_proxy('pointrend', player_crop_broad_image_folder+'Blur'+'_11_11', player_broad_proxy_folder+'unrefine'+'_11_11', player_broad_proxy_vis_folder+'unrefine'+'_11_11')
#create_proxy('pointrend', player_crop_broad_image_folder+'Blur'+'_0_11', player_broad_proxy_folder+'unrefine'+'_0_11', player_broad_proxy_vis_folder+'unrefine'+'_0_11')
#create_proxy('pointrend', player_crop_broad_image_folder+'Blur'+'_11_0', player_broad_proxy_folder+'unrefine'+'_11_0', player_broad_proxy_vis_folder+'unrefine'+'_11_0')

#create_proxy('densepose', player_crop_data_folder, player_recon_proxy_folder+'unrefine', player_recon_proxy_vis_folder+'unrefine')
#predict()
#single_view_optimization()
#multi_view_optimization()
#eval_metric_strap()
#eval_metric_strap(proxy_folder=player_recon_proxy_folder+'unrefine', 
#                  result_folder=player_recon_strap_result_folder+'unrefine',
#                  vis_folder=player_recon_strap_result_vis_folder+'unrefine')
#eval_metric_strap(image_folder = player_crop_broad_image_folder,
#                  proxy_folder = player_broad_proxy_folder,
#                  result_folder=player_crop_broad_image_folder+'_strap',
#                  vis_folder=player_crop_broad_image_folder+'_strap_vis')
#eval_metric_strap(image_folder = player_crop_broad_image_folder,
#                  proxy_folder = player_broad_proxy_folder+'unrefine',
#                  result_folder=player_crop_broad_image_folder+'_strap_unrefine',
#                  vis_folder=player_crop_broad_image_folder+'_strap_vis_unrefine')

#create_proxy('pointrend', player_crop_broad_image_folder, player_broad_proxy_folder+'unrefine', player_broad_proxy_vis_folder+'unrefine')
create_proxy('pointrend', real_images_player, real_images_player_proxy, real_images_player_proxy_vis)