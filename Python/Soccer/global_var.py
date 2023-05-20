import torch
import os
from torchvision import transforms

dataset_folder = r'E:\DataSet\Final'
fifa_folder = r'E:\DataSet\FIFA_test'

default_size = (1080, 1920)
default_size_reverse = (1920, 1080)
default_size_color = (1080, 1920, 3)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
device_text = 'cuda:0'
gpu_index = 0
device = torch.device(device_text)

learning_rate = 2e-4
weight_decay = 1e-5

transform = transforms.Compose([transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(mean=[0.485, 0.456, 0.406],
 std=[0.229, 0.224, 0.225])])

transform_general = transforms.Compose([
    transforms.ToTensor(),
])

proxy_rep_input_wh = 512

classification_frame_per_video = 2
classification_folder = 'Data/ImageClassification'
classification_data_folder = 'Data/ImageClassification/data'
classification_data_folder_train = 'Data/ImageClassification/data_train'
classification_data_folder_eval = 'Data/ImageClassification/data_eval'
classification_data_model_file = 'model.pth'
classification_result_folder = 'Data/ImageClassification/result'

classification_num_epochs = 10

field_detection_data = 'Data/FieldDetection'

player_detection_color = (0, 0, 255)
player_detection_data_folder = 'Data/PlayerDetection'

player_crop_data_folder = 'Data/PlayerCrop'
player_crop_border = 40
player_crop_size = (512, 512)
player_crop_broad_folder = 'Data/PlayerBroad'
player_crop_border_broad = 15
player_crop_broad_vis_folder = 'Data/PlayerBroadVis'
player_crop_broad_image_folder = 'Data/PlayerBroadImage'
player_crop_broad_image_folder_aug = 'Data/PlayerBroadImageAug'

player_recon_check_points_path = 'Data/CheckPoints/straps_model_checkpoint.tar'
player_recon_proxy_folder = 'Data/PlayerProxy'
player_recon_result_folder = 'Data/PlayerRecon'
player_recon_proxy_vis_folder = 'Data/PlayerProxyVis'
player_recon_strap_result_folder = 'Data/strap'
player_recon_strap_result_vis_folder = 'Data/strap_vis'

player_broad_proxy_folder = 'Data/PlayerBroadProxy'
player_broad_proxy_vis_folder = 'Data/PlayerBroadProxyVis'
player_broad_proxy_folder_aug = 'Data/PlayerBroadProxyAug'
player_broad_proxy_vis_folder_aug = 'Data/PlayerBroadProxyVisAug'

player_recon_train_regressor_epoch = 10
player_recon_train_regressor_learning_rate = 0.001
player_recon_train_regressor_checkpoints_folder = 'Data/CheckPoints/ModelTraining'
player_recon_train_regressor_logs_folder = 'Data/Logs'

player_recon_single_view_iteration = 100
player_recon_single_view_opt_folder = 'Data/PlayerSingleViewOpt'
player_recon_single_view_opt_result_folder = 'Data/PlayerSingleViewOptRes'

player_recon_multi_view_iteration = 50
player_recon_multi_view_opt_folder = 'Data/PlayerMultiViewOpt'
player_recon_multi_view_opt_result_folder = 'Data/PlayerMultiViewOptRes'

player_recon_broad_view_iteration = 100
player_recon_broad_view_learning_rate = 0.01
player_recon_broad_view_opt_folder = 'Data/PlayerBroadViewOpt'
player_recon_broad_view_opt_result_folder = 'Data/PlayerBroadViewOptRes'
player_recon_broad_view_opt_folder_aug = 'Data/PlayerBroadViewOptAug'

real_images = 'Data/RealImages'
real_images_box = 'Data/RealPlayer'
real_images_box_vis = 'Data/RealPlayerVis'
real_images_player = 'Data/RealPlayerImage'
real_images_player_proxy = 'Data/RealPlayerProxy'
real_images_player_proxy_vis = 'Data/RealPlayerProxyVis'

texture_crop_data_folder = 'Data/TextureCrop'
player_texture_data_folder = 'Data/TextureImage'
texture_mask_folder = 'Data/TextureMask'
player_texture_iuv_folder = 'Data/TextureIUV'
texture_proxy = 'Data/TextureProxy'
texture_proxy_vis = 'Data/TextureProxyVis'
texture_render = 'Data/TextureRender'
texture_multiview = 'Data/TextureMultiview'
texture_multiview_mask = 'Data/TextureMultiviewMask'
texture_smpl_single = 'Data/TextureSMPLSingle'
texture_smpl_single_vis = 'Data/TextureSMPLSingleVis'
texture_smpl_mult = 'Data/TextureSMPLMult'
texture_smpl_mult_vis = 'Data/TextureSMPLMultVis'
texture_uv_coordinates = 'Data/TextureUVCoordinates'
texture_single_opt_interation = 50
texture_mult_opt_interation = 20
texture_team_fusion='Data/TextureTeamFusion'
