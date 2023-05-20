import cv2
import numpy as np
import torchvision
import torch
import os
import sys
import timeit
import json

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from global_var import *
from global_utils import *

def predict(image, model, detection_threshold):
    starttime = timeit.default_timer()
    outputs = model(image) # get the predictions on the image
    # print the results individually
                              # print(f"BOXES: {outputs[0]['boxes']}")
                              # print(f"LABELS: {outputs[0]['labels']}")
                              # print(f"SCORES: {outputs[0]['scores']}")
                              # get all the predicited class names
    pred_labels = outputs[0]['labels'].detach().cpu().numpy()
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    labels = pred_labels[pred_scores >= detection_threshold].astype(np.int32)
    boxes = boxes[(labels == 1)]
    labels = labels[(labels == 1)]

    endtime = timeit.default_timer()
    #print('time: {:.3f}'.format(endtime - starttime))
    return boxes, labels

def test_single_image():
    folder = os.path.join(player_detection_data_folder, 'Data')
    files = os.listdir(folder)
    result_folder = os.path.join(player_detection_data_folder, 'Result')
    remake_dir(result_folder)
    annotation_folder = os.path.join(player_detection_data_folder, 'Annotation')
    remake_dir(annotation_folder)
    for file in files:
        filename_full = os.path.join(folder, file)
        image = cv2.imread(filename_full)
        image_tensor = preprocess_image(image, transform_general).to(device)
        filename = os.path.basename(filename_full)

        orig_filename = os.path.join(result_folder, filename)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, max_size = 1920)
        model.eval().to(device)
        boxes, labels = predict(image_tensor, model, 0.7)

        annotation_filename = os.path.join(annotation_folder, file.replace('.png', '.xml'))
        file_content = generate_annotation(file, default_size_color, 'people', boxes)
        with open(annotation_filename, "w") as fp:
            fp.write(file_content)

        newImg = 255 * np.ones(image.shape, image.dtype)
        newImg[:] = [255,255,255]
        for i, box in enumerate(boxes):
            cv2.rectangle(image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                player_detection_color, 2)
            cv2.putText(image, 'player' if labels[i] == 1 else 'ball', (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, player_detection_color, 2, 
                        lineType=cv2.LINE_AA)
            cv2.rectangle(newImg,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                player_detection_color, 4)

        cv2.imwrite(orig_filename.replace('.png', '_result.png'), image)
        cv2.imwrite(orig_filename.replace('.png', '_geo.png'), newImg)

#test_single_image()

def crop_player(folder = player_crop_data_folder, check_board = True, check_index = True, save_mid=False, res = player_crop_size):
    games = os.listdir(fifa_folder)
    remake_dir(folder)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, max_size = 1920)
    model.eval().to(device)
    min_width, min_height = float('inf'), float('inf')
    max_width, max_height = 0, 0
    for game in games:
        game_full = os.path.join(fifa_folder, game)
        game_broad = os.path.join(player_crop_broad_image_folder, game)
        game_dst = os.path.join(folder, game)
        remake_dir(game_dst)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_broad = os.path.join(game_broad, scene)
            scene_dst = os.path.join(game_dst, scene)
            if not check_index and scene != '1':
                continue
            remake_dir(scene_dst)
            players = os.listdir(scene_full)
            for player in players:
                player_full = os.path.join(scene_full, player)
                player_broad = os.path.join(scene_broad, player)
                if (os.path.isfile(player_full)):
                    continue
                if (check_index):
                    if (player == '1'):
                        continue
                if check_board:
                    if (os.path.exists(player_broad)):
                        continue
                print('process {}'.format(player_full))
                player_dst = os.path.join(scene_dst, player)
                remake_dir(player_dst)
                views = os.listdir(player_full)
                for view in views:
                    view_full = os.path.join(player_full, view)
                    image = cv2.imread(view_full)
                    image_tensor = preprocess_image(image, transform_general).to(device)
                    boxes, labels = predict(image_tensor, model, 0.7)
                    box, label = get_center_object(boxes, labels, image.shape[1], image.shape[0])
                    croped_image = crop_image(image, box, player_crop_border)
                    croped_image = cv2.resize(croped_image, res,
                               interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(player_dst, view), croped_image)
                    cv2.rectangle(image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        player_detection_color, 2)
                    if (save_mid):
                        cv2.imwrite(os.path.join(player_dst, view).replace('.png', '_rect.png'), image)
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    if (width < min_width):
                        min_width = width
                    if (width > max_width):
                        max_width = width
                    if (height < min_height):
                        min_height = height
                    if (height > max_height):
                        max_height = height
                    if height < 200:
                        print(height)
    print('min_width: {}, max_width: {}, min_height: {}, max_height: {}'.format(min_width, max_width, min_height, max_height))

#crop_player()
crop_player(texture_crop_data_folder, False, False, res=(512,512))

def crop_broad_player(save_mid=False, source=fifa_folder, box_folder=player_crop_broad_folder,
                      vis_folder = player_crop_broad_vis_folder):
    games = os.listdir(source)
    remake_dir(box_folder)
    remake_dir(vis_folder)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, max_size = 1920)
    model.eval().to(device)
    min_width, min_height = float('inf'), float('inf')
    max_width, max_height = 0, 0
    for game in games:
        game_full = os.path.join(source, game)
        game_dst = os.path.join(box_folder, game)
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
            broad_file = 'broad.png'
            broad_full = os.path.join(scene_full, broad_file)
            
            image = cv2.imread(broad_full)
            image_vis = np.copy(image)
            image_tensor = preprocess_image(image, transform_general).to(device)
            boxes, labels = predict(image_tensor, model, 0.7)
            for i, (box, label) in enumerate(zip(boxes, labels)):
                croped_image = crop_image(image, box, player_crop_border_broad)
                croped_image = cv2.resize(croped_image, player_crop_size,
                            interpolation=cv2.INTER_LINEAR)
                #cv2.imwrite(os.path.join(scene_dst, str(i) + '.png'), croped_image)
                cv2.rectangle(image_vis,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    player_detection_color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                fontColor = (0, 0, 255)
                cv2.putText(image_vis, str(i), (int(box[0]), int(box[1]) - 5),
                                         font, fontScale, fontColor, lineType=2)
                width = box[2] - box[0]
                height = box[3] - box[1]
                if (width < min_width):
                    min_width = width
                if (width > max_width):
                    max_width = width
                if (height < min_height):
                    min_height = height
                if (height > max_height):
                    max_height = height
            cv2.imwrite(os.path.join(scene_vis, broad_file), image_vis)
            with open(os.path.join(scene_dst, 'boxes.xml'), 'w') as fs:
                fs.write(json.dumps(boxes.tolist()))
    print('min_width: {}, max_width: {}, min_height: {}, max_height: {}'.format(min_width, max_width, min_height, max_height))

#crop_broad_player()
#crop_broad_player(False, real_images, real_images_box, real_images_box_vis)

def crop_broad_player_images(folder = player_crop_broad_folder, broad_folder = fifa_folder,
                             image_folder = player_crop_broad_image_folder):
    games = os.listdir(folder)
    remake_dir(image_folder)
    for game in games:
        game_full = os.path.join(folder, game)
        game_image = os.path.join(broad_folder, game)
        game_data = os.path.join(folder, game)
        game_broad_image = os.path.join(image_folder, game)
        remake_dir(game_broad_image)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_image = os.path.join(game_image, scene)
            scene_data = os.path.join(game_data, scene)
            scene_broad_image = os.path.join(game_broad_image, scene)
            remake_dir(scene_broad_image)

            broad_image_full = os.path.join(scene_image, 'broad.png')
            boxes_full = os.path.join(scene_data, 'boxes.xml')
            index_full = os.path.join(scene_data, 'index.xml')

            image = cv2.imread(broad_image_full)
            with open(boxes_full, 'r') as fs:
                boxes = json.load(fs)
            if os.path.exists(index_full):
                with open(index_full, 'r') as fs:
                    indexes = json.load(fs)
            else:
                indexes = [i for i in range(2, len(boxes) + 2)]
            for (box, index) in zip(boxes, indexes):
                player_full = os.path.join(scene_broad_image, str(index))
                remake_dir(player_full)
                player_full = os.path.join(player_full, 'player.png')
                croped_image = crop_image(image, box, 0)
                croped_image = cv2.resize(croped_image, player_crop_size,
                            interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(player_full, croped_image)

#crop_broad_player_images()
#crop_broad_player_images(real_images_box, real_images, real_images_player)

def move_real_data():
    folder = 'Data/ImageClassification/result'
    remake_dir(real_images)
    game_dst = os.path.join(real_images, 'real')
    remake_dir(game_dst)
    scenes = os.listdir(folder)
    count = 1
    for scene in scenes:
        scene_full = os.path.join(folder, scene)
        scene_dst = os.path.join(game_dst, str(count))
        remake_dir(scene_dst)
        image = cv2.imread(scene_full)
        cv2.imwrite(os.path.join(scene_dst, 'broad.png'), image)
        count+=1
#move_real_data()
