import os
import shutil
import time
import cv2
import numpy as np
import collections
import numpy as np
import json
import random

import matplotlib
#matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from PIL import Image
from torch.autograd import Variable
from xml.dom.minidom import getDOMImplementation

from global_var import *
from scipy.stats import norm

from PlayerReconstruction.utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps

def remake_dir(dir):
    if (os.path.exists(dir)):
        shutil.rmtree(dir)
        time.sleep(1)
    os.makedirs(dir, exist_ok=True)

def preprocess_image(img, trans):
    pil_image = Image.fromarray(img)
    transformed = trans(pil_image)
    transformed = transformed.unsqueeze(0)
    var = Variable(transformed)
    return var

def largest_connected_components(image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape, dtype=np.uint8)
    img2[output == max_label] = 255

    return img2

def generate_annotation(filename, size, category, boxes):
    impl = getDOMImplementation()

    newdoc = impl.createDocument(None, "annotation", None)

    top_element = newdoc.documentElement

    element = newdoc.createElement('folder')
    top_element.appendChild(element)

    element = newdoc.createElement('filename')
    element.appendChild(newdoc.createTextNode(filename))
    top_element.appendChild(element)

    element = newdoc.createElement('database')
    top_element.appendChild(element)
    element = newdoc.createElement('annotation')
    top_element.appendChild(element)
    element = newdoc.createElement('image')
    top_element.appendChild(element)

    element = newdoc.createElement('size')
    sub_element = newdoc.createElement('height')
    sub_element.appendChild(newdoc.createTextNode(str(size[0])))
    element.appendChild(sub_element)
    sub_element = newdoc.createElement('width')
    sub_element.appendChild(newdoc.createTextNode(str(size[1])))
    element.appendChild(sub_element)
    sub_element = newdoc.createElement('depth')
    sub_element.appendChild(newdoc.createTextNode(str(size[2])))
    element.appendChild(sub_element)
    top_element.appendChild(element)

    element = newdoc.createElement('segmented')
    top_element.appendChild(element)

    for box in boxes:
        element = newdoc.createElement('object')
        sub_element = newdoc.createElement('name')
        sub_element.appendChild(newdoc.createTextNode(category))
        element.appendChild(sub_element)
        sub_element = newdoc.createElement('pose')
        element.appendChild(sub_element)
        sub_element = newdoc.createElement('truncated')
        element.appendChild(sub_element)
        sub_element = newdoc.createElement('difficult')
        element.appendChild(sub_element)
        sub_element = newdoc.createElement('bndbox')
        sub_sub_element = newdoc.createElement('xmin')
        sub_sub_element.appendChild(newdoc.createTextNode(str(box[0])))
        sub_element.appendChild(sub_sub_element)
        sub_sub_element = newdoc.createElement('ymin')
        sub_sub_element.appendChild(newdoc.createTextNode(str(box[1])))
        sub_element.appendChild(sub_sub_element)
        sub_sub_element = newdoc.createElement('xmax')
        sub_sub_element.appendChild(newdoc.createTextNode(str(box[2])))
        sub_element.appendChild(sub_sub_element)
        sub_sub_element = newdoc.createElement('ymax')
        sub_sub_element.appendChild(newdoc.createTextNode(str(box[3])))
        sub_element.appendChild(sub_sub_element)
        element.appendChild(sub_element)
        top_element.appendChild(element)

    return newdoc.childNodes[0].toprettyxml()

#print(generate_annotation('5b5c78a3-84fa-4ccc-923a-6075b01bd84e.bmp', (1080,
#1920, 3), 'people', [[1,2,3,4]]))
def get_center_object(boxes, labels, width, height):
    if (len(boxes) == 0):
        return None, None

    distance = float('inf')
    width = width // 2
    height = height * 2 // 3
    index = 0
    for i in range(len(boxes)):
        box = boxes[i]
        x_center = (boxes[i][0] + boxes[i][2]) // 2
        current = abs(x_center - width)
        if (current < distance and boxes[i][3] > height and (boxes[i][3] - boxes[i][1] > 150)):
            distance = current
            index = i
    return boxes[index], labels[index]

def view_images(folder, name=None, binary=False):
    queue = collections.deque()
    queue.append(folder)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    exit = False
    while queue:
        node = queue.popleft()
        files = os.listdir(node)
        files.sort()
        for file in files:
            file_full = os.path.join(node, file)
            if (os.path.isdir(file_full)):
                queue.append(file_full)
            else:
                if (name is not None and name not in file):
                    continue
                if ('.png' not in file):
                    continue
                image = cv2.imread(file_full)
                
                fontScale = 0.3
                fontColor = (130, 130, 200)
                y0, dy = 20, 20
                for i, line in enumerate(file_full.split('\\')):
                    y = y0 + i*dy
                    cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, fontColor)
                if binary:
                    image = image * 255
                cv2.imshow('frame', image)
                if cv2.waitKey(0) & 0xFF == 27:
                    exit = True
                    break
        if (exit):
            break

def view_images_score(folder, name=None, binary=False, score = 8):
    queue = collections.deque()
    queue.append(folder)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    exit = False
    while queue:
        node = queue.popleft()
        files = os.listdir(node)
        files.sort()
        if 'metrics.xml' in files:
            file_full = os.path.join(node, 'metrics.xml')
            with open(file_full, 'r') as fs:
                scores = np.array(json.load(fs))
            if scores[1] > score:
                continue
        for file in files:
            file_full = os.path.join(node, file)
            if (os.path.isdir(file_full)):
                queue.append(file_full)
            else:
                if (name is not None and name not in file):
                    continue
                if ('.png' not in file):
                    continue
                image = cv2.imread(file_full)
                
                fontScale = 0.3
                fontColor = (130, 130, 200)
                y0, dy = 20, 20
                for i, line in enumerate(file_full.split('\\')):
                    y = y0 + i*dy
                    cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, fontColor)
                if binary:
                    image = image * 255
                cv2.imshow('frame', image)
                if cv2.waitKey(0) & 0xFF == 27:
                    exit = True
                    break
        if (exit):
            break

def compare_images(folders, name=None):
    queue = collections.deque()
    queue.append(folders[0])
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    exit = False
    while queue:
        node = queue.popleft()
        files = os.listdir(node)
        random.shuffle(files)
        for file in files:
            file_full = os.path.join(node, file)
            if (os.path.isdir(file_full)):
                queue.append(file_full)
            else:
                if (name is not None and name not in file):
                    continue
                image1 = cv2.imread(file_full)

                fontScale = 0.3
                fontColor = (130, 130, 200)
                y0, dy = 20, 20
                for i, line in enumerate(file_full.split('\\')):
                    if i < 1:
                        continue
                    y = y0 + i*dy
                    cv2.putText(image1, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, fontColor)
                images = [image1]
                for i in range(1, len(folders)):
                    file_full2 = file_full.replace(folders[0], folders[i])
                    if (os.path.exists(file_full2)):
                        image2 = cv2.imread(file_full2)
                        images.append(image2)
                
                images = np.hstack(images)
                cv2.imshow('frame', images)
                if cv2.waitKey(0) & 0xFF == 27:
                    exit = True
                    break
        if (exit):
            break

def crop_image(image, box, border):
    left = box[0] - border
    top = box[1] - border
    right = box[2] + border
    bottom = box[3] + border
    if (left < 0):
        left = 0
    if (top < 0):
        top = 0
    if (right > image.shape[1]):
        right = image.shape[1]
    if (bottom > image.shape[0]):
        bottom = image.shape[0]
    croped = image[top:bottom, left:right]

    size = max(croped.shape[0], croped.shape[1])
    if (size == croped.shape[0]):
        diff = size - croped.shape[1]
        pad_top = 0
        pad_bottom = 0
        pad_left = diff // 2
        pad_right = diff // 2
        if (diff % 2 != 0):
            pad_left += 1
    else:
        diff = size - croped.shape[0]
        pad_left = 0
        pad_right = 0
        pad_top = diff // 2
        pad_bottom = diff // 2
        if (diff % 2 != 0):
            pad_top += 1
    ans = cv2.copyMakeBorder(croped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value = (53,88,83))
    return ans

def test_norm():
    mu, std = norm.fit([1,2,3,4,5])
    print(mu)
    print(std)

def calc_metrics(folder):
    games = os.listdir(folder)
    ans = None
    count = 0
    for game in games:
        game_full = os.path.join(folder, game)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            players = os.listdir(scene_full)
            for player in players:
                player_full = os.path.join(scene_full, player)
                file_full = os.path.join(player_full, 'metrics.xml')
                with open(file_full, 'r') as fs:
                    scores = np.array(json.load(fs))
                count += 1
                if ans is None:
                    ans = scores
                else:
                    ans += scores
    ans /= count
    print(ans)

def count_players(folder):
    games = os.listdir(folder)
    count = 0
    for game in games:
        game_full = os.path.join(folder, game)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            players = os.listdir(scene_full)
            for player in players:
                count += 1
    print(count)

def count_players_need_refine(folder, threshold):
    games = os.listdir(folder)
    count = 0
    for game in games:
        game_full = os.path.join(folder, game)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            players = os.listdir(scene_full)
            for player in players:
                player_opt_result = os.path.join(scene_full, player)
                with open(os.path.join(player_opt_result, 'metrics.xml'), 'r') as fs:
                        before = json.load(fs)
                if before[1] >= threshold:
                    count += 1
    print(count)

def random_select_train_set(num):
    games = os.listdir(player_crop_broad_folder)
    count = 0
    train_set = set()
    length = len(games)
    while count < num:
        index = random.randint(0, length-1)
        if games[index] not in train_set:
            train_set.add(games[index])
            count += 1
    filename = 'Data/train_set.xml'
    with open(filename, 'w') as fs:
        fs.write(json.dumps(list(train_set)))

def apply_colormap(image, vmin=None, vmax=None, cmap='viridis', cmap_seed=1):
    """
    Apply a matplotlib colormap to an image.

    This method will preserve the exact image size. `cmap` can be either a
    matplotlib colormap name, a discrete number, or a colormap instance. If it
    is a number, a discrete colormap will be generated based on the HSV
    colorspace. The permutation of colors is random and can be controlled with
    the `cmap_seed`. The state of the RNG is preserved.
    """
    image = image.astype("float64")  # Returns a copy.
    # Normalization.
    if vmin is not None:
        imin = float(vmin)
        image = np.clip(image, vmin, sys.float_info.max)
    else:
        imin = np.min(image)
    if vmax is not None:
        imax = float(vmax)
        image = np.clip(image, -sys.float_info.max, vmax)
    else:
        imax = np.max(image)
    image -= imin
    image /= (imax - imin)
    # Visualization.
    cmap_ = plt.get_cmap(cmap)
    vis = cmap_(image, bytes=True)
    return vis

def convert_proxy_vis(folder):
    j2d_full = os.path.join(folder, 'player_j2d.xml')
    sil_full = os.path.join(folder, 'player_sil.npy')
    with open(j2d_full, 'r') as fs:
        joints2D = np.array(json.load(fs))
    silhouette = np.load(sil_full)
    silhouette_vis = apply_colormap(silhouette)
    cv2.imwrite(os.path.join(folder, 'sil.png'), silhouette_vis)

    joints2D = joints2D[:, :2]
    heatmaps = convert_2Djoints_to_gaussian_heatmaps(joints2D.astype(np.int16),
                                                     512)
    for i in range(heatmaps.shape[2]):
        cv2.imwrite(os.path.join(folder, '{}.png'.format(str(i))), apply_colormap(heatmaps[:,:,i]))

def add_motion_blur(folder, dst, motion_size = 21, out_of_focus_size = 21):
    # generating the kernel
    if (motion_size > 0):
        kernel_motion_blur = np.zeros((motion_size, motion_size))
        kernel_motion_blur[(motion_size-1)//2, :] = np.ones(motion_size)
        kernel_motion_blur = kernel_motion_blur / motion_size

    games = os.listdir(folder)
    remake_dir(dst)
    for game in games:
        game_full = os.path.join(folder, game)
        game_dst = os.path.join(dst, game)
        remake_dir(game_dst)
        scenes = os.listdir(game_full)
        for scene in scenes:
            scene_full = os.path.join(game_full, scene)
            scene_dst = os.path.join(game_dst, scene)
            remake_dir(scene_dst)
            players = os.listdir(scene_full)
            for player in players:
                player_full = os.path.join(scene_full, player)
                player_dst = os.path.join(scene_dst, player)
                remake_dir(player_dst)
                image = cv2.imread(os.path.join(player_full, 'player.png'))
                if (motion_size > 0):
                    output = cv2.filter2D(image, -1, kernel_motion_blur)
                else:
                    output = image
                if (out_of_focus_size > 0):
                    blur = cv2.GaussianBlur(output,(out_of_focus_size,out_of_focus_size),0)
                else:
                    blur = output
                cv2.imwrite(os.path.join(player_dst, 'player.png'), blur)

def recreate_proxy_vis_broad(image_folder, proxy_folder, vis_folder):
    remake_dir(vis_folder)

    games = os.listdir(proxy_folder)
    for game in games:
        game_image = os.path.join(image_folder, game)
        game_proxy = os.path.join(proxy_folder, game)
        game_vis = os.path.join(vis_folder, game)
        remake_dir(game_vis)

        scenes = os.listdir(game_proxy)
        for scene in scenes:
            scene_image = os.path.join(game_image, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            scene_vis = os.path.join(game_vis, scene)
            remake_dir(scene_vis)

            print(scene)

            players = os.listdir(scene_proxy)
            for player in players:
                player_image = os.path.join(scene_image, player)
                player_proxy = os.path.join(scene_proxy, player)
                player_vis = os.path.join(scene_vis, player)
                remake_dir(player_vis)

                image = cv2.imread(os.path.join(player_image, 'player.png'))
                j2d_full = os.path.join(player_proxy, 'player_j2d.xml')
                sil_full = os.path.join(player_proxy, 'player_sil.npy')
                with open(j2d_full, 'r') as fs:
                    joints2D = np.array(json.load(fs))
                silhouette = np.load(sil_full)

                silhouette_vis = cv2.addWeighted(image.astype(silhouette.dtype), 0.7,
                              128 * np.tile(silhouette[:, :, None], [1, 1, 3]),
                              0.3, gamma=0)
                for j in range(joints2D.shape[0]):
                    cv2.circle(silhouette_vis, (int(joints2D[j, 0]), int(joints2D[j, 1])), 5, (0, 255, 0), -1)
                cv2.imwrite(os.path.join(player_vis, 'player_silhouette.png'), silhouette_vis)

def recreate_proxy_vis(image_folder, proxy_folder, vis_folder):
    remake_dir(vis_folder)

    games = os.listdir(proxy_folder)
    for game in games:
        game_image = os.path.join(image_folder, game)
        game_proxy = os.path.join(proxy_folder, game)
        game_vis = os.path.join(vis_folder, game)
        remake_dir(game_vis)

        scenes = os.listdir(game_proxy)
        for scene in scenes:
            scene_image = os.path.join(game_image, scene)
            scene_proxy = os.path.join(game_proxy, scene)
            scene_vis = os.path.join(game_vis, scene)
            remake_dir(scene_vis)

            print(scene)

            players = os.listdir(scene_proxy)
            for player in players:
                player_image = os.path.join(scene_image, player)
                player_proxy = os.path.join(scene_proxy, player)
                player_vis = os.path.join(scene_vis, player)
                remake_dir(player_vis)

                for i in range(1, 5):
                    image_file = os.path.join(player_image, 'view_{}.png'.format(i))
                    j2d_full = os.path.join(player_proxy, 'view_{}_j2d.xml'.format(i))
                    sil_full = os.path.join(player_proxy, 'view_{}_sil.npy'.format(i))
                    if not os.path.exists(image_file) or not os.path.exists(j2d_full) or not os.path.exists(sil_full):
                        continue
                    image = cv2.imread(image_file)
                    with open(j2d_full, 'r') as fs:
                        joints2D = np.array(json.load(fs))
                    silhouette = np.load(sil_full)

                    silhouette_vis = cv2.addWeighted(image.astype(np.float64), 1,
                                  128 * np.tile(silhouette[:, :, None], [1, 1, 3]),
                                  0, gamma=0)
                    for j in range(joints2D.shape[0]):
                        cv2.circle(silhouette_vis, (int(joints2D[j, 0]), int(joints2D[j, 1])), 5, (0, 255, 0), -1)
                    cv2.imwrite(os.path.join(player_vis, 'view_{}_silhouette.png'.format(i)), silhouette_vis)

# some image been remove, need delete the proxy files
def delete_files():
    image_folder = "Data/RealPlayerImage/real"
    proxy_folder = "Data/RealPlayerProxyunrefine/real"

    scenes = os.listdir(proxy_folder)
    for scene in scenes:
        scene_image = os.path.join(image_folder, scene)
        scene_proxy = os.path.join(proxy_folder, scene)

        if (not os.path.exists(scene_image)):
            print(scene_image)
            shutil.rmtree(scene_proxy)
            time.sleep(1)
            continue

        players = os.listdir(scene_proxy)
        for player in players:
            player_image = os.path.join(scene_image, player)
            player_proxy = os.path.join(scene_proxy, player)

            if (not os.path.exists(player_image)):
                print(player_image)
                shutil.rmtree(player_proxy)
                time.sleep(1)


#calc_metrics(player_recon_broad_view_opt_result_folder)
#view_images(player_recon_result_folder, 'rend.png')
#view_images(player_crop_broad_image_folder_aug)
#view_images(player_recon_broad_view_opt_result_folder, '_2')
#compare_images(['Data/real_ours', 'Data/real_ours_'])
#view_images(player_recon_broad_view_opt_result_folder)
#count_players(player_recon_multi_view_opt_result_folder)
#count_players_need_refine(player_recon_broad_view_opt_result_folder, 10)
#random_select_train_set(30)
#view_images('E:\Code\Soccer\Data\PlayerBroadViewOptRes\D - Ajax - Liverpool')
#convert_proxy_vis('Data/PlayerBroadProxy/D - Ajax - Liverpool/5/20')
#view_images('Data/RealPlayerProxyVis')
#count_players(real_images_player_proxy_vis)
#view_images('Data/RealPlayerImageHmrVis')
#count_players(real_images_player_proxy)
#compare_images(['Data/real_ours_opt_global','Data/RealPlayerImage','Data/real_STR_opt_global','Data/real_hmr_opt_global','Data/real_spin_opt'])
#compare_images(['Data/aug_relate_opt','Data/PlayerBroadImage', 'Data/STA','Data/hmr','Data/spin'])
#view_images('Data/PlayerMultiViewOptRes/C - FC Porto - OM')
#view_images('Data/Temp2')
#view_images(texture_smpl_mult_vis)
#compare_images(['Data/RealPlayerImage', 'Data/TextureIUVReal'])
#view_images(real_images)
#add_motion_blur(player_crop_broad_image_folder, player_crop_broad_image_folder + 'Blur_21_21')
#compare_images([player_crop_broad_image_folder, player_crop_broad_image_folder + 'Blur'])
#compare_images([player_crop_broad_image_folder, player_texture_iuv_folder + 'Broad'])
#count_players(player_recon_data_folder)
#count_players(player_crop_broad_image_folder)

recreate_proxy_vis(player_crop_data_folder, player_recon_proxy_folder, player_recon_proxy_vis_folder)
#view_images('Data/PlayerCrop_spin_multi_multi_vis/A - Atletico de Madrid - FC Bayern', '_2_2')
recreate_proxy_vis(player_crop_data_folder, player_recon_proxy_folder+'unrefine', player_recon_proxy_vis_folder+'unrefine')

#recreate_proxy_vis_broad(player_crop_broad_image_folder, player_broad_proxy_folder, player_broad_proxy_vis_folder)
#recreate_proxy_vis_broad(player_crop_broad_image_folder, player_broad_proxy_folder+'unrefine', player_broad_proxy_vis_folder+'unrefine')

#compare_images(['Data/PlayerProxyVis/A - Lokomotiv Moscow - RB Salzburg', 'Data/PlayerProxyVisunrefine/A - Lokomotiv Moscow - RB Salzburg'])

#view_images('Data/PlayerCrop_oneview_vis', '_1.png')
#view_images_score('Data\PlayerCrop_pare_multi_multi_vis\A - Atletico de Madrid - FC Bayern', '2_2.png')
#count_players(player_crop_broad_image_folder + '_pare')
#view_images('Data/RealPlayerImage')

#delete_files()
#count_players(real_images_player);