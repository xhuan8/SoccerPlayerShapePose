
import cv2
import os
import shutil
import random
import time
import random
import sys
import timeit
import traceback
import uuid

import torch
import torchvision
from torchvision import transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
from torch import nn

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from global_var import *
from global_utils import *
from classifi_model import ClassifyNet

'''
prepare image for classification
'''
def prepare_data():
    leagues = os.listdir(dataset_folder)
    remake_dir(classification_data_folder)
    one = os.path.join(classification_data_folder, '1')
    os.mkdir(one)
    two = os.path.join(classification_data_folder, '2')
    os.mkdir(two)

    count = 0
    for league in leagues:
        league_full = os.path.join(dataset_folder, league)
        seasons = os.listdir(league_full)
        for season in seasons:
            season_full = os.path.join(league_full, season)
            games = os.listdir(season_full)
            for game in games:
                game_full = os.path.join(season_full, game)
                for file in ['1_HQ.mkv', '2_HQ.mkv']:
                    count += 1
                    print(count)
                    file_full = os.path.join(game_full, file)
                    print(file_full)
                    videoCapture = cv2.VideoCapture(file_full)
                    length = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
                    for i in range(classification_frame_per_video):
                        try:
                            frame_number = random.randint(0, length - 1)
                            videoCapture.set(1, frame_number)
                            ret, frame = videoCapture.read()
                            if (frame.shape[0] != default_size[0] or frame.shape[1] != default_size[1]):
                                frame = cv2.resize(frame, default_size_reverse, interpolation = cv2.INTER_AREA)
            
                            frame_name = str(uuid.uuid4()) + '.png'
                            cv2.namedWindow ('frame', cv2.WINDOW_NORMAL)
                            cv2.setWindowProperty ('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            cv2.imshow('frame', frame)
                            if cv2.waitKey(0) & 0xFF == ord('1'):
                                cv2.imwrite(os.path.join(one, frame_name), frame)
                            else:
                                cv2.imwrite(os.path.join(two, frame_name), frame)
                            cv2.destroyAllWindows()
                        except:
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            print("Unexpected error:", exc_type, exc_value)
                            traceback.print_tb(exc_traceback, file=sys.stdout)
                    videoCapture.release()

def split_train_eval():
    one = os.path.join(classification_data_folder, '1')
    two = os.path.join(classification_data_folder, '2')

    remake_dir(classification_data_folder_train)
    one_train = os.path.join(classification_data_folder_train, '1')
    os.mkdir(one_train)
    two_train = os.path.join(classification_data_folder_train, '2')
    os.mkdir(two_train)

    remake_dir(classification_data_folder_eval)
    one_eval = os.path.join(classification_data_folder_eval, '1')
    os.mkdir(one_eval)
    two_eval = os.path.join(classification_data_folder_eval, '2')
    os.mkdir(two_eval)

    files = os.listdir(one)
    random.shuffle(files)
    for i in range(len(files)//2):
        source = os.path.join(one, files[i])
        target = os.path.join(one_train, files[i])
        shutil.copyfile(source, target)
    for i in range(len(files)//2, len(files)):
        source = os.path.join(one, files[i])
        target = os.path.join(one_eval, files[i])
        shutil.copyfile(source, target)

    files = os.listdir(two)
    random.shuffle(files)
    for i in range(len(files)//2):
        source = os.path.join(two, files[i])
        target = os.path.join(two_train, files[i])
        shutil.copyfile(source, target)
    for i in range(len(files)//2, len(files)):
        source = os.path.join(two, files[i])
        target = os.path.join(two_eval, files[i])
        shutil.copyfile(source, target)


def train(pretrained=True, print_log=True):
    dataset = datasets.ImageFolder(classification_data_folder_train, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1,shuffle=True)

    dataset_eval = datasets.ImageFolder(classification_data_folder_eval, transform=transform)
    dataloader_eval = DataLoader(dataset_eval, batch_size=1,shuffle=False)

    os.makedirs(classification_folder, exist_ok=True)
    model_file = os.path.join(classification_folder, classification_data_model_file)

    model = ClassifyNet().to(device)
    if (pretrained and os.path.exists(model_file)):
        model.load_state_dict(torch.load(model_file))
    resnet18 = models.resnet18(pretrained=True).to(device)
    resnet18.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(classification_num_epochs):
        actual_loss = 0
        actual_loss_eval = 0
        starttime = timeit.default_timer()
        for data, labels in dataloader:
            try:
                data = data.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    resnet_output = resnet18(data)

                output = model(resnet_output)
                loss = criterion(output, labels)
                actual_value = loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                actual_loss += actual_value
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("Unexpected error:", exc_type, exc_value)
                traceback.print_tb(exc_traceback, file=sys.stdout)

        for data, labels in dataloader_eval:
            try:
                data = data.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    resnet_output = resnet18(data)
                    output = model(resnet_output)
                loss = criterion(output, labels)
                actual_value = loss.item()
                actual_loss_eval += actual_value
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("Unexpected error:", exc_type, exc_value)
                traceback.print_tb(exc_traceback, file=sys.stdout)

        endtime = timeit.default_timer()
        if (print_log):
            print('epoch [{}/{}], loss train:{:.4f}, loss test:{:.4f}, time:{:.3f}'.format(
                epoch + 1, classification_num_epochs, actual_loss, actual_loss_eval, (endtime - starttime)))
        torch.save(model.state_dict(), model_file)

class Classifier:
    def __init__(self):
        model_file = os.path.join(classification_folder, classification_data_model_file)

        self.model = ClassifyNet().to(device)
        self.model.eval()
        if (os.path.exists(model_file)):
            self.model.load_state_dict(torch.load(model_file))
        self.resnet18 = models.resnet18(pretrained=True).to(device)
        self.resnet18.eval()

    '''
    Checks if an image is captured from the main camera.
    Returns True if it is from the main camera.
    '''
    def process(self, image):
        data = preprocess_image(image, transform)
        data = data.to(device)
        with torch.no_grad():
            resnet_output = self.resnet18(data)
            output = self.model(resnet_output)
        _, predicted = torch.max(output.data, 1)
        return predicted.detach().cpu().numpy().tolist()[0] == 0

def test_classifier():
    image = cv2.imread(os.path.join(classification_data_folder, '2/00db464e-da82-4391-9ec6-b31ca56d630c.bmp'))
    instance = Classifier()
    predicted = instance.process(image)
    print(predicted)

def predict_with_classifier():
    leagues = os.listdir(dataset_folder)
    remake_dir(classification_result_folder)

    count = 0
    number_per_video = 5
    instance = Classifier()
    for league in leagues:
        league_full = os.path.join(dataset_folder, league)
        seasons = os.listdir(league_full)
        for season in seasons:
            season_full = os.path.join(league_full, season)
            games = os.listdir(season_full)
            for game in games:
                game_full = os.path.join(season_full, game)
                for file in ['1_HQ.mkv', '2_HQ.mkv']:
                    file_full = os.path.join(game_full, file)
                    
                    videoCapture = cv2.VideoCapture(file_full)
                    length = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
                    current = 0
                    for i in range(50):
                        try:
                            frame_number = random.randint(0, length - 1)
                            videoCapture.set(1, frame_number)
                            ret, frame = videoCapture.read()
                            if i == 0:
                                if frame.shape[1] != 1920:
                                    #print('ignore')
                                    break
                                else:
                                    count += 1
                                    print(count)
                                    print(file_full)
                            if (frame.shape[0] != default_size[0] or frame.shape[1] != default_size[1]):
                                frame = cv2.resize(frame, default_size_reverse, interpolation = cv2.INTER_AREA)
            
                            frame_name = str(uuid.uuid4()) + '.png'
                            predicted = instance.process(frame)
                            if predicted:
                                cv2.imwrite(os.path.join(classification_result_folder, frame_name), frame)
                                current += 1
                            if current >= number_per_video:
                                break
                        except:
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            print("Unexpected error:", exc_type, exc_value)
                            traceback.print_tb(exc_traceback, file=sys.stdout)
                    videoCapture.release()

def predict_with_classifier_final():
    leagues = os.listdir(dataset_folder)
    remake_dir(classification_result_folder)

    count = 0
    number_per_video = 200
    instance = Classifier()
    for league in leagues:
        file_full = os.path.join(dataset_folder, league)
                    
        videoCapture = cv2.VideoCapture(file_full)
        length = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        current = 0
        for i in range(500):
            try:
                frame_number = random.randint(0, length - 1)
                videoCapture.set(1, frame_number)
                ret, frame = videoCapture.read()
                if i == 0:
                    print(frame.shape)
                if (frame.shape[0] != default_size[0] or frame.shape[1] != default_size[1]):
                    frame = cv2.resize(frame, default_size_reverse, interpolation = cv2.INTER_AREA)
            
                frame_name = str(uuid.uuid4()) + '.png'
                predicted = instance.process(frame)
                if predicted:
                    cv2.imwrite(os.path.join(classification_result_folder, frame_name), frame)
                    current += 1
                if current >= number_per_video:
                    break
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("Unexpected error:", exc_type, exc_value)
                traceback.print_tb(exc_traceback, file=sys.stdout)
        videoCapture.release()
#test_classifier()
#prepare_data()
#split_train_eval()
#train()
#predict_with_classifier()
predict_with_classifier_final()