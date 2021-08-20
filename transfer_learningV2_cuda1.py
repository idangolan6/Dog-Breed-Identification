import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cv2
import os
import tqdm as tq
import scipy.io


def read_annotation_file(file_path):
    f = open(file_path, "r")
    data = f.read()
    data = data.split('\n')
    x_min = int(data[18].split('<')[1].split('>')[1])
    y_min = int(data[19].split('<')[1].split('>')[1])
    x_max = int(data[20].split('<')[1].split('>')[1])
    y_max = int(data[21].split('<')[1].split('>')[1])
    return x_min, y_min, x_max, y_max


# paths
path_to_save_results_ephoc = 'runs/exp32_SGD_ep_bz_128'
path_to_save_results_batch = 'runs/exp32_SGD_bt_bz_128'
path_data_set = './data_set'
path_images_data = os.path.join(path_data_set, 'Images')
path_annotation = os.path.join(path_data_set, 'Annotation')
path_test_list = os.path.join(path_data_set, 'test_list.mat')
path_train_list = os.path.join(path_data_set, 'train_list.mat')

# read test and train lists
test_list = scipy.io.loadmat(path_test_list)
train_list = scipy.io.loadmat(path_train_list)

# remove unnesscery keys
to_remove = ['__header__', '__version__', '__globals__']
for rem in to_remove:
    test_list.pop(rem)
    train_list.pop(rem)

# reduce dims for dataFrame
for key in test_list.keys():
    test_list[key] = np.squeeze(test_list[key])
    train_list[key] = np.squeeze(train_list[key])

# create dataFrames for tests and lists
test_data_table = pd.DataFrame(data=test_list)
train_data_table = pd.DataFrame(data=train_list)

# add full paths to ims
test_data_table['file_list'] = test_data_table['file_list'].apply(lambda x: os.path.join(path_images_data, x.item()))
train_data_table['file_list'] = train_data_table['file_list'].apply(lambda x: os.path.join(path_images_data, x.item()))

# add full paths to annotation
test_data_table['annotation_list'] = test_data_table['annotation_list'].apply(lambda x: os.path.join(path_annotation, x.item()))
train_data_table['annotation_list'] = train_data_table['annotation_list'].apply(lambda x: os.path.join(path_annotation, x.item()))

# convert labels to start from 0
test_data_table['labels'] = test_data_table['labels'].apply(lambda x: x - 1)
train_data_table['labels'] = train_data_table['labels'].apply(lambda x: x - 1)

# shuffle the data
test_data_table = test_data_table.sample(frac=1)
train_data_table = train_data_table.sample(frac=1)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
writer_e = SummaryWriter(log_dir=path_to_save_results_ephoc)
writer_b = SummaryWriter(log_dir=path_to_save_results_batch)

data_train = []
labels_train = []
data_test = []
labels_test = []

for i in tq.tqdm(test_data_table.index):
    annotation_path = test_data_table['annotation_list'][i]
    x_min, y_min, x_max, y_max = read_annotation_file(annotation_path)

    im_path = test_data_table['file_list'][i]
    frame = cv2.imread(im_path)
    frame = frame[y_min:y_max, x_min:x_max, :]
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame = transform(frame)
    data_test.append(frame)

    label = test_data_table['labels'][i]
    labels_test.append(torch.tensor(label))

for i in tq.tqdm(train_data_table.index):
    annotation_path = train_data_table['annotation_list'][i]
    x_min, y_min, x_max, y_max = read_annotation_file(annotation_path)

    im_path = train_data_table['file_list'][i]
    frame = cv2.imread(im_path)
    frame = frame[y_min:y_max, x_min:x_max, :]
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame = transform(frame)
    data_train.append(frame)

    label = train_data_table['labels'][i]
    labels_train.append(torch.tensor(label))

data_test = torch.stack(data_test)
labels_test = torch.stack(labels_test)
data_train = torch.stack(data_train)
labels_train = torch.stack(labels_train)

model = torchvision.models.vgg16(pretrained=True)
in_featurs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features=in_featurs, out_features=120, bias=True)

# Hyper Parameters
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
model = model.to(device)

EPHOC = 40
lr = 0.01
bz = 128
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr)

len_data_train = len(data_train)
len_data_test = len(data_test)

train_num_of_batches = len_data_train//bz
test_num_of_batches = len_data_test//bz

loss_train_list = []
acc_train_list = []
loss_test_list = []
acc_test_list = []

for ephoc in range(EPHOC):
    temp_loss_train_list = []
    temp_acc_train_list = []
    temp_loss_test_list = []
    temp_acc_test_list = []

    for i in tq.tqdm(range(train_num_of_batches + 1)):
        if i == train_num_of_batches:
            in_data = data_train[i*bz:]
            lab = labels_train[i*bz:]
        else:
            in_data = data_train[i * bz:(i + 1)*bz]
            lab = labels_train[i * bz:(i + 1) * bz]

        in_data = in_data.to(device)
        lab = lab.to(device)

        pred = model(in_data)
        loss = loss_fn(pred, lab)

        optim.zero_grad()
        loss.backward()
        optim.step()

        output_class = torch.argmax(pred, dim=1)
        acc = torch.mean((lab.to(device) == output_class).type(torch.float32))

        temp_loss_train_list.append(loss.to("cpu").item())
        temp_acc_train_list.append(acc.to("cpu").item())

        writer_b.add_scalar('Loss/train', temp_loss_train_list[-1])
        writer_b.add_scalar('Accuracy/train', temp_acc_train_list[-1])

    with torch.no_grad():
        for i in tq.tqdm(range(test_num_of_batches + 1)):
            if i == test_num_of_batches:
                in_data = data_test[i * bz:]
                lab = labels_test[i * bz:]
            else:
                in_data = data_test[i * bz:(i + 1) * bz]
                lab = labels_test[i * bz:(i + 1) * bz]
            in_data = in_data.to(device)
            lab = lab.to(device)

            pred = model(in_data)
            loss = loss_fn(pred, lab)

            output_class = torch.argmax(pred, dim=1)
            acc = torch.mean((lab.to(device) == output_class).type(torch.float32))

            temp_loss_test_list.append(loss.to("cpu").item())
            temp_acc_test_list.append(acc.to("cpu").item())

            writer_b.add_scalar('Loss/test', temp_loss_test_list[-1])
            writer_b.add_scalar('Accuracy/test', temp_acc_test_list[-1])

    loss_train_list.append(np.mean(temp_loss_train_list))
    acc_train_list.append(np.mean(temp_acc_train_list))
    loss_test_list.append(np.mean(temp_loss_test_list))
    acc_test_list.append(np.mean(temp_acc_test_list))

    writer_e.add_scalar('Loss/train',     np.mean(temp_loss_train_list),   ephoc)
    writer_e.add_scalar('Accuracy/train', np.mean(temp_acc_train_list),    ephoc)
    writer_e.add_scalar('Loss/test',      np.mean(temp_loss_test_list),    ephoc)
    writer_e.add_scalar('Accuracy/test',  np.mean(temp_acc_test_list),     ephoc)

    print("Ephoc {}: train_loss - {}, test_loss - {}, train_acc - {}, test_acc - {}".format(ephoc,
                                                                                            loss_train_list[-1],
                                                                                            loss_test_list[-1],
                                                                                            acc_train_list[-1],
                                                                                            acc_test_list[-1]))
writer_b.flush()
writer_b.close()
writer_e.flush()
writer_e.close()
print("finish")
