import torch
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np
import json
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import random


def WaveletTransformAxisY(batch_img):
    odd_img = batch_img[:, 0::2]
    even_img = batch_img[:, 1::2]
    L = (odd_img + even_img) / 2.0
    H = torch.abs(odd_img - even_img)

    return L, H


def WaveletTransformAxisX(batch_img):
    # transpose + fliplr
    tmp_batch = torch.permute(batch_img, (0, 2, 1))
    tmp_batch = torch.fliplr(tmp_batch)
    _dst_L, _dst_H = WaveletTransformAxisY(tmp_batch)
    # transpose + flipud
    dst_L = torch.permute(_dst_L, [0, 2, 1])
    dst_L = torch.flipud(dst_L)
    dst_H = torch.permute(_dst_H, [0, 2, 1])
    dst_H = torch.flipud(dst_H)

    return dst_L, dst_H


def Wavelet(batch_image):
    r = batch_image[:, 0]
    g = batch_image[:, 1]
    b = batch_image[:, 2]

    # level 1 decomposition
    wavelet_L, wavelet_H = WaveletTransformAxisY(r)
    r_wavelet_LL, r_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    r_wavelet_HL, r_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_L, wavelet_H = WaveletTransformAxisY(g)
    g_wavelet_LL, g_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    g_wavelet_HL, g_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_L, wavelet_H = WaveletTransformAxisY(b)
    b_wavelet_LL, b_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    b_wavelet_HL, b_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_data = [r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH,
                    g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH,
                    b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH]
    transform_batch = torch.stack(wavelet_data, axis=1)

    # level 2 decomposition
    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(r_wavelet_LL)
    r_wavelet_LL2, r_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    r_wavelet_HL2, r_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(g_wavelet_LL)
    g_wavelet_LL2, g_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    g_wavelet_HL2, g_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(b_wavelet_LL)
    b_wavelet_LL2, b_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    b_wavelet_HL2, b_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_data_l2 = [r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2,
                       g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2,
                       b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2]
    transform_batch_l2 = torch.stack(wavelet_data_l2, dim=1)

    # level 3 decomposition
    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(r_wavelet_LL2)
    r_wavelet_LL3, r_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    r_wavelet_HL3, r_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(g_wavelet_LL2)
    g_wavelet_LL3, g_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    g_wavelet_HL3, g_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(b_wavelet_LL2)
    b_wavelet_LL3, b_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    b_wavelet_HL3, b_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_data_l3 = [r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3,
                       g_wavelet_LL3, g_wavelet_LH3, g_wavelet_HL3, g_wavelet_HH3,
                       b_wavelet_LL3, b_wavelet_LH3, b_wavelet_HL3, b_wavelet_HH3]
    transform_batch_l3 = torch.stack(wavelet_data_l3, dim=1)

    # level 4 decomposition
    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(r_wavelet_LL3)
    r_wavelet_LL4, r_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    r_wavelet_HL4, r_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(g_wavelet_LL3)
    g_wavelet_LL4, g_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    g_wavelet_HL4, g_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(b_wavelet_LL3)
    b_wavelet_LL4, b_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    b_wavelet_HL4, b_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    wavelet_data_l4 = [r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4,
                       g_wavelet_LL4, g_wavelet_LH4, g_wavelet_HL4, g_wavelet_HH4,
                       b_wavelet_LL4, b_wavelet_LH4, b_wavelet_HL4, b_wavelet_HH4]
    transform_batch_l4 = torch.stack(wavelet_data_l4, dim=1)

    return [transform_batch, transform_batch_l2, transform_batch_l3, transform_batch_l4]


class Wavelet_Model(torch.nn.Module):
    def __init__(self):
        super(Wavelet_Model, self).__init__()
        self.conv_1 = nn.Conv2d(12, 64, kernel_size=(3, 3), padding=1)
        self.norm_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.ReLU()

        self.conv_1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm_1_2 = nn.BatchNorm2d(64)
        self.relu_1_2 = nn.ReLU()
        #################################################################################################
        self.conv_a = nn.Conv2d(12, 64, kernel_size=(3, 3), padding=1)
        self.norm_a = nn.BatchNorm2d(64)
        self.relu_a = nn.ReLU()
        #################################################################################################
        self.conv_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.norm_2 = nn.BatchNorm2d(128)
        self.relu_2 = nn.ReLU()

        self.conv_2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm_2_2 = nn.BatchNorm2d(128)
        self.relu_2_2 = nn.ReLU()
        #################################################################################################
        self.conv_b = nn.Conv2d(12, 128, kernel_size=(3, 3), padding=1)
        self.norm_b = nn.BatchNorm2d(128)
        self.relu_b = nn.ReLU()

        self.conv_b_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.norm_b_2 = nn.BatchNorm2d(128)
        self.relu_b_2 = nn.ReLU()
        #################################################################################################
        self.conv_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.norm_3 = nn.BatchNorm2d(256)
        self.relu_3 = nn.ReLU()

        self.conv_3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm_3_2 = nn.BatchNorm2d(256)
        self.relu_3_2 = nn.ReLU()
        #################################################################################################
        self.conv_c = nn.Conv2d(12, 256, kernel_size=(3, 3), padding=1)
        self.norm_c = nn.BatchNorm2d(256)
        self.relu_c = nn.ReLU()

        self.conv_c_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.norm_c_2 = nn.BatchNorm2d(256)
        self.relu_c_2 = nn.ReLU()

        self.conv_c_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.norm_c_3 = nn.BatchNorm2d(256)
        self.relu_c_3 = nn.ReLU()
        #################################################################################################
        self.conv_4 = nn.Conv2d(512, 256, kernel_size=(3, 3), padding=1)
        self.norm_4 = nn.BatchNorm2d(256)
        self.relu_4 = nn.ReLU()

        self.conv_4_2 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm_4_2 = nn.BatchNorm2d(128)
        self.relu_4_2 = nn.ReLU()
        #################################################################################################
        self.conv_5 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.norm_5 = nn.BatchNorm2d(128)
        self.relu_5 = nn.ReLU()

        self.pool_5 = nn.AvgPool2d(kernel_size=(7, 7), stride=1, padding=1)
        self.flat_5 = nn.Flatten()

        self.fc_5 = nn.Linear(1152, 2048)
        self.norm_5_1 = nn.BatchNorm1d(2048)
        self.relu_5_1 = nn.ReLU()
        self.drop_5 = nn.Dropout(0.5)
        #################################################################################################
        self.fc_6 = nn.Linear(2048, 7)
        self.norm_6 = nn.BatchNorm1d(7)
        self.relu_6 = nn.ReLU()
        self.drop_6 = nn.Dropout(0.5)
        #################################################################################################
        self.output_fc = nn.Linear(7, 7)

    def forward(self, x):
        input_l1, input_l2, input_l3, input_l4 = Wavelet(x)
        #################################################################################################
        # print('input shape: ', input_l1.shape)
        out_1 = self.conv_1(input_l1)
        # print('conv_1 output shape: ', out_1.shape)
        out_1 = self.norm_1(out_1)
        out_1 = self.relu_1(out_1)

        out_1 = self.conv_1_2(out_1)
        # print('conv_1_2 output shape: ', out_1.shape)
        out_1 = self.norm_1_2(out_1)
        out_1 = self.relu_1_2(out_1)
        #################################################################################################
        out_2 = self.conv_a(input_l2)
        # print('conv_a output shape: ', out_2.shape)
        out_2 = self.norm_a(out_2)
        out_2 = self.relu_a(out_2)

        cat_2 = torch.cat((out_1, out_2), 1)
        # print('concatenate result: ', cat_2.shape)
        out_2 = self.conv_2(cat_2)
        # print('conv_2 output shape: ', out_2.shape)
        out_2 = self.norm_2(out_2)
        out_2 = self.relu_2(out_2)

        out_2 = self.conv_2_2(out_2)
        # print('conv_2_2 output shape: ', out_2.shape)
        out_2 = self.norm_2_2(out_2)
        out_2 = self.relu_2_2(out_2)
        #################################################################################################
        out_3 = self.conv_b(input_l3)
        # print('conv_b output shape: ', out_3.shape)
        out_3 = self.norm_b(out_3)
        out_3 = self.relu_b(out_3)

        out_3 = self.conv_b_2(out_3)
        # print('conv_b_2 output shape: ', out_3.shape)
        out_3 = self.norm_b_2(out_3)
        out_3 = self.relu_b_2(out_3)
        #################################################################################################
        cat_3 = torch.cat((out_2, out_3), 1)
        # print('concatenate result: ', cat_3.shape)
        out_3 = self.conv_3(cat_3)
        # print('conv_3 output shape: ', out_3.shape)
        out_3 = self.norm_3(out_3)
        out_3 = self.relu_3(out_3)

        out_3 = self.conv_3_2(out_3)
        # print('conv_3_2 output shape: ', out_3.shape)
        out_3 = self.norm_3_2(out_3)
        out_3 = self.relu_3_2(out_3)
        #################################################################################################
        out_4 = self.conv_c(input_l4)
        # print('conv_c output shape: ', out_4.shape)
        out_4 = self.norm_c(out_4)
        out_4 = self.relu_c(out_4)

        out_4 = self.conv_c_2(out_4)
        # print('conv_c_2 output shape: ', out_4.shape)
        out_4 = self.norm_c_2(out_4)
        out_4 = self.relu_c_2(out_4)

        out_4 = self.conv_c_3(out_4)
        # print('conv_c_3 output shape: ', out_4.shape)
        out_4 = self.norm_c_3(out_4)
        out_4 = self.relu_c_3(out_4)
        #################################################################################################
        cat_4 = torch.cat((out_3, out_4), 1)
        # print('concatenate result: ', cat_4.shape)
        out_4 = self.conv_4(cat_4)
        # print('conv_4 output shape: ', out_4.shape)
        out_4 = self.norm_4(out_4)
        out_4 = self.relu_4(out_4)

        out_4 = self.conv_4_2(out_4)
        # print('conv_4_2 output shape: ', out_4.shape)
        out_4 = self.norm_4_2(out_4)
        out_4 = self.relu_4_2(out_4)
        #################################################################################################
        out_5 = self.conv_5(out_4)
        # print('conv_5 output shape: ', out_5.shape)
        out_5 = self.norm_5(out_5)
        out_5 = self.relu_5(out_5)

        out_5 = self.pool_5(out_5)
        out_5 = self.flat_5(out_5)
        #################################################################################################
        out_5 = self.fc_5(out_5)
        # print('fc_5 output shape: ', out_5.shape)
        out_5 = self.norm_5_1(out_5)
        out_5 = self.relu_5_1(out_5)
        out_5 = self.drop_5(out_5)

        out_6 = self.fc_6(out_5)
        # print('fc6 output shape: ', out_6.shape)
        out_6 = self.norm_6(out_6)
        out_6 = self.relu_6(out_6)
        out_6 = self.drop_6(out_6)
        #################################################################################################
        output = self.output_fc(out_6)

        return output


batch_size = 512
random_seed = 888
random.seed(random_seed)
torch.manual_seed(random_seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = './dataset/'
texture_dataset = datasets.ImageFolder(
    data_path,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

train_idx, val_idx = train_test_split(list(range(len(texture_dataset))), test_size=0.2, random_state=random_seed)
datasets = {}
datasets['train'] = Subset(texture_dataset, train_idx)
datasets['valid'] = Subset(texture_dataset, val_idx)

dataloaders, batch_num = {}, {}
dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'],
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=4)
dataloaders['valid'] = torch.utils.data.DataLoader(datasets['valid'],
                                                   batch_size=batch_size, shuffle=False,
                                                   num_workers=4)

batch_num['train'], batch_num['valid'] = len(dataloaders['train']), len(dataloaders['valid'])
print('batch_size : %d,  tvt : %d / %d' % (batch_size, batch_num['train'], batch_num['valid']))

model = Wavelet_Model().to(device)

since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

lmbda = lambda epoch: 0.98739
exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
num_epochs = 100

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss, running_corrects, num_cnt = 0.0, 0, 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            num_cnt += len(labels)
        if phase == 'train':
            exp_lr_scheduler.step()

        epoch_loss = float(running_loss / num_cnt)
        epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

        if phase == 'train':
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
        else:
            valid_loss.append(epoch_loss)
            valid_acc.append(epoch_acc)
        print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'valid' and epoch_acc > best_acc:
            best_idx = epoch
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            #                 best_model_wts = copy.deepcopy(model.module.state_dict())
            print('==> best model saved - %d / %.1f' % (best_idx, best_acc))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'Wavelet_cnn.pt')
print('model saved')
