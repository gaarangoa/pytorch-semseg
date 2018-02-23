import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *


# Setup Augmentations
data_aug= Compose([RandomRotate(10),                                        
                    RandomHorizontallyFlip()])

# Setup Dataloader

args_dataset = "retinopathy" #pascal
args_img_rows = 128
args_img_cols = 128
args_batch_size = 10
args_arch = "fcn8s"
args_l_rate = 0.01
args_resume = None
args_n_epoch = 10
args_visdom = False

print("loading dataset")

data_loader = get_loader(args_dataset)
data_path = get_data_path(args_dataset)
t_loader = data_loader(data_path, is_transform=True, img_size=(args_img_rows, args_img_cols), augmentations=data_aug)
v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args_img_rows, args_img_cols), augmentations=data_aug)

print("dataset read: loading data")
n_classes = t_loader.n_classes
trainloader = data.DataLoader(t_loader, batch_size=args_batch_size, num_workers=8, shuffle=True)
valloader = data.DataLoader(v_loader, batch_size=args_batch_size, num_workers=8)

for i, (images, labels) in enumerate(trainloader):
    images = Variable(images.cuda())
    labels = Variable(labels.cuda())
    break








# Setup Metrics
running_metrics = runningScore(n_classes)


# Setup Model
model = get_model(args_arch, n_classes)

model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
model.cuda()

# Check if model has custom optimizer / loss
if hasattr(model.module, 'optimizer'):
    optimizer = model.module.optimizer
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args_l_rate, momentum=0.99, weight_decay=5e-4)

if hasattr(model.module, 'loss'):
    print('Using custom loss')
    loss_fn = model.module.loss
else:
    loss_fn = cross_entropy2d


best_iou = -100.0 
for epoch in range(args_n_epoch):
    model.train()
    for i, (images, labels) in enumerate(trainloader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        break

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(input=outputs, target=labels)

        loss.backward()
        optimizer.step()

        if (i+1) % 20 == 0:
            print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args_n_epoch, loss.data[0]))


max(labels[1][:].tolist())



local_path = '/home/gustavo1/deep_learning/datasets/IDRiD/'
dst = rt.RetinopathyLoader (local_path, is_transform=True)
trainloader = data.DataLoader(dst, batch_size=4)
for i, data in enumerate(trainloader):
    imgs, labels = data
    break

max(labels[1][:].tolist())



img_path = '/home/gustavo1/deep_learning/datasets/IDRiD/images/training/IDRiD_35.jpg'
lbl_path = img_path[:-4] + '_MA.tif'

img = m.imread(img_path)
img = np.array(img, dtype=np.uint8)

lbl = m.imread(lbl_path)
lbl = np.array(lbl, dtype=np.int32)

# if self.is_transform:
#     img, lbl = self.transform(img, lbl)

# return img, lbl

# def transform(self, img, lbl):
s_mean = np.array([104.00699, 116.66877, 122.67892])
img_size = [128, 128]

img = img[:, :, ::-1]
img = img.astype(np.float64)
img -= s_mean
img = m.imresize(img, (img_size[0], img_size[1]))
# Resize scales images from 0 to 255, thus we need
# to divide by 255.0
img = img.astype(float) / 255.0
# NHWC -> NCWH
img = img.transpose(2, 0, 1)

lbl = self.encode_segmap(lbl)
classes = np.unique(lbl)
lbl = lbl.astype(float)
lbl = m.imresize(lbl, (img_size[0], img_size[1]), 'nearest', mode='F')
lbl = lbl.astype(int)
assert(np.all(classes == np.unique(lbl)))

img = torch.from_numpy(img).float()
lbl = torch.from_numpy(lbl).long()
# return img, lbl


# def encode_segmap(self, lbl):
# Refer : http://groups.csail.mit.edu/vision/datasets/ADE20K/code/loadAde20K.m
lbl = lbl.astype(int)
label_mask = np.zeros((lbl.shape[0], lbl.shape[1]))
label_mask[ lbl[:,:,0] > 0 ] = 1
label_mask = np.array(label_mask, dtype=np.uint8)
# return np.array(label_mask, dtype=np.uint8)


max(max(label_mask.tolist()))
max(max(lbl[:,:,0].tolist()))