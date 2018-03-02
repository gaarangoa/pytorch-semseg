from ptsemseg.loader import retinopathy_loader as rt
from ptsemseg.augmentations import *
import ptsemseg.augmentations as augmentations
from torchvision import transforms, datasets
from ptsemseg.augmentations import *
import torchvision
import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from torch.utils import data

data_aug = [ RandomRotate(degree=10) ]

reload(rt)
reload(augmentations)
local_path = '/Volumes/drive/projects/machine_learning/deep-learning/datasets/retinopathy/'
dst = rt.RetinopathyLoader (local_path, is_transform=True, augmentations=data_aug)
trainloader = data.DataLoader(dst, batch_size=2)
for i, dtx in enumerate(trainloader):
    imgs, labels = dtx
    if i == 0:
        img = torchvision.utils.make_grid(imgs).numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]
        plt.imshow(img)
        plt.show()
        for j in range(2):
            plt.imshow(labels[j]*255)
            plt.show()
        break


# img = m.imread('/Volumes/drive/projects/machine_learning/deep-learning/datasets/retinopathy/images/training/IDRiD_01.jpg')
# img = np.array(img, dtype=np.uint8)

# reload(augmentations)



