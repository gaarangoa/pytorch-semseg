import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data

from ptsemseg.utils import recursive_glob

class RetinopathyLoader(data.Dataset):
    def __init__(self, root, split="training", is_transform=False, img_size=512, augmentations=[]):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.classes = ["MA"]
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)

        for split in ["training", "validation",]:
            file_list = recursive_glob(rootdir=self.root + 'images/' + self.split + '/', suffix='.jpg')
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        # print("processing: ", img_path)
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if self.is_transform:
            img, lbl = self.transform(img, img_path)

        return img, lbl


    def transform(self, img, lbl_path):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        lbl = self.encode_segmap(lbl_path)
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)
        # print("classes:", classes, "labels in images:",np.unique(lbl))

        assert(np.all(classes == np.unique(lbl)))

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    def encode_segmap(self, lbl_path):
        labels = []
        lbl_t_path = lbl_path[:-4]
        for lbx,lbi in enumerate(self.classes):
            lbl_path = lbl_t_path + '_' + lbi + '.tif'
            mask = m.imread(lbl_path)
            mask = np.array(mask, dtype=np.int32)
            mask = mask.astype(int)
            label_mask = np.zeros((mask.shape[0], mask.shape[1]))
            label_mask[ mask[:,:,0] > 0 ] = lbx+1
            label_mask = np.array(label_mask, dtype=np.uint8)
            labels.append(label_mask)
        # 
        # condense the labels into one big label image
        lbs = labels[0]
        for i in labels[1:]:
            lbs+=i
        return lbs

    def decode_segmap(self, temp, plot=False):
        # TODO:(@meetshah1995)
        # Verify that the color mapping is 1-to-1
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = l
            g[temp == l] = l
            b[temp == l] = 0

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = (r/255.0)
        rgb[:, :, 1] = (g/255.0)
        rgb[:, :, 2] = (b/255.0)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

if __name__ == '__main__':
    local_path = '/home/gustavo1/deep_learning/datasets/IDRiD/'
    dst = RetinopathyLoader (local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            for j in range(4):
                plt.imshow(dst.decode_segmap(labels.numpy()[j]))
                plt.show()
