#!/usr/bin/env mdl
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from dataset import Dataset as localDataset
from filelist import AtomLabelGenerator, Label, Tag
import os

import cv2
import json
from typing import List, Tuple, Iterator
from common import config
import augmentor


def rand_range(rng, lo, hi):
    return rng.rand()*(hi-lo)+lo


class CASIADataset(Dataset):
    def __init__(self, dataset_name):
        self.rng = np.random.RandomState(np.random.RandomState(np.fromstring(os.urandom(4), 'uint8')).randint(2 ** 32))
        self.do_train = (dataset_name == "train")

        self.is_train = (dataset_name == "train")
        self.atom_generator = AtomLabelGenerator(rng=self.rng, tag='train').atom_generator()

    def __len__(self) -> int:
       return 256 * 1024

    def randWarp(self, img, rng):
        h, w = img.shape[:2]
        pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
        pts2 = np.float32([[w*0.1*rand_range(rng, -1, 1), h*0.1*rand_range(rng, -1, 1)],
                           [w*(1-0.1*rand_range(rng, -1, 1)), h*0.1*rand_range(rng, -1, 1)],
                           [w*0.1*rand_range(rng, -1, 1), h*(1-0.1*rand_range(rng, -1, 1))]])
        M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(img, M, (w, h))

    def randPadding(self, img, rng):
        h, w = img.shape[:2]
        h_l = int(h * 0.2 * rng.rand())
        h_r = int(h * 0.2 * rng.rand())
        w_l = int(w * 0.2 * rng.rand())
        w_r = int(w * 0.2 * rng.rand())
        img_new = np.zeros((h+h_l+h_r, w+w_l+w_r, 3))
        img_new[h_l:-h_r, w_l:-w_r, :] = img
        return img_new

    def randCrop(self, img, rng):
        h, w = img.shape[:2]
        h_start = int(h * 0.2 * rng.rand())
        h_end = h - int(h * 0.2 * rng.rand())
        w_start = int(w * 0.2 * rng.rand())
        w_end = w - int(w * 0.2 * rng.rand())
        return img[h_start:h_end, w_start:w_end, :]

    def rmZeros(self, img):
        img_gray = img[:, :, 0]
        h, w = img.shape[:2]
        h_start = 0
        h_end = h-1
        w_start = 0
        w_end = w-1

        h_sum = img_gray.sum(axis=0)
        w_sum = img_gray.sum(axis=1)
        while h_sum[w_start] < 5:
            w_start += 1
        while h_sum[w_end] < 5:
            w_end -= 1
        while w_sum[h_start] < 5:
            h_start += 1
        while w_sum[h_end] < 5:
            h_end -= 1

        return img[h_start:h_end, w_start:w_end]

    def img_aug(self, dep_img, dep_ld, scale, translation, rotation, sa, sb, canvas_size=[224, 192]):
        if dep_ld is None or dep_ld.shape[0] == 0:
            dep_img = self.rmZeros(dep_img)
            dep_img = cv2.resize(dep_img, (192, 224))
        else:
            dep_img, = augmentor.align(
                dep_img, ld=dep_ld,
                face_width=canvas_size[1]*0.45, canvas_size=canvas_size,
                scale=scale,
                translation=translation,
                rotation=rotation,
                sa=sa,
                sb=sb,
            )
        return dep_img

    def base_generator(self, depth_img, dep_ld, blacklist: False):
        rng = self.rng
        scale = (1 + rand_range(rng, -0.3, 0.3)**3 if self.is_train else 1)
        translation = ([rand_range(rng, -0.5, 0.5), rand_range(rng, -0.5, 0.5)] if self.is_train else [0, 0])
        ir_translation = list(map(lambda x: x*0.02*224, translation))
        rotation = (10*rand_range(rng, -1, 1)**3 if self.is_train else 0)
        sa = 1
        sb = 1
        if dep_ld == []:
            return None

        depth_img = self.img_aug(depth_img, dep_ld, scale, ir_translation, rotation, sa, sb)

        if depth_img is None:
            print('depth_img is None')
            return None

        h, w = depth_img.shape[:2]
        if self.is_train and rng.rand() >= 0.5:
            depth_img = cv2.flip(depth_img, 1)
        if self.is_train and rng.rand() >= 0.3:
            depth_img[h//2:, :] = 0

        return depth_img

    def instance_generator(self, encoded=False):
        while True:
            dep_img, dep_ld, label, tag = next(self.atom_generator)
            depth_img = self.base_generator(np.asarray(dep_img), np.asarray(dep_ld), False)
            if depth_img is None:
                continue

            depth_img = depth_img.astype('float32')
            depth_img = np.rollaxis(depth_img, 2)
            yield {
                'depth': depth_img,
                'label': np.array(label.value),
            }

    def __getitem__(self, index: int):
        data = next(self.instance_generator())
        dep_img = data['depth']
        label = data['label']
        return dep_img, label


if __name__ == '__main__':
    CASI_dataset = CASIADataset('train')
    dataloader = DataLoader(dataset=CASI_dataset, batch_size=256, shuffle=False, num_workers=1)
    # for data in dataloader:
    #     print(data[0].size(), data[1].size())
    for i in range(1000):
        data = next(iter(dataloader))
        print(data[0].size(), data[1].size())
        for j in range(10):
            dep = np.asarray(data[0][j][0])
            print(dep.shape)
            cv2.imshow('dep', dep/dep.max())
            cv2.waitKey()


# vim: ts=4 sw=4 sts=4 expandtab
