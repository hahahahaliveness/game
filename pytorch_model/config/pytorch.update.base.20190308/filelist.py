#!/usr/bin/env mdl
import os
import enum
from typing import List, Tuple, Iterator
from tqdm import tqdm
import numpy as np
# import lovelive2
# from lovelive2.client.dataset_client import DatasetClient
import glob
import cv2
import json

class Device(enum.Enum):
    Face3D = 0
    OrbecOld = 1
    OrbecNew = 2
    ADI_TOF = 3
    Polynomial = 4


class Label(enum.Enum):
    GENUINE = 1
    ATTACK = 0


class Tag(enum.Enum):
    RGB = 0
    IR = 1
    DEPTH = 2


def combine_generator(generators: List[Tuple[Iterator, float]], rng=None):
    # generators is a list of tuple( generator, weight )
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState()

    gen_list = [g for g, w in generators if g is not None]
    weight_probs = np.array([float(w) for g, w in generators], "float")
    weight_probs = list(weight_probs / weight_probs.sum())
    weight_probs = [sum(weight_probs[:i]) for i in range(len(weight_probs))]

    while True:
        r = rng.rand()
        idx = -1
        for i in weight_probs:  # must be ascending
            if i < r:
                idx += 1
            else:
                break

        yield next(gen_list[idx])


def getLd(tag='train'):
    if tag == 'val':
        ldInfoPath = '../../../dataset/val_dep_ld.txt'
    else:
        ldInfoPath = '../../../dataset/train_dep_ld.txt'
    with open(ldInfoPath, 'r') as load_f:
        filename2ld = json.load(load_f)
    return filename2ld


def getTrainList(fileList):
    attacklist = []
    genuinelist = []
    filename2ld = getLd('train')
    num = 0
    with open(fileList) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            datasetInfo = dict()
            line = line.strip()
            info = line.split(' ')
            if not len(info) == 4:
                print('info: ', info)
                continue
            datasetInfo['img'] = cv2.imread(info[1])
            if info[1] in filename2ld.keys():
                datasetInfo['ld'] = filename2ld[info[1]]
            else:
                datasetInfo['ld'] = []
                num += 1
            if int(info[3]) == 0:
                attacklist.append(datasetInfo)
            else:
                genuinelist.append(datasetInfo)
    print('num no ld: ', num)
    return attacklist, genuinelist


class AtomLabelGenerator:

    # def __init__(self, rng=None, is_train=True):
    def __init__(self, rng=None, tag='train'):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()
        # self.lovelive_client = DatasetClient()
        self.is_train = (tag == 'train')
        # self.blacklist = get_blacklist()
        # self.attacklist, self.genuinelist = getTrainList('train_list.txt')
        if tag == 'val':
            dataset_list = '../../../dataset/val_private_list.txt'
        else:
            dataset_list = '../../../dataset/train_list.txt'
        self.attacklist, self.genuinelist = getTrainList(dataset_list)

    def base_generator(self, dataset_list, label, tag):
        # print("Initializing dataset: " + dataset_list)
        # nid_list = self.read_nidlist(dataset_list)
        # print("Initialized dataset: " + dataset_list)
        length = len(dataset_list)
        while True:
            dataInfo = dataset_list[self.rng.randint(length)]
            yield (np.asarray(dataInfo['img']), np.asarray(dataInfo['ld']), label, tag)

    def atom_generator(self):
        # NOTE: change here
        # export to dataset.py as the main generator

        attack_generator = self.base_generator(self.attacklist, label=Label.ATTACK, tag=Tag.RGB)
        genuine_generator = self.base_generator(self.genuinelist, label=Label.GENUINE, tag=Tag.RGB)
        main_generator = combine_generator(
            [(genuine_generator, 1.), (attack_generator, 1.)], rng=self.rng
        )
        yield from main_generator


if __name__ == "__main__":
    atom_generator = AtomLabelGenerator().atom_generator()
    count = 0
    total = 1e4
    for img, label, tag in tqdm(atom_generator, total=total):
        count += 1
        cv2.imshow('img', img)
        cv2.waitKey()
        if count > total:
            break

# vim: ts=4 sw=4 sts=4 expandtab
