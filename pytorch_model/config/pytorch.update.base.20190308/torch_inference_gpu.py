#!/usr/bin/env mdl
import argparse
import numpy as np
import os
import sys
import json
from glob import glob
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from model import Network
from pathlib import Path

datasets_catch = dict()


MEAN_FACE = np.array([
    [-0.17607, -0.172844],  # left eye pupil
    [0.1736, -0.17356],  # right eye pupil
    [-0.00182, 0.0357164],  # nose tip
    [-0.14617, 0.20185],  # left mouth corner
    [0.14496, 0.19943],  # right mouth corner
])


def get_mean_face(mf, face_width, canvas_size):
    ratio = face_width / (canvas_size * 0.34967)
    left_eye_pupil_y = mf[0][1]
    # In an aligned face image, the ratio between the vertical distances from eye to the top and bottom is 1:1.42
    ratioy = (left_eye_pupil_y * ratio + 0.5) * (1 + 1.42)
    mf[:, 0] = (mf[:, 0] * ratio + 0.5) * canvas_size
    mf[:, 1] = (mf[:, 1] * ratio + 0.5) * canvas_size / ratioy

    return mf


def get_align_transform(lm, mf):
    mx = mf[:, 0].mean()
    my = mf[:, 1].mean()
    dmx = lm[:, 0].mean()
    dmy = lm[:, 1].mean()

    ux = mf[:, 0] - mx
    uy = mf[:, 1] - my
    dux = lm[:, 0] - dmx
    duy = lm[:, 1] - dmy
    c1 = (ux * dux + uy * duy).sum()
    c2 = (ux * duy - uy * dux).sum()
    c3 = (dux**2 + duy**2).sum()
    a = c1 / c3
    b = c2 / c3

    kx, ky = 1, 1

    transform = np.zeros((2, 3))
    transform[0][0] = kx * a
    transform[0][1] = kx * b
    transform[0][2] = mx - kx * a * dmx - kx * b * dmy
    transform[1][0] = -ky * b
    transform[1][1] = ky * a
    transform[1][2] = my - ky * a * dmy + ky * b * dmx
    return transform


# aligh faces
def align5p(*imgs, ld, face_width, canvas_size, translation=[0, 0], rotation=0, scale=1, sa=1, sb=1, debug=False):

    lm = ld[[23, 68, 63, 36, 45]]
    if debug:
        for pt in lm:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(imgs[0], (x, y), 0, (0, 255, 0), 5)

    mf = MEAN_FACE * scale
    mf = get_mean_face(mf, face_width, canvas_size[1])

    M1 = np.eye(3)
    M1[:2] = get_align_transform(lm, mf)

    M2 = np.eye(3)
    M2[:2] = cv2.getRotationMatrix2D((canvas_size[1]/2, canvas_size[0]/2), rotation, 1)

    def stretch(va, vb, s):
        m = (va+vb)*0.5
        d = (va-vb)*0.5
        va[:] = m+d*s
        vb[:] = m-d*s

    mf = mf[[0, 1, 3, 4]].astype(np.float32)
    mf2 = mf.copy()
    stretch(mf2[0], mf2[1], sa)
    stretch(mf2[2], mf2[3], 1.0/sa)
    stretch(mf2[0], mf2[2], sb)
    stretch(mf2[1], mf2[3], 1.0/sb)
    mf2 += np.array(translation)
    M3 = cv2.getPerspectiveTransform(mf, mf2)
    M = M3.dot(M2).dot(M1)
    dshape = (canvas_size[1], canvas_size[0])
    return [cv2.warpPerspective(img, M, dshape, flags=cv2.INTER_LINEAR) for img in imgs]


def rmZeros(img):
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


def getirdepthimage(img, ld):
    ld = np.asarray(ld)
    img = np.asarray(img)
    if ld is None or ld.shape[0] == 0:
        img = rmZeros(img)
        img = cv2.resize(img, (192, 224))
    else:
        img, = align5p(
            img, ld=ld, face_width=192*0.45,
            canvas_size=[224, 192], scale=1
        )
    return img


def getLd(tag='val'):
    if tag == 'test':
        ldInfoPath = '../../../dataset/test_dep_ld.txt'
    elif tag == 'val':
        ldInfoPath = '../../../dataset/val_dep_ld.txt'
    else:
        return {}
    with open(ldInfoPath, 'r') as load_f:
        filename2ld = json.load(load_f)
    return filename2ld


def getTestList(tag):
    datasetList = []
    filename2ld = getLd(tag)
    num = 0
    if tag == 'val':
        fileList = '../../../dataset/val_private_list.txt'
    elif tag == 'test':
        fileList = '../../../dataset/test_public_list.txt'
    else:
        print('error')
        return []
    with open(fileList) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            datasetInfo = dict()
            line = line.strip()
            info = line.split(' ')
            if not len(info) == 4:
                print('info: ', info)
                continue
            datasetInfo['id'] = info[1]
            # datasetInfo['img'] = cv2.imread(info[1])
            if info[1] in filename2ld.keys():
                datasetInfo['ld'] = filename2ld[info[1]]
            else:
                datasetInfo['ld'] = []
                num += 1
            datasetList.append(datasetInfo)
    print('num no ld: ', num)
    return datasetList


def get_sore(datasetInfo, model):
    img_name = datasetInfo['id']
    img_dep = cv2.imread(img_name)
    ld_dep = datasetInfo['ld']
    img_dep = getirdepthimage(img_dep, ld_dep)

    img_dep = img_dep.astype('float32')
    # img_dep = i01c_to_ic01(img_dep)
    img_dep = np.rollaxis(img_dep, 2)
    img_dep = img_dep[np.newaxis, :, :, :]
    input_data = torch.from_numpy(img_dep).type(torch.FloatTensor)
    input_data = Variable(input_data)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    dense_pred = model(input_data)
    pred = dense_pred.mean(dim=1)
    pred = pred.tolist()
    return pred[0][1]


def get_resut(tag, model):
    # atoms = get_all_atoms(datasetIds)
    datasetList = getTestList(tag)
    # print(datasetList[:10])
    atomId_scores = []
    num = 0
    for datasetInfo in tqdm(datasetList):
        atomId_score = dict()
        score = get_sore(datasetInfo, model)
        # 分数取反
        atomId_score['score'] = score
        atomId_score['label'] = -1
        atomId_score['atom_id'] = datasetInfo['id']
        atomId_scores.append(atomId_score)
        num += 1

    result_json = dict()
    result_json['datasets'] = tag
    result_json['scores'] = atomId_scores

    return result_json


def ensure_dir(path: Path):
    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser("INFERENCE")
    parser.add_argument('tag', type=str, default='val', help='val or test')
    args = parser.parse_args()
    checkpoint_all = glob('train_log/models/epoch_*')
    ensure_dir((Path('train_log/eval/')))
    for checkpoint_path in checkpoint_all:
        print(checkpoint_path)
        save_name = 'train_log/eval/' + args.tag + '_224_192_epoch_' + checkpoint_path.split('epoch_')[-1] + '.json'
        if os.path.isfile(save_name):
            continue
        checkpoint = torch.load(checkpoint_path)
        model = Network()
        model.cuda()
        new_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        result_json = get_resut(args.tag, model)
        with open(save_name, 'w') as f:
            json.dump(result_json, f)


if __name__ == '__main__':
    main()


# vim: ts=4 sw=4 sts=4 expandtab
