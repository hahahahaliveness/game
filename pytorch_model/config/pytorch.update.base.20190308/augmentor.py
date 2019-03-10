#!/usr/bin/env mdl
# -*- coding:utf-8 -*-

import cv2
import numpy as np

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
    ratioy = (left_eye_pupil_y * ratio + 0.5) * (1 + 2)
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


def align(*imgs, ld, face_width, canvas_size, translation=[0, 0], rotation=0, scale=1, sa=1, sb=1, debug=False):
    # nose_tip = ld[63]
    # left_eye, right_eye = ld[23], ld[68]
    # left_mouth, right_mouth = ld[36], ld[45]

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

# vim: ts=4 sw=4 sts=4 expandtab
