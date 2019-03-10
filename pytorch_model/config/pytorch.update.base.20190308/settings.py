#!/usr/bin/env mdl
# -*- coding:utf-8 -*-

import os
from common import config


class Setting:
    imagenet_path = '../../../dataset/imagenet_basemodel'

    base_epoch = 0
    stop_epoch = 120
    dump_epoch_interval = 1
    instance_per_epoch = 256 * 1024  # 32768 * 8

    minibatch_size = 256
    minibatch_per_epoch = instance_per_epoch // minibatch_size

    learning_rate = 1e-4
    eps = 1e-8
    betas = (0.9, 0.999)
    weight_decay = 4e-5

    log_dir = 'train_log'
    rg_channel = 3
    log_model_dir = os.path.join(log_dir, 'models')

    def get_learning_rate(self, epoch, step):
        M = self.minibatch_per_epoch
        Tw = M * 3
        if step < Tw:
            f = step / Tw
            return f * 1e-3
        if epoch < 20:
            return 1e-3
        if epoch < 40:
            return 2e-4 * 2
        if epoch < 70:
            return 5e-5 * 2
        if epoch < 90:
            return 5e-5
        return 1e-5


train_spec = Setting()
