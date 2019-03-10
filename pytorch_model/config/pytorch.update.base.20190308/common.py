#!/usr/bin/env mdl

import os
import hashlib
# import getpass


class Config:

    log_dir = 'train_log'

    log_model_dir = os.path.join(log_dir, 'models')
    '''where to write model snapshots to'''

    log_file = os.path.join(log_dir, 'log.txt')

    exp_name = os.path.basename(log_dir)
    '''name of this experiment'''

    minibatch_size = 256

    nr_channel = 5  # input channels
    rgb_channel = 3  # input channels
    ir_channel = 1  # input channels
    image_shape = (224, 192)
    depth_image_shape = (224, 192)
    nr_class = 2

    @property
    def input_shape(self):
        return (self.minibatch_size,
                self.nr_channel) + self.image_shape

    @property
    def depth_input_shape(self):
        return (self.minibatch_size,
                self.ir_channel) + self.depth_image_shape

    def real_path(self, path):
        '''convert relative path to base_dir to absolute path
        :return: absolute path '''
        return os.path.join(self.base_dir, path)


config = Config()

# vim: foldmethod=marker
