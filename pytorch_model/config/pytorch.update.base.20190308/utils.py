import argparse
import datetime
import glob
from IPython import embed
import logging
import numpy as np
import os
import random
import sys
import shutil
import time

import nori2 as nori
import torch
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import visdom


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class Denseloss(nn.Module):
    def __init__(self, dropout=6):
        super(Denseloss, self).__init__()
        self.eps = 1e-9
        self.dropout = dropout

    def forward(self, dense_pred, targets):
        n, num, c = dense_pred.size()
        pred = dense_pred.mean(dim=1)
        targets = torch.zeros_like(pred).scatter_(1, targets.unsqueeze(1), 1)
        targets = targets.reshape(n, 1, c).expand(n, num, c)

        flat_preds = dense_pred.reshape(-1, c)
        targets = targets.reshape(-1, c)
        pt = torch.clamp(flat_preds, self.eps, 1-self.eps)
        pt = - torch.log(pt) * targets
        flat_xent = pt.sum(dim=1)

        dense_xent = flat_xent.reshape(n, num)
        sorted_xent, _ = dense_xent.sort(dim=1)
        if self.dropout <= 0:
            selected_xent = sorted_xent
        else:
            selected_xent = sorted_xent[:, :-self.dropout]
        return selected_xent.mean(dim=1).mean(dim=0)


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class TrainClock(object):
    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


def save_checkpoint(state, is_best, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = os.path.join(model_dir, 'checkpoint_' + str(state['epoch']) + '.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(model_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)
