#!/usr/bin/env mdl
import argparse
import os
# from setproctitle import setproctitle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

# from common import config
import model
from settings import train_spec
import utils
from pathlib import Path
from dataset_pytorch import CASIADataset
from torch.utils.data import DataLoader
from utils import TrainClock


def ensure_dir(path: Path):
    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)


class Session:

    def __init__(self, train_spec, net=None):
        # setproctitle(config.exp_name)

        self.log_dir = train_spec.log_dir
        ensure_dir(self.log_dir)
        # logconf.set_output_file(os.path.join(self.log_dir, 'log.txt'))
        self.model_dir = train_spec.log_model_dir
        ensure_dir(self.model_dir)

        self.net = net
        self.clock = TrainClock()

    def start(self):
        self.save_checkpoint('start')

    def save_checkpoint(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        tmp = {
            'network': self.net.cpu(),
            'epoch': self.clock.epoch,
            'step': self.clock.step,
            'state_dict': self.net.state_dict(),
        }
        torch.save(tmp, ckp_path)
        self.net = self.net.cuda()

    def load_checkpoint(self, ckp_path):
        checkpoint = torch.load(ckp_path)
        self.clock.epoch = checkpoint['epoch']
        self.clock.step = checkpoint['step']
        self.net.load_state_dict(checkpoint['state_dict'], strict=False)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--fast-run', action='store_true', default=False)
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('-c', '--continue', dest='continue_path', required=False)
    args = parser.parse_args()

    cudnn.benchmark = True
    cudnn.enabled = True

    net = model.Network()
    # logger.info(net)
    net = nn.DataParallel(net).cuda()

    # create session
    sess = Session(train_spec, net=net)

    # worklog = WorklogLogger(os.path.join(sess.log_dir, 'worklog.txt'))

    criterion = utils.Denseloss(dropout=5)
    criterion = criterion.cuda()

    all_parameters = net.parameters()

    optimizer = torch.optim.Adam(
        [{'params': all_parameters,
          'weight_decay': train_spec.weight_decay,
          'lr': train_spec.learning_rate}],
    )

    adam_opt = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    def adjust_lr(epoch, step):
        lr = train_spec.get_learning_rate(epoch, step)
        for params_group in adam_opt.param_groups:
            params_group['lr'] = lr
        return lr

    # Now start train
    clock = sess.clock
    clock.epoch = 0
    clock.step = 0 * 1024
    sess.start()

    # restore checkpoint
    checkpoint = torch.load(train_spec.imagenet_path)
    sess.net.load_state_dict(checkpoint['state_state'], strict=False)

    if args.continue_path and os.path.exists(args.continue_path):
        sess.load_checkpoint(args.continue_path)
    for ite in range(sess.clock.step):
        adam_opt.step()

    # log_output = log_rate_limited(min_interval=1)(worklog.put_line)

    CASI_dataset = CASIADataset('train')
    dataloader = DataLoader(dataset=CASI_dataset, batch_size=train_spec.minibatch_size, shuffle=False, num_workers=8)

    # for epoch in train_ds.epoch_generator():
    for epoch in range(train_spec.stop_epoch):
        # if clock.epoch > train_spec.stop_epoch:
        #     break
        time_epoch_start = tstart = time.time()
        step = 0
        sess.net.train()
        adjust_lr(epoch, clock.step)
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        # for step in range(train_spec.minibatch_per_epoch):
        for step in range(train_spec.minibatch_per_epoch):


            minibatch = next(iter(dataloader))
            # scheduler.step()
            adam_opt.step()
            # input_data = minibatch['depth']
            # target = minibatch['label']
            input_data = minibatch[0]
            target = minibatch[1]
            input_data = input_data.type(torch.FloatTensor)
            # target = torch.from_numpy(target).type(torch.LongTensor)
            target = target.type(torch.LongTensor)
            input_data = Variable(input_data).cuda()
            target = Variable(target).cuda(async=True)
            tdata = time.time() - tstart

            optimizer.zero_grad()
            dense_pred = sess.net(input_data)
            pred = dense_pred.mean(dim=1)
            loss = criterion(dense_pred, target)
            loss.backward()
            optimizer.step()

            cur_time = time.time()
            ttrain = cur_time - tstart
            time_passed = cur_time - time_epoch_start

            # time_expected = time_passed / (clock.minibatch + 1) * train_ds.minibatch_per_epoch
            time_expected = time_passed / (clock.minibatch + 1) * train_spec.minibatch_per_epoch
            eta = time_expected - time_passed

            prec1, = utils.accuracy(pred, target, topk=(1,))

            n = input_data.size(0)
            objs.update(loss.item(), n)  # accumulated loss
            top1.update(prec1.item(), n)

            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']
            outputs = [
                # "e:{},{}/{}".format(clock.epoch, clock.minibatch, train_ds.minibatch_per_epoch),
                "e:{},{}/{}".format(clock.epoch, clock.minibatch, train_spec.minibatch_per_epoch),
                "{:.2g} mb/s".format(1./ttrain),
            ] + [
                "lr:{:.6f}, loss:{:.3f}, top1_acc:{:.2f}%".format(cur_lr, objs.avg, top1.avg)
            ] + [
                'passed:{:.2f}'.format(time_passed),
                'eta:{:.2f}'.format(eta),
            ]
            if tdata/ttrain > .05:
                outputs += ["dp/tot: {:.2g}".format(tdata/ttrain)]
            print(outputs)
            # log_output(' '.join(outputs))
            clock.tick()
            tstart = time.time()
            # sess.save_checkpoint('epoch_{}_{}'.format(clock.epoch, clock.step))

            # sess.save_checkpoint('epoch_{}'.format(clock.epoch))

        clock.tock()

        if clock.epoch % train_spec.dump_epoch_interval == 0:
            sess.save_checkpoint('epoch_{}'.format(clock.epoch))
        sess.save_checkpoint('latest')

    # logger.info("Training is done, exit.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # logger.info("KeyboardInterrupt, exit.")
        os._exit(1)

# vim: ts=4 sw=4 sts=4 expandtab
