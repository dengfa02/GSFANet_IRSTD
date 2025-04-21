import numpy as np
from torch.nn import init

from utils.data import *
from utils.metric_basic import *
from argparse import ArgumentParser
import torch
import torch.utils.data as Data
from model.GSFANet import *
from model.loss import *
from torch.optim import Adagrad
from torch.optim import lr_scheduler
from utils.warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import os.path as osp
import os
import time
import matplotlib.pyplot as plt
import cv2


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')
    parser.add_argument('--dataset-dir', type=str, default='../../../../data/dcy/')
    parser.add_argument('--dataset', type=str, default='NUDT-SIRST',
                        help='dataset name:  NUDT-SIRST(50:50/256), IRSTD-1k(80:20/512), NUAA-SIRST(50:50/256)')
    parser.add_argument('--root', type=str, default='../../../../data/dcy')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1500, help='bigmodule1000')
    parser.add_argument('--lr', type=float, default=0.001, help='adam0.001,adagrad0.05')
    parser.add_argument('--warm-epoch', type=int, default=10)
    parser.add_argument('--ngpu', type=int, default=0, help='GPU number')

    parser.add_argument('--base-size', type=int, default=None)  # IRSTD-1K 512
    parser.add_argument('--crop-size', type=int, default=None)  # IRSTD-1K 512
    parser.add_argument('--multi-gpus', type=bool, default=False)
    parser.add_argument('--if-checkpoint', type=bool, default=False)

    parser.add_argument('--mode', type=str, default='train', help='train or test')  # The test can be conducted by directly modifying the parameters.
    parser.add_argument('--weight-path', type=str, default='', help='path to weights')

    args = parser.parse_args()
    # 根据dataset自动设置尺寸，无需手动改
    if args.dataset == 'IRSTD-1k':
        args.base_size = 512 if args.base_size is None else args.base_size
        args.crop_size = 512 if args.crop_size is None else args.crop_size
    elif args.dataset in ['NUAA-SIRST', 'NUDT-SIRST']:
        args.base_size = 256 if args.base_size is None else args.base_size
        args.crop_size = 256 if args.crop_size is None else args.crop_size
    else:
        args.base_size = 256 if args.base_size is None else args.base_size
        args.crop_size = 256 if args.crop_size is None else args.crop_size
    return args


def seed_pytorch(seed=50):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_pytorch(42)  # 与基准一致


class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'

        self.args = args
        self.start_epoch = 0
        self.mode = args.mode
        self.warm_epoch = args.warm_epoch

        trainset = TrainDataset(args, mode='train')
        valset = TrainDataset(args, mode='val')

        self.train_loader = Data.DataLoader(trainset, args.batch_size, shuffle=True, drop_last=False, num_workers=8,
                                            pin_memory=True)
        self.val_loader = Data.DataLoader(valset, 1, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)

        device = torch.device(f'cuda:{args.ngpu}' if torch.cuda.is_available() and
                                                     args.ngpu < torch.cuda.device_count() else 'cpu')
        self.device = device

        model = GSFANet(args.crop_size, 1)

        if args.multi_gpus:
            if torch.cuda.device_count() > 1:
                print('use ' + str(torch.cuda.device_count()) + ' gpus')
                model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(self.device)
        self.model = model

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        scheduler_cosine = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs - self.warm_epoch,
                                                          eta_min=1e-5)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=self.warm_epoch,
                                                after_scheduler=scheduler_cosine)

        self.loss_fun = AdaFocalLoss()
        self.PD_FA = PD_FA()
        self.mIoU = mIoU()
        self.ROC = ROCMetric(1, 10)
        self.F_metric = F_metric(nclass=1)
        self.best_fmeasure = 0
        self.best_iou = 0
        self.best_fa = 1e3

        if args.mode == 'train':
            if args.if_checkpoint:
                check_folder = 'weight/'
                checkpoint = torch.load(check_folder + '/checkpoint.pkl')
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_iou = checkpoint['iou']
                self.save_folder = check_folder
            else:
                self.save_folder = 'weight/%s/GSFANet-%s' % (
                    args.dataset, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
                if not osp.exists(self.save_folder):
                    os.makedirs(self.save_folder)
        if args.mode == 'test':
            weight = torch.load(args.weight_path, map_location=f'cuda:{args.ngpu}')
            self.model.load_state_dict(weight,
                                       strict=False)  # checkpoint continue training: weight['state_dict'] test:weight
            self.warm_epoch = -1

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        tag = True
        for i, (data, mask) in enumerate(tbar):

            data = data.to(self.device)
            labels = mask.to(self.device)

            masks, pred = self.model(data, tag)
            loss = 0

            loss = loss + self.loss_fun(pred, labels)
            for j in range(len(masks)):
                if j > 0:
                    labels = F.interpolate(labels, scale_factor=0.5, mode='nearest')
                loss = loss + self.loss_fun(masks[j], labels)  # , self.warm_epoch, epoch

            loss = loss / (len(masks) + 1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), pred.size(0))
            tbar.set_description(
                'Epoch %d, lr:%f, loss %.4f' % (
                    epoch, trainer.optimizer.param_groups[0]['lr'],
                    losses.avg))

        self.scheduler.step(epoch)

    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        self.F_metric.reset()
        # self.ROC.reset()
        tbar = tqdm(self.val_loader)
        tag = False
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):
                data = data.to(self.device)
                mask = mask.to(self.device)
                h, w = data.size()[2], data.size()[3]

                _, pred = self.model(data, tag)

                self.mIoU.update((pred.sigmoid() > 0.5).cpu(), mask.cpu())
                self.PD_FA.update((pred[0, 0, :, :].sigmoid() > 0.5).cpu(), mask[0, 0, :, :].cpu(),
                                  (h, w))
                self.F_metric.update((pred.sigmoid() > 0.5).cpu(), mask.cpu())
                results = self.mIoU.get()

                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, results[1]))
            results1 = self.mIoU.get()
            results2 = self.PD_FA.get()
            prec, recall, fmeasure = self.F_metric.get()

            if self.mode == 'train':
                if fmeasure > self.best_fmeasure:
                    self.best_fmeasure = fmeasure
                    torch.save(self.model.state_dict(),
                               self.save_folder + '/Epoch%s-weight-%s-bestf%s.pkl' % (
                                   epoch, args.dataset, self.best_fmeasure))  # 训练时最优模型保存
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:05d}\t - IoU {:.5f}\t - F1 {:.5f}\t - PD {:.5f}\t - FA {:.5f}\n'.
                                format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                       epoch, results1[1], self.best_fmeasure, results2[0], results2[1] * 1000000))
                if results1[1] > self.best_iou:
                    self.best_iou = results1[1]

                    torch.save(self.model.state_dict(),
                               self.save_folder + '/weight-%s.pkl' % (args.dataset))  # 训练时最优模型保存
                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:05d}\t - IoU {:.5f}\t - F1 {:.5f}\t - PD {:.5f}\t - FA {:.5f}\n'.
                                format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                       epoch, self.best_iou, fmeasure, results2[0], results2[1] * 1000000))

                all_states = {"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                              "epoch": epoch, "fa": self.best_fa}
                torch.save(all_states, self.save_folder + '/checkpoint.pkl')
            elif self.mode == 'test':
                print("pixAcc, mIoU:\t" + str(results1))
                print("PD, FA:\t" + str(results2))
                print('F_measure:\t' + str(fmeasure))


if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)

    if trainer.mode == 'train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            if epoch >= args.epochs / 2:
                trainer.test(epoch)
    else:
        trainer.test(1)
