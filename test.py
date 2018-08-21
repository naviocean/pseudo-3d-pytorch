from __future__ import print_function
import os
import os.path
import time
import logging
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
from meter import AverageMeter
from lib.p3d_model import P3D199, get_optim_policies
from logger import Logger
import video_transforms


class Testing(object):
    def __init__(self, Dataset, num_classes=400, modality='RGB', **kwargs):
        self.__dict__.update(kwargs)

        self.num_classes = num_classes
        self.modality = modality
        self.Dataset = Dataset

        # Set best precision = 0
        self.best_prec1 = 0
        # init start epoch = 0
        self.start_epoch = 0

        self.loading_model()

        self.test_loader = self.loading_data()

        # run
        self.test()

    # Loading P3D model
    def loading_model(self):
        # Loading P3D model
        if self.pretrained:
            print("=> using pre-trained model")
            self.model = P3D199(pretrained=True, num_classes=400)

        else:
            print("=> creating model P3D")
            self.model = P3D199(pretrained=False, num_classes=400)

        # Transfer classes
        self.transfer_model()

        # Check gpu and run parallel
        if self.check_gpu() > 0:
            self.model = torch.nn.DataParallel(self.model).cuda()

        # define loss function (criterion) and optimizer
        if self.check_gpu() > 0:
            self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            self.criterion = nn.CrossEntropyLoss()

        policies = get_optim_policies(model=self.model, modality=self.modality, enable_pbn=True)

        self.optimizer = optim.SGD(policies, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        if os.path.isfile('model_best.pth.tar'):
            print("=> loading checkpoint '{}'".format('model_best.pth.tar'))
            checkpoint = torch.load('model_best.pth.tar')
            self.start_epoch = checkpoint['epoch']
            self.best_prec1 = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded model best ")
        else:
            print("=> no model best found at ")
            exit()

        cudnn.benchmark = True

    # Loading data
    def loading_data(self):

        normalize = video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        val_transformations = video_transforms.Compose([
            video_transforms.Resize((182, 242)),
            video_transforms.CenterCrop(160),
            video_transforms.ToTensor(),
            normalize
        ])

        test_dataset = self.Dataset(
            self.data,
            data_folder="test",
            version="1",
            transform=val_transformations
        )

        test_loader = data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False)
        return test_loader

    # Test
    def test(self):
        acc = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        logger = Logger('test', 'test.log')
        # switch to evaluate mode
        self.model.eval()

        start_time = time.clock()
        print("Begin testing")
        for i, (images, labels) in enumerate(self.test_loader):
            if self.check_gpu() > 0:
                images = images.cuda(async=True)
                labels = labels.cuda(async=True)

            image_var = torch.autograd.Variable(images)
            label_var = torch.autograd.Variable(labels)

            # compute y_pred
            y_pred = self.model(image_var)
            loss = self.criterion(y_pred, label_var)

            # measure accuracy and record loss
            prec1, prec5 = self.accuracy(y_pred.data, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            acc.update(prec1.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

            if i % self.print_freq == 0:
                print('TestVal: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(self.test_loader), loss=losses, top1=top1, top5=top5))

        print(
            ' * Accuracy {acc.avg:.3f}  Acc@5 {top5.avg:.3f} Loss {loss.avg:.3f}'.format(acc=acc, top5=top5,
                                                                                         loss=losses))

        end_time = time.clock()
        print("Total testing time %.2gs" % (end_time - start_time))
        logger.info("Total testing time %.2gs" % (end_time - start_time))
        logging.info(
            ' * Accuracy {acc.avg:.3f}  Acc@5 {top5.avg:.3f} Loss {loss.avg:.3f}'.format(acc=acc, top5=top5,
                                                                                         loss=losses))
        return

    # get accuracy from y pred
    def accuracy(self, y_pred, y_actual, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = y_actual.size(0)

        _, pred = y_pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res

    def check_gpu(self):
        num_gpus = 0
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
        return num_gpus

    def transfer_model(self):
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
