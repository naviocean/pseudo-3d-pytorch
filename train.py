from __future__ import print_function
import os
import os.path
import shutil
import time
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
from Dataset import MyDataset


class Training(object):
    def __init__(self, name_list, num_classes=400, modality='RGB', **kwargs):
        self.__dict__.update(kwargs)
        self.num_classes = num_classes
        self.modality = modality
        self.name_list = name_list
        # set accuracy avg = 0
        self.count_early_stop = 0
        # Set best precision = 0
        self.best_prec1 = 0
        # init start epoch = 0
        self.start_epoch = 0

        self.checkDataFolder()

        self.loading_model()

        self.train_loader, self.val_loader = self.loading_data()

        # run
        self.processing()

    def check_early_stop(self, accuracy, logger, start_time):
        if self.best_prec1 <= accuracy:
            self.count_early_stop = 0
        else:
            self.count_early_stop += 1

        if self.count_early_stop > self.early_stop:
            print('Early stop')
            end_time = time.time()
            print("--- Total training time %s seconds ---" % (end_time - start_time))
            logger.info("--- Total training time %s seconds ---" % (end_time - start_time))
            exit()

    def checkDataFolder(self):
        try:
            os.stat('./' + self.data_set)
        except:
            os.mkdir('./' + self.data_set)
        self.data_folder = './' + self.data_set

    # Loading P3D model
    def loading_model(self):

        print('Loading P3D model')
        if self.pretrained:
            print("=> using pre-trained model")
            self.model = P3D199(pretrained=True, num_classes=400, dropout=self.dropout)

        else:
            print("=> creating model P3D")
            self.model = P3D199(pretrained=False, num_classes=400, dropout=self.dropout)

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

        # if self.pretrained:
        self.optimizer = optim.SGD(policies, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # else:
        #     self.optimizer = optim.SGD(policies, lr=self.lr, momentum=self.momentum,
        #                                weight_decay=self.weight_decay)

        # optionally resume from a checkpoint
        if self.resume:
            if os.path.isfile(self.resume):
                print("=> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(self.evaluate, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.resume))

        if self.evaluate:
            file_model_best = os.path.join(self.data_folder, 'model_best.pth.tar')
            if os.path.isfile(file_model_best):
                print("=> loading checkpoint '{}'".format('model_best.pth.tar'))
                checkpoint = torch.load(file_model_best)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(self.evaluate, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.resume))

        cudnn.benchmark = True

    # Loading data
    def loading_data(self):
        size = 160

        normalize = video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transformations = video_transforms.Compose([
            video_transforms.RandomResizedCrop(size),
            video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor(),
            normalize])

        val_transformations = video_transforms.Compose([
            video_transforms.Resize((182, 242)),
            video_transforms.CenterCrop(size),
            video_transforms.ToTensor(),
            normalize
        ])

        train_dataset = MyDataset(
            self.data,
            data_folder="train",
            name_list=self.name_list,
            version="1",
            transform=train_transformations,
        )

        val_dataset = MyDataset(
            self.data,
            data_folder="validation",
            name_list=self.name_list,
            version="1",
            transform=val_transformations,
        )

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True)

        val_loader = data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=False)

        return (train_loader, val_loader)

    def processing(self):
        log_file = os.path.join(self.data_folder, 'train.log');

        logger = Logger('train', log_file)

        if self.evaluate:
            self.validate()
            return

        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            self.adjust_learning_rate(epoch)

            # train for one epoch
            self.train(logger, epoch)

            # evaluate on validation set
            prec1 = self.validate(logger)

            # remember best Accuracy and save checkpoint
            is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1, self.best_prec1)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

            self.check_early_stop(prec1, logger, start_time)

        end_time = time.time()
        print("--- Total training time %s seconds ---" % (end_time - start_time))
        logger.info("--- Total training time %s seconds ---" % (end_time - start_time))

    # Training
    def train(self, logger, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (images, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if self.check_gpu() > 0:
                images = images.cuda(async=True)
                target = target.cuda(async=True)
            image_var = torch.autograd.Variable(images)
            label_var = torch.autograd.Variable(target)

            # compute y_pred
            y_pred = self.model(image_var)
            loss = self.criterion(y_pred, label_var)

            # measure accuracy and record loss
            prec1, prec5 = self.accuracy(y_pred.data, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            acc.update(prec1.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, self.epochs, i, len(self.train_loader),
                                                                      batch_time=batch_time, data_time=data_time,
                                                                      loss=losses, top1=top1, top5=top5))

        logger.info('Epoch: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, self.epochs, batch_time=batch_time,
                                                                    data_time=data_time, loss=losses, top1=top1,
                                                                    top5=top5))

    # Validation
    def validate(self, logger):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (images, labels) in enumerate(self.val_loader):
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
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print('TrainVal: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(self.val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

        print(' * Accuracy {acc.avg:.3f}  Acc@5 {top5.avg:.3f}'.format(acc=acc, top5=top5))
        logger.info(' * Accuracy {acc.avg:.3f}  Acc@5 {top5.avg:.3f}'.format(acc=acc, top5=top5))

        return acc.avg

    # save checkpoint to file
    def save_checkpoint(self, state, is_best):
        checkpoint = os.path.join(self.data_folder, 'checkpoint.pth.tar')
        torch.save(state, checkpoint)
        model_best = os.path.join(self.data_folder, 'model_best.pth.tar')
        if is_best:
            shutil.copyfile(checkpoint, model_best)

    # adjust learning rate for each epoch
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = self.weight_decay * param_group['decay_mult']

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
        if self.model_type == 'P3D':
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        elif self.model_type == 'C3D':
            self.model.fc8 = nn.Linear(4096, self.num_classes)
