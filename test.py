from __future__ import print_function
import os
import os.path
import time
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
from meter import AverageMeter

from logger import Logger
# from video_transforms import *
from transforms import *
from Dataset import MyDataset
from models.p3d_model import P3D199, get_optim_policies
from models.C3D import C3D
from models.i3dpt import I3D
from utils import check_gpu, transfer_model, accuracy

class Testing(object):
    def __init__(self, name_list, num_classes=400, modality='RGB', **kwargs):
        self.__dict__.update(kwargs)

        self.num_classes = num_classes
        self.modality = modality
        self.name_list = name_list

        # Set best precision = 0
        self.best_prec1 = 0
        # init start epoch = 0
        self.start_epoch = 0

        self.checkDataFolder()

        self.loading_model()

        self.test_loader = self.loading_data()

        # run
        self.process()

    def checkDataFolder(self):
        try:
            os.stat('./' + self.model_type + '_' + self.data_set)
        except:
            os.mkdir('./' + self.model_type + '_' + self.data_set)
        self.data_folder = './' + self.model_type + '_' + self.data_set

    # Loading P3D model
    def loading_model(self):

        print('Loading %s model' % (self.model_type))
        if self.model_type == 'C3D':
            self.model = C3D()
        elif self.model_type == 'I3D':
            self.model = I3D(num_classes=400, modality='rgb')
        else:
            self.model = P3D199(pretrained=False, num_classes=400, dropout=self.dropout)


        # Transfer classes
        self.model = transfer_model(model=self.model, model_type=self.model_type, num_classes=self.num_classes)

        # Check gpu and run parallel
        if check_gpu() > 0:
            self.model = torch.nn.DataParallel(self.model).cuda()

        # define loss function (criterion) and optimizer
        if check_gpu() > 0:
            self.criterion = nn.CrossEntropyLoss().cuda()
        else:
            self.criterion = nn.CrossEntropyLoss()

        policies = get_optim_policies(model=self.model, modality=self.modality, enable_pbn=True)

        self.optimizer = optim.SGD(policies, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        file = os.path.join(self.data_folder, 'model_best.pth.tar')
        if os.path.isfile(file):
            print("=> loading checkpoint '{}'".format('model_best.pth.tar'))

            checkpoint = torch.load(file)
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
        size = 160
        if self.model_type == 'C3D':
            size = 112
        if self.model_type == 'I3D':
            size = 224
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        val_transformations = Compose([
            Resize((size, size)),
            ToTensor(),
            normalize
        ])

        test_dataset = MyDataset(
            self.data,
            name_list=self.name_list,
            data_folder="test",
            version="1",
            transform=val_transformations,
            num_frames=self.num_frames
        )

        test_loader = data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=False)

        return test_loader

    # Test
    def process(self):
        acc = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        log_file = os.path.join(self.data_folder, 'test.log')
        logger = Logger('test', log_file)
        # switch to evaluate mode
        self.model.eval()

        start_time = time.clock()
        print("Begin testing")
        for i, (images, labels) in enumerate(self.test_loader):
            if check_gpu() > 0:
                images = images.cuda(async=True)
                labels = labels.cuda(async=True)

            image_var = torch.autograd.Variable(images)
            label_var = torch.autograd.Variable(labels)

            # compute y_pred
            y_pred = self.model(image_var)
            loss = self.criterion(y_pred, label_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(y_pred.data, labels, topk=(1, 5))
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
        logger.info(
            ' * Accuracy {acc.avg:.3f}  Acc@5 {top5.avg:.3f} Loss {loss.avg:.3f}'.format(acc=acc, top5=top5,
                                                                                         loss=losses))




