from __future__ import print_function
import argparse
from train import Training
from test import Testing
from ufc101_dataset import UCF101Dataset

parser = argparse.ArgumentParser(description='PyTorch Pseudo-3D fine-tuning')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=75, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')


def main():
    args = parser.parse_args()
    args = vars(args)
    if args['test']:
        Testing(Dataset=UCF101Dataset, num_classes=101, modality='RGB', **args)
    else:
        Training(Dataset=UCF101Dataset, num_classes=101, modality='RGB', **args)


if __name__ == '__main__':
    main()
