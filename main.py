from __future__ import print_function
import argparse
from train import Training
from test import Testing

parser = argparse.ArgumentParser(description='PyTorch Pseudo-3D fine-tuning')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--data-set', default='UCF101', const='UCF101', nargs='?', choices=['UCF101', 'Breakfast', 'merl'])
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--early-stop', default=10, type=int, metavar='N', help='number of early stopping')
parser.add_argument('--epochs', default=75, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--dropout', default=0.5, type=float, metavar='M', help='dropout')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--random', dest='random', action='store_true', help='random pick image')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--model-type', default='P3D', choices=['P3D', 'C3D', 'I3D'], help='which model to run the code')
parser.add_argument('--num-frames', default=16, type=int, metavar='N', help='number frames per clip')
parser.add_argument('--log-visualize', default='./runs', type=str, metavar='PATH', help='tensorboard log')

def main():
    args = parser.parse_args()
    args = vars(args)

    if args['data_set'] == 'UCF101':
        print('UCF101 data set')
        name_list = 'ucfTrainTestlist'
        num_classes = 101
    elif args['data_set'] == 'Breakfast':
        print("breakfast data set")
        num_classes = 37
        name_list = 'breakfastTrainTestList'
    else:
        print('Merl data set')
        num_classes = 5
        name_list = 'merlTrainTestList'

    if args['test']:
        Testing(name_list=name_list, num_classes=num_classes, modality='RGB', **args)
    else:
        Training(name_list=name_list, num_classes=num_classes, modality='RGB', **args)


if __name__ == '__main__':
    main()
