import torch
import torch.nn as nn
from models.i3dpt import Unit3Dpy, I3D


# get accuracy from y pred
def accuracy(y_pred, y_actual, topk=(1,)):
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


def check_gpu():
    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    return num_gpus


def transfer_model(model, model_type='P3D', num_classes=101):
    if model_type == 'P3D':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_type == 'C3D':
        model.fc8 = nn.Linear(4096, num_classes)
    elif model_type == "I3D":
        conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        model.conv3d_0c_1x1 = conv3d_0c_1x1
    return model


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr