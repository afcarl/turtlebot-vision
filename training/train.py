#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import os
import sys
import json

import numpy as np

from code.models import CNN1
from code.dataset import NavigationDataset
import time

#Parse arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', '-g', type=int, nargs="*", default=[0, 1, 2, 3], help=u"GPU IDs. CPU is -1")
parser.add_argument('--gpu', '-g', type=int, default=0, help=u"GPU IDs. CPU is -1")
parser.add_argument('--freeze', type=int, default=1, help=u"Freeze pretrained or not. 0 is no, 1 is yes.")

parser.add_argument("--savedir",default="./experiments/ex1", type=str, help=u"The directory to save models and log")
parser.add_argument('--resume',default="", type=str,help='path to the saved model')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--lr-decay",default=0, type=int, help=u"decrese learning rate")
parser.add_argument("--epochs",default=10, type=int, help=u"the number of iterations")
parser.add_argument("--batch",default=32, type=int, help=u"mini batchsize")
parser.add_argument('--w', type=int, default=224, help='input width')
parser.add_argument('--h', type=int, default=224, help='input height')

parser.add_argument('--mode',required = True, type=str,help='regression or classify')

parser.add_argument('--equalsample',default = 0, type=int ,help='1 is true. default is 0 (False)')


args = parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy_func(output, target):
    batch_size = target.size(0)
    prediction = output.data.max(1)[1]   # first column has actual prob.
    accuracy = prediction.eq(target.data)
    accuracy = accuracy.float().sum()/batch_size
    return accuracy

def validate(val_loader, model, criterion,args):
    print("evaluating")
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = Variable(input, volatile=True)
        target_var = Variable(target, volatile=True)
        if args.gpu >= 0:
            input_var, target_var = input_var.cuda(), target_var.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        if args.mode == "classify":
            accuracy = accuracy_func(output, target_var)
            accuracies.update(accuracy, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if args.mode == "classify":
        with open(args.savedir+"/acc_val_history.txt", "a") as f:
            f.write(str(accuracies.avg)+'\n') 
    with open(args.savedir+"/loss_val_history.txt", "a") as f:
        f.write(str(losses.avg)+'\n') 
    print('Loss: {:.9f}\tAccuracy: {:.3f}'.format(losses.avg, accuracies.avg))

    #back to training mode
    model.train()

    return None

os.environ['CUDA_VISIBLE_DEVICES'] = ""
if args.gpu >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

#save dir
if not os.path.isdir(args.savedir):
    os.makedirs(args.savedir)
    print("made the save directory",args.savedir)

#save args as dict
with open(os.path.join(args.savedir,"args.json"), 'w') as f:
    json.dump( vars(args), f, sort_keys=True, indent=4)

#Prepare Data
print("loading training data")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Scale(256),
    transforms.Lambda(lambda x : x.resize([args.w,args.h])),
    transforms.ToTensor(),
    normalize,
])

descritize = args.mode == "classify"
train_dataset = NavigationDataset(image_dirs=["./data/turtlebot1/capture_train1/","./data/turtlebot1/capture_train2/","./data/turtlebot1/capture_train3/","./data/turtlebot1/capture_train4/"],transform = transform, descritize=descritize)
if args.equalsample:
    #taken from https://discuss.pytorch.org/t/how-to-prevent-overfitting/1902/13
    from collections import Counter
    train_counter=Counter([item[1] for item in train_dataset.all_data])
    class_sample_count = [train_counter[i] for i in range(len(train_counter))]
    prob = 1.0 / np.array(class_sample_count)

    weights = np.zeros(len(train_dataset))
    for index in range(len(train_dataset)):
        weights[index] = prob[train_dataset.all_data[index][1]]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch, sampler = sampler, num_workers=32)
else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=32)

test_dataset = NavigationDataset(image_dirs=["./data/turtlebot1/capture_test/"],transform = transform,descritize=descritize)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=False, num_workers=32)


# load model
if args.mode == 'regression':
    model = CNN1(output_dim=1,regression=True)
    loss_func = nn.MSELoss()
if args.mode == 'classify':
    model = CNN1(output_dim=3,regression=False)
    loss_func = nn.CrossEntropyLoss()

# define optimizer
optimizer = torch.optim.Adam(model.classifier.parameters(), weight_decay=args.weight_decay)

if args.gpu >= 0:
    model = model.cuda()

if os.path.isdir(args.savedir):
    print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort."%args.savedir)
    time.sleep(5)
    os.system('rm -rf %s/'%args.savedir)
os.makedirs(args.savedir)
print("made the log directory",args.savedir)

model = model.train()

# def adjust_learning_rate(optimizer, epoch,k = 1):
#     """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
#     lr = args.lr * (0.1 ** (epoch // k))
#     optimizer = torch.optim.SGD(model.get_config_optim(lr, args.lrp),
#                                 lr=lr,
#                                 momentum=args.momentum,
#                                 weight_decay=args.weight_decay)

i = 0
for epoch in range(args.epochs):
    if args.lr_decay > 0:
        adjust_learning_rate(optimizer, epoch)
    for data, target in train_loader:
        data, target = Variable(data), Variable(target)
        if args.gpu >= 0:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()    # calc gradients
        optimizer.step()   # update gradients

        accuracy = 0.0
        if args.mode == "classify":
            accuracy = accuracy_func(output, target)
            with open(args.savedir+"/acc_history.txt", "a") as f:
                f.write(str(accuracy)+'\n') 
        with open(args.savedir+"/loss_history.txt", "a") as f:
            f.write(str(loss.data[0])+'\n') 
        print('Train Step: {}\tEpoch: {}\tLoss: {:.9f}\tAccuracy: {:.5f}'.format(i,epoch, loss.data[0], accuracy))
        i += 1

    validate(test_loader, model, loss_func,args)

    torch.save(model.state_dict(), args.savedir + "/model%d"%epoch)