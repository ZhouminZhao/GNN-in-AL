'''
GCN Active Learning
'''

# Python
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from data.sampler import SubsetSequentialSampler

# Custom
import resnet as resnet
from train_test import train, test
from load_dataset import load_dataset
from selection_methods import query_samples
# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from config import *
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST, SVHN
from sklearn.neighbors import kneighbors_graph
from rsgnn import representation_selection
import matplotlib.pyplot as plt
from kcenterGreedy import kCenterGreedy
from selection_methods import get_kcg

import common_args

args = common_args.parser.parse_args()


class MyDataset(Dataset):
    def __init__(self, dataset_name, train_flag, transf):
        self.dataset_name = dataset_name
        if self.dataset_name == "cifar10":
            self.cifar10 = CIFAR10('../cifar10', train=train_flag,
                                   download=True, transform=transf)

    def __getitem__(self, index):
        if self.dataset_name == "cifar10":
            data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        if self.dataset_name == "cifar10":
            return len(self.cifar10)


def get_features(models, train_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _ in train_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features  # .detach().cpu().numpy()
    return feat


def get_features_centers(models, train_loader):
    models['backbone'].eval()
    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
        features = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _, _ in train_loader:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                inputs = inputs.cuda()
                _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features  # .detach().cpu().numpy()
    return feat


def aff_to_adj(x, y=None):
    adj = np.matmul(x, x.transpose())
    adj -= np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=1)
    adj_diag[adj_diag == 0] = 1
    adj = np.matmul(adj, np.diag(1 / adj_diag))
    return adj


##
# Main
if __name__ == '__main__':

    method = args.method_type
    methods = ['Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss', 'VAAL', 'Popular', 'RSGNN']
    datasets = ['cifar10', 'cifar100', 'fashionmnist', 'svhn']
    assert method in methods, 'No method %s! Try options %s' % (method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s' % (args.dataset, datasets)
    '''
    method_type: 'Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','Popular', 'RSGNN'
    '''
    results = open(
        'results_' + str(args.method_type) + "_" + args.dataset + '_main' + str(args.cycles) + str(args.total) + '.txt',
        'w')
    print("Dataset: %s" % args.dataset)
    print("Method type:%s" % method)
    if args.total:
        TRIALS = 1
        CYCLES = 1
    else:
        CYCLES = args.cycles
    for trial in range(TRIALS):
        # Load training and testing dataset
        data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args.dataset)
        # Don't predefine budget size. Configure it in the config.py: ADDENDUM = adden
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)

        if args.total:
            labeled_set = indices
        else:
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                resnet18 = resnet.ResNet18(num_classes=NO_CLASSES).cuda()

            models = {'backbone': resnet18}
            torch.backends.cudnn.benchmark = True

            _, rep_ids = representation_selection(models=models, subset=None, select_round='first', labeled_set=[], lbl=None, nlbl=None, cycle=0)
            labeled_set = rep_ids

            unlabeled_set = [x for x in indices if x not in labeled_set]

        train_loader = DataLoader(data_train, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True, drop_last=True)
        test_loader = DataLoader(data_test, batch_size=BATCH)
        dataloaders = {'train': train_loader, 'test': test_loader}

        for cycle in range(CYCLES):

            # Randomly sample 10000 unlabeled data points
            if not args.total:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:SUBSET]

            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                       momentum=MOMENTUM, weight_decay=WDECAY)

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}

            # Training and testing
            train(models, method, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs, EPOCHL)
            acc = test(models, EPOCH, method, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc))
            np.array([method, trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
            results.write("\n")

            if cycle == (CYCLES - 1):
                # Reached final training cycle
                print("Finished.")
                break

            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args)
            print(len(arg), min(arg), max(arg))
            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][:ADDENDUM].numpy())
            print(len(labeled_set), min(labeled_set), max(labeled_set))

            listd = list(torch.tensor(subset)[arg][ADDENDUM:].numpy())
            unlabeled_set = listd + unlabeled_set[SUBSET:]
            print(len(unlabeled_set), min(unlabeled_set), max(unlabeled_set))
            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

    results.close()
