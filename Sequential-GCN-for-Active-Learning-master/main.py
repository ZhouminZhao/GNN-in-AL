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
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from data.sampler import SubsetSequentialSampler
import torchvision.transforms as T
import torchvision.models as models
import argparse
# Custom
import models.resnet as resnet
from models.resnet import vgg11
from models.query_models import LossNet
from train_test import train, test
from load_dataset import load_dataset
from selection_methods import query_samples
from config import *
from torchvision.models import resnet18
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST, SVHN
from sklearn.neighbors import kneighbors_graph

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--lambda_loss", type=float, default=1.2,
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s", "--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n", "--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r", "--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d", "--dataset", type=str, default="cifar100",
                    help="")
parser.add_argument("-e", "--no_of_epochs", type=int, default=200,
                    help="Number of epochs for the active learner")
parser.add_argument("-m", "--method_type", type=str, default="lloss",
                    help="")
parser.add_argument("-c", "--cycles", type=int, default=10,
                    help="Number of active learning cycles")
parser.add_argument("-t", "--total", type=bool, default=False,
                    help="Training on the entire dataset")

args = parser.parse_args()


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


def select_instances(data, model, num_instances=1000):
    similarity_scores = {}
    instances = list(data)

    # Calculate similarity scores using randomly initialized model
    for instance in instances:
        similarity_scores[instance] = model.calculate_similarity(instance)

    # Sort instances based on similarity scores
    sorted_instances = sorted(instances, key=lambda x: similarity_scores[x], reverse=True)

    # Select the top num_instances instances
    selected_instances = sorted_instances[:num_instances]

    return selected_instances


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


def aff_to_adj(x, y=None):
    adj = np.matmul(x, x.transpose())
    adj -= np.eye(adj.shape[0])  # 将对角线元素设置为0而不是-1
    adj_diag = np.sum(adj, axis=1)  # 沿行求和而不是列
    adj_diag[adj_diag == 0] = 1  # 处理度为0的情况，避免除以零
    adj = np.matmul(adj, np.diag(1 / adj_diag))
    return adj


##
# Main
if __name__ == '__main__':

    method = args.method_type
    methods = ['Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss', 'VAAL', 'MinCut', 'Popular', 'Similar']
    datasets = ['cifar10', 'cifar100', 'fashionmnist', 'svhn']
    assert method in methods, 'No method %s! Try options %s' % (method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s' % (args.dataset, datasets)
    '''
    method_type: 'Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss', 'VAAL', 'MinCut', 'Popular', 'Similar'
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
        original_indices = list(range(NUM_TRAIN))

        data_train_loader = DataLoader(data_train, batch_size=BATCH,
                                       sampler=SubsetSequentialSampler(original_indices),
                                       pin_memory=True)

        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            resnet18 = resnet.ResNet18(num_classes=NO_CLASSES).cuda()

        model = {'backbone': resnet18}
        torch.backends.cudnn.benchmark = True

        data_train_features = get_features(model, data_train_loader).cpu().numpy()

        adj = aff_to_adj(data_train_features)

        degrees = np.sum(adj[:NUM_TRAIN, :], axis=0)
        sorted_indices = np.argsort(degrees)

        indices = sorted_indices

        if args.total:
            labeled_set = indices
        else:
            labeled_set = indices[:ADDENDUM]
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

            # Model - create new instance for every cycle so that it resets
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                resnet18 = resnet.ResNet18(num_classes=NO_CLASSES).cuda()

            models = {'backbone': resnet18}
            torch.backends.cudnn.benchmark = True

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

            arg_array = arg.numpy().astype(int)

            # Update the labeled dataset and the unlabeled dataset, respectively
            first10000 = [subset[i] for i in arg_array]
            listf = first10000[-ADDENDUM:]
            labeled_set = np.concatenate([labeled_set, listf])
            listd = first10000[:-ADDENDUM]
            unlabeled_set = np.concatenate([listd, unlabeled_set[SUBSET:]])
            print(len(labeled_set), min(labeled_set), max(labeled_set))
            print(len(unlabeled_set), min(unlabeled_set), max(unlabeled_set))
            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

    results.close()
