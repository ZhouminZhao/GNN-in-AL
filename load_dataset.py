import numpy as np
from torch.utils.data import DataLoader, Dataset
from config import *
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST, SVHN
from torchvision.transforms import RandAugment
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from timm.data.mixup import Mixup
from timm.data.dataset import ImageDataset
from timm.data.loader import create_loader
from PIL import Image


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


##

# Data
def load_dataset(dataset):
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    if dataset == 'cifar10':
        data_train = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
        NO_CLASSES = 10
        adden = ADDENDUM
        no_train = NUM_TRAIN
    return data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train
