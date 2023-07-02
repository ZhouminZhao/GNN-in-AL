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
        if self.dataset_name == "cifar100":
            self.cifar100 = CIFAR100('../cifar100', train=train_flag,
                                     download=True, transform=transf)
        if self.dataset_name == "fashionmnist":
            self.fmnist = FashionMNIST('../fashionMNIST', train=train_flag,
                                       download=True, transform=transf)
        if self.dataset_name == "svhn":
            self.svhn = SVHN('../svhn', split="train",
                             download=True, transform=transf)

    def __getitem__(self, index):
        if self.dataset_name == "cifar10":
            data, target = self.cifar10[index]
        if self.dataset_name == "cifar100":
            data, target = self.cifar100[index]
        if self.dataset_name == "fashionmnist":
            data, target = self.fmnist[index]
        if self.dataset_name == "svhn":
            data, target = self.svhn[index]
        return data, target, index

    def __len__(self):
        if self.dataset_name == "cifar10":
            return len(self.cifar10)
        elif self.dataset_name == "cifar100":
            return len(self.cifar100)
        elif self.dataset_name == "fashionmnist":
            return len(self.fmnist)
        elif self.dataset_name == "svhn":
            return len(self.svhn)


##

# Data
def load_dataset(dataset):
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.RandomApply([RandAugment(num_ops=2, magnitude=9)], p=0.5),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        #T.RandomErasing(p=0.25)
        # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])

    if dataset == 'cifar10':
        data_train = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)

        mixup_fn = Mixup(prob=0.8, switch_prob=1.0, num_classes=10)

        data_tensor = torch.from_numpy(data_train.data)
        images = torch.stack([image.permute(0, 1, 2).float() for image in data_tensor])
        labels = torch.tensor(data_train.targets).long()

        images, labels = mixup_fn(images, labels)
        images_numpy = images.numpy()
        labels_list = labels.tolist()
        data_train.data = np.uint8(images_numpy)
        data_train.targets = labels_list

        random_erasing = T.RandomErasing(p=0.25)
        torch_tensor = torch.from_numpy(data_train.data.transpose((0, 3, 1, 2)))
        erased_tensor = random_erasing(torch_tensor)
        erased_data_train = erased_tensor.numpy().transpose((0, 2, 3, 1))
        data_train.data = erased_data_train

        '''
        # visualization
        original_train_transform = T.Compose([
            T.ToTensor()
        ])
        original_data_train = CIFAR10('../cifar10', train=True, download=True, transform=original_train_transform)

        to_pil = ToPILImage()
        class_labels = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        fig, axes = plt.subplots(10, 2, figsize=(8, 2.5 * 5))

        for i in range(10):
            original_image, label = original_data_train[i]

            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            mean_tensor = torch.tensor(mean)
            std_tensor = torch.tensor(std)
            denormalize = T.Normalize((-mean_tensor / std_tensor).tolist(), (1.0 / std_tensor).tolist())

            image = data_train.data[i]
            #image = denormalize(image)

            #augmented_image, _ = augmented_data_train[i]
            #augmented_image = denormalize(augmented_image)

            original_image = to_pil(original_image)
            image = to_pil(image)
            #augmented_image = to_pil(augmented_image)

            axes[i, 0].imshow(original_image)
            axes[i, 0].axis('off')
            axes[i, 0].set_title('Original: {}'.format(class_labels[label]))

            axes[i, 1].imshow(image)
            axes[i, 1].axis('off')
            axes[i, 1].set_title('Augmented: {}'.format(class_labels[label]))

            #axes[i, 2].imshow(augmented_image)
            #axes[i, 2].axis('off')
            #axes[i, 2].set_title('Augmented: {}'.format(class_labels[label]))

        plt.tight_layout()
        plt.show()  '''

        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
        NO_CLASSES = 10
        adden = ADDENDUM
        no_train = NUM_TRAIN
    elif dataset == 'cifar10im':
        data_train = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)
        # data_unlabeled   = CIFAR10('../cifar10', train=True, download=True, transform=test_transform)
        targets = np.array(data_train.targets)
        # NUM_TRAIN = targets.shape[0]
        classes, _ = np.unique(targets, return_counts=True)
        nb_classes = len(classes)
        imb_class_counts = [500, 5000] * 5
        class_idxs = [np.where(targets == i)[0] for i in range(nb_classes)]
        imb_class_idx = [class_id[:class_count] for class_id, class_count in zip(class_idxs, imb_class_counts)]
        imb_class_idx = np.hstack(imb_class_idx)
        no_train = imb_class_idx.shape[0]
        # print(NUM_TRAIN)
        data_train.targets = targets[imb_class_idx]
        data_train.data = data_train.data[imb_class_idx]
        data_unlabeled = MyDataset(dataset[:-2], True, test_transform)
        data_unlabeled.cifar10.targets = targets[imb_class_idx]
        data_unlabeled.cifar10.data = data_unlabeled.cifar10.data[imb_class_idx]
        data_test = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
        NO_CLASSES = 10
        adden = ADDENDUM
        no_train = NUM_TRAIN
    elif dataset == 'cifar100':
        data_train = CIFAR100('../cifar100', train=True, download=True, transform=train_transform)
        data_unlabeled = MyDataset(dataset, True, test_transform)
        data_test = CIFAR100('../cifar100', train=False, download=True, transform=test_transform)
        NO_CLASSES = 100
        adden = 2000
        no_train = NUM_TRAIN
    elif dataset == 'fashionmnist':
        data_train = FashionMNIST('../fashionMNIST', train=True, download=True,
                                  transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test = FashionMNIST('../fashionMNIST', train=False, download=True,
                                 transform=T.Compose([T.ToTensor()]))
        NO_CLASSES = 10
        adden = ADDENDUM
        no_train = NUM_TRAIN
    elif dataset == 'svhn':
        data_train = SVHN('../svhn', split='train', download=True,
                          transform=T.Compose([T.ToTensor()]))
        data_unlabeled = MyDataset(dataset, True, T.Compose([T.ToTensor()]))
        data_test = SVHN('../svhn', split='test', download=True,
                         transform=T.Compose([T.ToTensor()]))
        NO_CLASSES = 10
        adden = ADDENDUM
        no_train = NUM_TRAIN
    return data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train
