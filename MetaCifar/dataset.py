import os
import errno
import torchvision
import shutil
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from cutmix.cutmix import CutMix

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
random.seed(222)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if True and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif True and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

def train_imbalance_create(fractions, dataset_type):
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True)

    root = './root_' + dataset_type + '/'

    del_folder(root)
    create_folder(root)
    counts = []

    for i in range(10):
        lable_root = root + str(i) + '/'
        create_folder(lable_root)
        counts.append(0)

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        counts[label] += 1

    for i in range(10):
        fractions[i] = counts[i] * fractions[i]
        counts[i] = 0

    print(fractions)

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        counts[label] += 1
        if counts[label] <= fractions[label]:
            img.save(root + str(label) + '/' + str(idx) + '.png')

    ########## test
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True)

    root = './root_test_' + dataset_type + '/'

    del_folder(root)
    create_folder(root)

    for i in range(10):
        lable_root = root + str(i) + '/'
        create_folder(lable_root)

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        img.save(root + str(label) + '/' + str(idx) + '.png')


def train_imbalance_mwn_create(fractions, meta_count, dataset_type):
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True)

    root = './root_' + dataset_type + "_" + str(meta_count) + '/'
    del_folder(root)
    create_folder(root)
    counts = []

    root_meta = './root_meta_' + dataset_type + "_" + str(meta_count) + '/'
    del_folder(root_meta)
    create_folder(root_meta)
    meta_counts = []

    for i in range(10):
        lable_root = root + str(i) + '/'
        create_folder(lable_root)
        counts.append(0)

        lable_root = root_meta + str(i) + '/'
        create_folder(lable_root)
        meta_counts.append(0)

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        counts[label] += 1

    for i in range(10):
        fractions[i] = counts[i] * fractions[i]
        counts[i] = 0
        meta_counts[i] = 0

    print(fractions)
    # remove the meta_count from each of these
    for i in range(10):
        fractions[i] = fractions[i] - meta_count
        if fractions[i] < 0:
            print("ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR")

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        counts[label] += 1
        if counts[label] <= fractions[label]:
            img.save(root + str(label) + '/' + str(idx) + '.png')
        else:
            meta_counts[label] += 1
            if meta_counts[label] <= meta_count:
                img.save(root_meta + str(label) + '/' + str(idx) + '.png')


    ########## test
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True)

    root = './root_test_' + dataset_type + "_" + str(meta_count) + '/'

    del_folder(root)
    create_folder(root)

    for i in range(10):
        lable_root = root + str(i) + '/'
        create_folder(lable_root)

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        img.save(root + str(label) + '/' + str(idx) + '.png')


def get_loaders(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    root = './root_' + args.dataset_type + '/'
    root_test = './root_test_' + args.dataset_type + '/'

    a = []
    for i in range(10):
        a.append(len(os.listdir(root + str(i) + '/')))

    print('Train Stats')
    print(a)

    a = []
    for i in range(10):
        a.append(len(os.listdir(root_test + str(i) + '/')))

    print('Test Stats')
    print(a)


    #batch_size = 20
    #outer_batch_size = 30

    trainset_outer = torchvision.datasets.ImageFolder(root=root, transform=transform_train)
    train_outer_dataloader = torch.utils.data.DataLoader(trainset_outer, sampler=BalancedBatchSampler(trainset_outer),
                                                         batch_size=args.outer_batch_size, num_workers=1, drop_last=True)

    trainset = torchvision.datasets.ImageFolder(root=root, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=BalancedBatchSampler(trainset),
                                              batch_size=args.batch_size, num_workers=1, drop_last=True)

    testset = torchvision.datasets.ImageFolder(root=root_test, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    return trainloader, train_outer_dataloader, testloader

def get_inner_simple_loaders(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    root = './root_' + args.dataset_type + '/'
    root_test = './root_test_' + args.dataset_type + '/'

    a = []
    for i in range(10):
        a.append(len(os.listdir(root + str(i) + '/')))

    print('Train Stats')
    print(a)

    a = []
    for i in range(10):
        a.append(len(os.listdir(root_test + str(i) + '/')))

    print('Test Stats')
    print(a)


    #batch_size = 20
    #outer_batch_size = 30

    trainset_outer = torchvision.datasets.ImageFolder(root=root, transform=transform_train)
    train_outer_dataloader = torch.utils.data.DataLoader(trainset_outer, sampler=BalancedBatchSampler(trainset_outer),
                                                         batch_size=args.outer_batch_size, num_workers=1, drop_last=True)

    trainset = torchvision.datasets.ImageFolder(root=root, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=1, shuffle=True, drop_last=True)


    testset = torchvision.datasets.ImageFolder(root=root_test, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    return trainloader, train_outer_dataloader, testloader

def get_mwn_loaders(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    root = './root_' + args.dataset_type + "_" + str(args.meta_count) + '/'
    root_meta = './root_meta_' + args.dataset_type + "_" + str(args.meta_count) + '/'
    root_test = './root_test_' + args.dataset_type + "_" + str(args.meta_count) + '/'

    a = []
    for i in range(10):
        a.append(len(os.listdir(root + str(i) + '/')))

    print('Train Stats')
    print(a)

    a = []
    for i in range(10):
        a.append(len(os.listdir(root_test + str(i) + '/')))

    print('Test Stats')
    print(a)

    metaset = torchvision.datasets.ImageFolder(root=root_meta, transform=transform_train)
    train_meta_dataloader = torch.utils.data.DataLoader(metaset, batch_size=args.outer_batch_size, num_workers=1, drop_last=False)

    trainset = torchvision.datasets.ImageFolder(root=root, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=1, shuffle=True, drop_last=True)


    testset = torchvision.datasets.ImageFolder(root=root_test, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    return trainloader, train_meta_dataloader, testloader

def get_loaders_cutmix(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    root = './root_' + args.dataset_type + '/'
    root_test = './root_test_' + args.dataset_type + '/'

    a = []
    for i in range(10):
        a.append(len(os.listdir(root + str(i) + '/')))

    print('Train Stats')
    print(a)

    a = []
    for i in range(10):
        a.append(len(os.listdir(root_test + str(i) + '/')))

    print('Test Stats')
    print(a)


    #batch_size = 20
    #outer_batch_size = 30

    trainset_outer = torchvision.datasets.ImageFolder(root=root, transform=transform_train)
    train_outer_dataloader = torch.utils.data.DataLoader(trainset_outer, batch_size=args.batch_size, num_workers=1, shuffle=True, drop_last=True)

    trainset = torchvision.datasets.ImageFolder(root=root, transform=transform_train)
    trainset = CutMix(trainset, num_class=10, beta=1.0, prob=0.5, num_mix=2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=1, shuffle=True, drop_last=True)

    testset = torchvision.datasets.ImageFolder(root=root_test, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    return trainloader, train_outer_dataloader, testloader


def get_test_loaders(bs):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    root_test = './root_test_severe_imbalance/'

    testset = torchvision.datasets.ImageFolder(root=root_test, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=1)

    return testloader



def get_simple_loaders(args):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    root = './root_' + args.dataset_type + '/'
    root_test = './root_test_' + args.dataset_type + '/'

    a = []
    for i in range(10):
        a.append(len(os.listdir(root + str(i) + '/')))

    print('Train Stats')
    print(a)

    a = []
    for i in range(10):
        a.append(len(os.listdir(root_test + str(i) + '/')))

    print('Test Stats')
    print(a)


    #batch_size = 20
    #outer_batch_size = 30

    trainset_outer = torchvision.datasets.ImageFolder(root=root, transform=transform_train)
    train_outer_dataloader = torch.utils.data.DataLoader(trainset_outer, batch_size=args.outer_batch_size, num_workers=1, shuffle=True, drop_last=True)

    trainset = torchvision.datasets.ImageFolder(root=root, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=1, shuffle=True, drop_last=True)

    testset = torchvision.datasets.ImageFolder(root=root_test, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    return trainloader, train_outer_dataloader, testloader


