import torch
from util.fairness_utils import ImageFolderWithProtectedAttributes, ImageFolderWithProtectedAttributesModify
import torchvision.transforms as transforms
from torch.utils.data import Subset
import random
import os
import errno
import shutil
import json

def load_dict_as_str(filename):
    with open(filename) as f:
        dict = json.loads(f.read())
        for k in dict.keys():
            for i in range(len(dict[k])):
                dict[k][i] = str(dict[k][i])
    return dict

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

def print_per_gender_count(dataset):

    labels_gender = {}
    proportion_set = [0] * len(dataset.class_to_idx.keys())

    for idx in range(0, len(dataset)):
        label = dataset.imgs[idx][1]
        gender = dataset[idx][3]

        if label not in labels_gender.keys():
            labels_gender[label] = 0

        labels_gender[label] += gender
        proportion_set[label] += 1

    gender_labels = {}
    gender_labels['male'] = []
    gender_labels['female'] = []

    for label in labels_gender.keys():
        if labels_gender[label] > 0:
            gender_labels['male'].append(label)
        else:
            gender_labels['female'].append(label)

    print(len(gender_labels['male']))
    print(len(gender_labels['female']))

    return;


def get_less_data(dataset, men_proportion, women_proportion):
    labels_gender = {}
    proportion_set = [0] * len(dataset.class_to_idx.keys())

    for idx in range(0, len(dataset)):
        label = dataset.imgs[idx][1]
        gender = dataset[idx][3]

        if label not in labels_gender.keys():
            labels_gender[label] = 0

        labels_gender[label] += gender
        proportion_set[label] += 1

    gender_labels = {}
    gender_labels['male'] = []
    gender_labels['female'] = []

    for label in labels_gender.keys():
        if labels_gender[label] > 0:
            gender_labels['male'].append(label)
        else:
            gender_labels['female'].append(label)

    for label in gender_labels['male']:
        #proportion_set[label] = proportion_set[label] * men_proportion
        #new
        l = proportion_set[label] * men_proportion
        if men_proportion != 0:
            if l < 2:
                l = 2
        proportion_set[label] = l

    for label in gender_labels['female']:
        #proportion_set[label] = proportion_set[label] * women_proportion
        l = proportion_set[label] * women_proportion
        if women_proportion != 0:
            if l < 2:
                l = 2
        proportion_set[label] = l

    return proportion_set


def get_desired_women_labels(dataset, proportion):
    labels_gender = {}

    for idx in range(0, len(dataset)):
        label = dataset.imgs[idx][1]
        gender = dataset[idx][3]

        if label not in labels_gender.keys():
            labels_gender[label] = 0

        labels_gender[label] += gender

    gender_labels = {}
    gender_labels['male'] = []
    gender_labels['female'] = []

    for label in labels_gender.keys():
        if labels_gender[label] > 0:
            gender_labels['male'].append(label)
        else:
            gender_labels['female'].append(label)

    male_count = len(gender_labels['male'])
    female_count = len(gender_labels['female'])

    print(male_count, female_count)
    gender_labels['female'] = gender_labels['female'][0:int(female_count * proportion)]

    male_count = len(gender_labels['male'])
    female_count = len(gender_labels['female'])
    print(male_count, female_count)

    return gender_labels


def get_desired_men_labels(dataset, proportion):
    labels_gender = {}

    for idx in range(0, len(dataset)):
        label = dataset.imgs[idx][1]
        gender = dataset[idx][3]

        if label not in labels_gender.keys():
            labels_gender[label] = 0

        labels_gender[label] += gender

    gender_labels = {}
    gender_labels['male'] = []
    gender_labels['female'] = []

    for label in labels_gender.keys():
        if labels_gender[label] > 0:
            gender_labels['male'].append(label)
        else:
            gender_labels['female'].append(label)

    male_count = len(gender_labels['male'])
    female_count = len(gender_labels['female'])

    print(male_count, female_count)
    gender_labels['male'] = gender_labels['male'][0:int(male_count * proportion)]

    male_count = len(gender_labels['male'])
    female_count = len(gender_labels['female'])
    print(male_count, female_count)

    return gender_labels


def balanced_weights(images, nclasses, attr=1):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        # print(item[attr])
        count[item[attr]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        if float(count[i]) != 0:
            weight_per_class[i] = N / float(count[i])
        else:
            weight_per_class[i] = 0
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[attr]]

    return weight


def get_label_count(dataset):
    All_labels = {}

    for idx in range(0, len(dataset)):
        label = dataset.imgs[idx][1]
        if label not in All_labels:
            All_labels[label] = {}

        All_labels[label][dataset.imgs[idx][0]] = 1

    return All_labels


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.orig_dataset = dict()
        self.balanced_max = 0

        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            # print(label)
            if label not in self.dataset:
                self.dataset[label] = list()
                self.orig_dataset[label] = list()

            self.dataset[label].append(idx)
            self.orig_dataset[label].append(idx)

            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            j = 0
            while len(self.dataset[label]) < self.balanced_max:
                # self.dataset[label].append(random.choice(self.dataset[label]))
                self.dataset[label].append(self.orig_dataset[label][j])
                j += 1

                j = j % len(self.orig_dataset[label])

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

        return dataset.imgs[idx][1]

    def __len__(self):
        return self.balanced_max * len(self.keys)


class MenSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.orig_dataset = dict()
        self.balanced_max = 0

        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            gender = dataset[idx][3]

            if gender == 1:  # Men

                if label not in self.dataset:
                    self.dataset[label] = list()

                self.dataset[label].append(idx)
                self.balanced_max = len(self.dataset[label]) \
                    if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            j = 0
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
        # print(dataset[idx])
        return dataset.imgs[idx][1]

    def __len__(self):
        return len(self.dataset.keys())


class WomenSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.orig_dataset = dict()
        self.balanced_max = 0

        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            gender = dataset[idx][3]

            if gender == -1:  # women

                if label not in self.dataset:
                    self.dataset[label] = list()

                self.dataset[label].append(idx)
                self.balanced_max = len(self.dataset[label]) \
                    if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            j = 0
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
        # print(dataset[idx])
        return dataset.imgs[idx][1]

    def __len__(self):
        return len(self.dataset.keys())


class BalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.orig_dataset = dict()
        self.balanced_max = 0

        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            gender = dataset[idx][3]

            if gender == -1:  # women

                if label not in self.dataset:
                    self.dataset[label] = list()

                self.dataset[label].append(idx)
                self.balanced_max = len(self.dataset[label]) \
                    if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            j = 0
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
        # print(dataset[idx])
        return dataset.imgs[idx][1]

    def __len__(self):
        return len(self.dataset.keys())


def prepare_data(args):
    # function prepares data: loads images and prepares dataloaders
    train_transform = transforms.Compose([
        transforms.Resize([int(128 * args.input_size[0] / 112), int(128 * args.input_size[0] / 112)]),
        transforms.RandomCrop([args.input_size[0], args.input_size[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean,
                             std=args.std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize([int(128 * args.input_size[0] / 112), int(128 * args.input_size[0] / 112)]),
        transforms.CenterCrop([args.input_size[0], args.input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean,
                             std=args.std)])

    ###################################################################################################################
    # ======= data, loss, network =======#
    demographic_to_classes = load_dict_as_str(args.demographics)
    classes_to_demographic = {}
    for k in demographic_to_classes.keys():
        for cl in demographic_to_classes[k]:
            if k == 'male':
                classes_to_demographic[cl] = 1
            if k == 'female':
                classes_to_demographic[cl] = -1

    datasets = {}
    datasets['train'] = ImageFolderWithProtectedAttributesModify(args.data_train_root, transform=train_transform, classes_to_demographic = classes_to_demographic)
    datasets['test'] = ImageFolderWithProtectedAttributesModify(args.data_test_root, transform=test_transform, classes_to_demographic = classes_to_demographic)

    print('Old')
    print(len(datasets['train']))
    print(len(datasets['train'].classes))
    print_per_gender_count(datasets['train'])

    if args.modify_identities:

        if args.modify_gender == 'men':
            gender_labels = get_desired_men_labels(datasets['train'], args.proportion)
        else:
            gender_labels = get_desired_women_labels(datasets['train'], args.proportion)
        datasets['train'] = ImageFolderWithProtectedAttributesModify(args.data_train_root, transform=train_transform,
                                                                     gender_labels=gender_labels, classes_to_demographic = classes_to_demographic)

        # change the proportion to 0 for men/ women on new dataset
        # now when I call the function for dataset creation, these proportions are already on new labels

        women = get_less_data(datasets['train'], 0, 1)
        men = get_less_data(datasets['train'], 1, 0)

        if args.proportion != 0 and args.modify_gender == 'men':
            datasets['women'] = ImageFolderWithProtectedAttributesModify(args.data_train_root, transform=train_transform,
                                                                         gender_labels=gender_labels, proportions=women, classes_to_demographic = classes_to_demographic)

        if args.proportion != 0 and args.modify_gender == 'women':
            datasets['men'] = ImageFolderWithProtectedAttributesModify(args.data_train_root, transform=train_transform,
                                                                       gender_labels=gender_labels, proportions=men, classes_to_demographic = classes_to_demographic)

    if args.modify_data:
        if args.modify_gender == 'men':
            proportions = get_less_data(datasets['train'], args.proportion, 1)
            women = get_less_data(datasets['train'], 0, 1)
            men = get_less_data(datasets['train'], args.proportion, 0)
        else:
            proportions = get_less_data(datasets['train'], 1, args.proportion)
            women = get_less_data(datasets['train'], 0, args.proportion)
            men = get_less_data(datasets['train'], 1, 0)


        datasets['train'] = ImageFolderWithProtectedAttributesModify(args.data_train_root, transform=train_transform,
                                                                     proportions=proportions, classes_to_demographic = classes_to_demographic)

        datasets['women'] = ImageFolderWithProtectedAttributesModify(args.data_train_root, transform=train_transform,
                                                                     proportions=women, classes_to_demographic = classes_to_demographic)

        datasets['men'] = ImageFolderWithProtectedAttributesModify(args.data_train_root, transform=train_transform,
                                                                   proportions=men, classes_to_demographic = classes_to_demographic)

        tlc = get_label_count(datasets['train'])
        mlc = get_label_count(datasets['men'])
        wlc = get_label_count(datasets['women'])

        print(len(tlc))
        print(len(mlc))
        print(len(wlc))

        for l in wlc.keys():
            for d in wlc[l].keys(): # the data d in the label l
                if d not in tlc[l].keys(): # check if this data d is in the tlc as well
                    print(l,d)

        for l in mlc.keys():
            for d in mlc[l].keys():  # the data d in the label l
                if d not in tlc[l].keys():  # check if this data d is in the tlc as well
                    print(l, d)


    print('New')
    print(len(datasets['train']))
    print(len(datasets['train'].classes))

    print_per_gender_count(datasets['train'])

    all_sample = datasets['train'].samples
    women_sample = datasets['women'].samples
    men_sample = datasets['men'].samples

    print('Check')
    print(len(all_sample))
    print(len(women_sample))
    print(len(men_sample))

    check = {}
    for inst in all_sample:
        path = inst[0]
        check[path] = 1

    print(len(check.keys()))

    k = 0
    for inst in women_sample:
        path = inst[0]
        if k == 0:
            print('Just check one', path)
        k+=1
        if path not in check.keys():
            print(inst)

    print(k)

    k = 0
    for inst in men_sample:
        path = inst[0]
        if k == 0:
            print('Just check one', path)
        k+=1
        if path not in check.keys():
            print(inst)

    print(k)



    ##################### Test modifications ############

    print('Old')
    print(len(datasets['test']))
    print(len(datasets['test'].classes))

    if args.test_modify_identities:
        if args.test_modify_gender == 'men':
            gender_labels = get_desired_men_labels(datasets['test'], args.test_proportion)
        else:
            gender_labels = get_desired_women_labels(datasets['test'], args.test_proportion)
        datasets['test'] = ImageFolderWithProtectedAttributesModify(args.data_test_root, transform=test_transform,
                                                                    gender_labels=gender_labels, classes_to_demographic = classes_to_demographic)

    if args.test_modify_data:
        if args.test_modify_gender == 'men':
            proportions = get_less_data(datasets['test'], args.test_proportion, 1)
        else:
            proportions = get_less_data(datasets['test'], 1, args.test_proportion)
        datasets['test'] = ImageFolderWithProtectedAttributesModify(args.data_test_root, transform=test_transform,
                                                                    proportions=proportions, classes_to_demographic = classes_to_demographic)

    print('New')
    print(len(datasets['test']))
    print(len(datasets['test'].classes))

    ######################################################

    dataloaders = {}
    train_imgs = datasets['train'].imgs
    weights_train = torch.DoubleTensor(balanced_weights(train_imgs, nclasses=len(datasets['train'].classes)))
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train))
    num_class = len(datasets['train'].classes)
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size,
                                                       sampler=train_sampler, num_workers=args.num_workers,
                                                       drop_last=True)

    dataloaders['naive_train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size,
                                                             shuffle=True, num_workers=args.num_workers, drop_last=True)

    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False,
                                                      num_workers=args.num_workers)

    if args.outer_loader == 'Women':
        print(args.outer_loader)
        dataloaders['outer'] = torch.utils.data.DataLoader(datasets['train'], batch_size=args.outer_batch_size,
                                                           sampler=WomenSampler(datasets['train']),
                                                           num_workers=args.num_workers, drop_last=True)
    elif args.outer_loader == 'Simple':
        dataloaders['outer'] = torch.utils.data.DataLoader(datasets['train'], batch_size=args.outer_batch_size,
                                                           num_workers=args.num_workers, drop_last=True)

    elif args.outer_loader == 'Both':
        print('Both')

        # orig_dataset = ImageFolderWithProtectedAttributes(args.data_train_root, train_transform)

        dataloaders['women'] = torch.utils.data.DataLoader(datasets['women'], batch_size=int(args.outer_batch_size / 2),
                                                           num_workers=args.num_workers, shuffle=True, drop_last=True)
        dataloaders['men'] = torch.utils.data.DataLoader(datasets['men'], batch_size=int(args.outer_batch_size / 2),
                                                         num_workers=args.num_workers, shuffle=True, drop_last=True)

        dataloaders['women_inner'] = torch.utils.data.DataLoader(datasets['women'], batch_size=int(args.outer_batch_size / 2),
                                                           num_workers=args.num_workers, shuffle=True, drop_last=True)
        dataloaders['men_inner'] = torch.utils.data.DataLoader(datasets['men'], batch_size=int(args.outer_batch_size / 2),
                                                         num_workers=args.num_workers, shuffle=True, drop_last=True)

    for k in dataloaders.keys():
        print('Len of {} dataloader is {}'.format(k, len(dataloaders[k])))

    return dataloaders, num_class
