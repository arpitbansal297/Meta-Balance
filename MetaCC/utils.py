import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, SVMSMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, AllKNN
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, TensorDataset
import torch.utils.data as data_utils


np.random.seed(2)

class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc.item()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def get_val_batch(data_loader, iterator):
    try:
        task_data = iterator.next()
    except StopIteration:
        iterator = iter(data_loader)
        task_data = iterator.next()

    inputs, labels = task_data
    # print(type(labels))
    # print(labels.shape)

    return inputs, labels, iterator


def getXy(X_train, y_train, method):
    if method == 'SMOTE':
        print(method)
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    elif method == 'BorderlineSMOTE':
        print(method)
        X_train, y_train = BorderlineSMOTE().fit_resample(X_train, y_train)

    elif method == 'SVMSMOTE':
        print(method)
        X_train, y_train = SVMSMOTE().fit_resample(X_train, y_train)

    elif method == 'ADASYN':
        print(method)
        X_train, y_train = ADASYN().fit_resample(X_train, y_train)

    elif method == 'RandomOverSampler':
        print(method)
        X_train, y_train = RandomOverSampler().fit_resample(X_train, y_train)

    elif method == 'RandomUnderSampler':
        print(method)
        X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)

    elif method == 'ClusterCentroids':
        print(method)
        X_train, y_train = ClusterCentroids().fit_resample(X_train, y_train)

    elif method == 'NearMiss':
        print(method)
        X_train, y_train = NearMiss(version=1).fit_resample(X_train, y_train)

    elif method == 'AllKNN':
        print(method)
        X_train, y_train = AllKNN().fit_resample(X_train, y_train)

    elif method == 'SMOTEENN':
        print(method)
        X_train, y_train = SMOTEENN().fit_resample(X_train, y_train)

    elif method == 'Simple':
        print(method)

    else:
        print('None')

    return X_train, y_train


def prepare_data(inner_method, outer_method, args):
    data = pd.read_csv('creditcard.csv')
    scaler = StandardScaler()
    data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Amount', 'Time'], axis=1)

    y = data['Class'].values
    X = data.drop(['Class'], axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=0)

    X_train_inner, y_train_inner = getXy(X_train, y_train, inner_method)
    X_train_outer, y_train_outer = getXy(X_train, y_train, outer_method)

    train_dataset = trainData(torch.FloatTensor(X_train_inner), torch.FloatTensor(y_train_inner))
    train_loader = data_utils.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    train_dataset_outer = trainData(torch.FloatTensor(X_train_outer), torch.FloatTensor(y_train_outer))
    train_loader_outer = data_utils.DataLoader(train_dataset_outer, shuffle=True, batch_size=args.outer_batch_size)

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = data_utils.DataLoader(test_dataset, batch_size=15, shuffle=True)

    return train_loader, train_loader_outer, test_loader


def prepare_data_separate(method, args):
    data = pd.read_csv('creditcard.csv')
    scaler = StandardScaler()
    data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Amount', 'Time'], axis=1)

    y = data['Class']
    X = data.drop(['Class'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    minority_class = 0
    majority_class = 1

    if len(y_train[y_train == 1]) < len(y_train[y_train == 0]):
        minority_class = 1
        majority_class = 0

    print(minority_class, len(y_train[y_train == 1]), len(y_train[y_train == 0]))

    minority_len = len(y_train[y_train == minority_class])
    print(minority_len)

    meta_size = args.meta_size
    meta_len = int(meta_size * minority_len)
    print(meta_len)

    X_meta_minor = pd.DataFrame(X_train[y_train == minority_class][0:meta_len])
    y_meta_minor = pd.DataFrame(y_train[y_train == minority_class][0:meta_len])
    X_meta_major = pd.DataFrame(X_train[y_train == majority_class][0:meta_len])
    y_meta_major = pd.DataFrame(y_train[y_train == majority_class][0:meta_len])
    X_meta = pd.concat([X_meta_major, X_meta_minor])
    y_meta = pd.concat([y_meta_major, y_meta_minor])

    X_train = X_train.drop(X_train[y_train == minority_class][0:meta_len].index)
    y_train = y_train.drop(y_train[y_train == minority_class][0:meta_len].index)
    X_train = X_train.drop(X_train[y_train == majority_class][0:meta_len].index)
    y_train = y_train.drop(y_train[y_train == majority_class][0:meta_len].index)

    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values
    X_meta = X_meta.values
    y_meta = y_meta.values


    X_train_inner, y_train_inner = getXy(X_train, y_train, method)
    X_train_outer, y_train_outer = X_meta, y_meta

    train_dataset = trainData(torch.FloatTensor(X_train_inner), torch.FloatTensor(y_train_inner))
    train_loader = data_utils.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    train_dataset_outer = trainData(torch.FloatTensor(X_train_outer), torch.FloatTensor(y_train_outer))
    train_loader_outer = data_utils.DataLoader(train_dataset_outer, shuffle=True, batch_size=args.outer_batch_size)

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = data_utils.DataLoader(test_dataset, batch_size=15, shuffle=True)

    return train_loader, train_loader_outer, test_loader


def prepare_baseline(method, args):
    data = pd.read_csv('creditcard.csv')
    scaler = StandardScaler()
    data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Amount', 'Time'], axis=1)

    y = data['Class'].values
    X = data.drop(['Class'], axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=0)

    X_train, y_train = getXy(X_train, y_train, method)

    train_dataset = trainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = data_utils.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = data_utils.DataLoader(test_dataset, batch_size=15, shuffle=True)

    return train_loader, test_loader


def prepare_data_meta_weight_net(args):
    data = pd.read_csv('creditcard.csv')
    scaler = StandardScaler()
    data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Amount', 'Time'], axis=1)

    y = data['Class']
    X = data.drop(['Class'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    minority_class = 0
    majority_class = 1

    if len(y_train[y_train == 1]) < len(y_train[y_train == 0]):
        minority_class = 1
        majority_class = 0

    print(minority_class, len(y_train[y_train == 1]), len(y_train[y_train == 0]))

    minority_len = len(y_train[y_train == minority_class])
    print(minority_len)

    meta_size = args.meta_size
    meta_len = int(meta_size * minority_len)
    print(meta_len)

    X_meta_minor = pd.DataFrame(X_train[y_train == minority_class][0:meta_len])
    y_meta_minor = pd.DataFrame(y_train[y_train == minority_class][0:meta_len])
    X_meta_major = pd.DataFrame(X_train[y_train == majority_class][0:meta_len])
    y_meta_major = pd.DataFrame(y_train[y_train == majority_class][0:meta_len])
    X_meta = pd.concat([X_meta_major, X_meta_minor])
    y_meta = pd.concat([y_meta_major, y_meta_minor])

    # print(y_train[y_train == minority_class].index)
    # print(y_train[y_train == majority_class].index)

    X_train = X_train.drop(X_train[y_train == minority_class][0:meta_len].index)
    y_train = y_train.drop(y_train[y_train == minority_class][0:meta_len].index)
    X_train = X_train.drop(X_train[y_train == majority_class][0:meta_len].index)
    y_train = y_train.drop(y_train[y_train == majority_class][0:meta_len].index)

    # print(y_train[y_train == minority_class].index)
    # print(y_train[y_train == majority_class].index)

    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values
    X_meta = X_meta.values
    y_meta = y_meta.values

    train_dataset = trainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = data_utils.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    train_metaset = trainData(torch.FloatTensor(X_meta), torch.FloatTensor(y_meta))
    train_meta_loader = data_utils.DataLoader(train_metaset, shuffle=True, batch_size=args.batch_size)

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = data_utils.DataLoader(test_dataset, batch_size=15, shuffle=True)

    return train_loader, train_meta_loader, test_loader