import matplotlib.pyplot as plt
import itertools
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, SVMSMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, AllKNN
from imblearn.combine import SMOTEENN

from torch.utils.data import Dataset, TensorDataset
import torch.utils.data as data_utils
import torch
import numpy as np
np.random.seed(2)


class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
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
    df = pd.read_csv(args.data_loc)
    X = df.iloc[:, :-1].drop(columns='purpose').values  # extracting features
    y = df.iloc[:, -1].values  # extracting labels

    sc = StandardScaler()
    X = sc.fit_transform(X)

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