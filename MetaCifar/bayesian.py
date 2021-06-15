from comet_ml import Experiment
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import copy
from torch.autograd import Variable
from models import *
from dataset import *

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
random.seed(222)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_location', default='meta', type=str)
    parser.add_argument('--dataset_type', default='severe_imbalance', type=str)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--outer_batch_size', default=30, type=int)

    args = parser.parse_args()

    trainloader, train_outer_dataloader, testloader = get_inner_simple_loaders(args)

    model = ResNet18().cuda()
    model = torch.nn.DataParallel(model)

    opt_params = model.parameters()

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_element = nn.CrossEntropyLoss(reduction='none').cuda()

    name_of_notebook = args.model_location
    save_file = './' + name_of_notebook
    checkpoint = torch.load(save_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    data_cnt = np.zeros(10, dtype=int)

    for k in range(2):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)

            for j in range(labels.shape[0]):
                l = labels[j].item()
                data_cnt[l] += 1

    data_cnt = np.asarray(data_cnt)
    #data_cnt = int(data_cnt/2)
    print(data_cnt)
    total_data = 0
    for dc in data_cnt:
        total_data += dc

    data_cnt = data_cnt/total_data
    data_cnt = 1/data_cnt
    data_cnt = torch.from_numpy(data_cnt)
    print(data_cnt)
    data_cnt = data_cnt.cuda()


    model.eval()
    test_accuracy = 0
    mat_test = np.zeros((10, 10), dtype=int)
    m = nn.Softmax(dim=1)

    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        outputs = m(outputs)

        if i==0:
            print(outputs[1])

        for l in range(outputs.shape[0]):
            outputs[l] = outputs[l] * data_cnt
        if i==0:
            print(outputs[1])

        max_vals, max_indices = torch.max(outputs, 1)

        correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
        test_accuracy += 100 * correct

        for j in range(labels.shape[0]):
            l = labels[j].item()
            p = max_indices[j].item()
            mat_test[l][p] += 1

    test_accuracy /= len(testloader)
    print(test_accuracy)
    for l in range(len(mat_test)):
        corr = mat_test[l][l]
        total = 0
        for p in range(len(mat_test[l])):
            total += mat_test[l][p]
        acc = corr / total
        print(acc)








