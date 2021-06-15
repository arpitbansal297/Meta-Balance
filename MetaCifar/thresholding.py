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

    args = parser.parse_args()

    testloader = get_test_loaders(10)

    model = ResNet18().cuda()
    model = torch.nn.DataParallel(model)

    opt_params = model.parameters()

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_element = nn.CrossEntropyLoss(reduction='none').cuda()

    name_of_notebook = args.model_location
    save_file = './' + name_of_notebook

    checkpoint = torch.load(save_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_accuracy = 0

    experiment = Experiment(api_key='7WMavViIUHPfi6za4DlEw0yZI',
                            project_name="Meta_Const")

    step_size = 0.00001
    m = nn.Softmax(dim=1)
    with experiment.train():

        for step in range(1000):
            step_val = 1 - step * step_size
            mat_test = np.zeros((10, 10), dtype=int)
            test_loss_element = np.zeros(10, dtype=int)

            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)
                outputs = m(outputs)

                add = torch.ones_like(outputs)
                for l in range(add.shape[0]):
                    add[l][1] = 0

                add = add * step_val

                if i == 0:
                    print(outputs[0])

                outputs = outputs + add # adding

                if i == 0:
                    print(outputs[0])


                max_vals, max_indices = torch.max(outputs, 1)

                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                test_accuracy += 100 * correct

                for j in range(labels.shape[0]):
                    l = labels[j].item()
                    p = max_indices[j].item()
                    mat_test[l][p] += 1

            test_accuracy /= len(testloader)

            experiment.log_metric("Accuracy Test", test_accuracy, step)
            for l in range(len(mat_test)):
                corr = mat_test[l][l]
                total = 0
                for p in range(len(mat_test[l])):
                    total += mat_test[l][p]
                acc = corr / total
                experiment.log_metric("Accuracy Test " + str(l), acc, step=step)






