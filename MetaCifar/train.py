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

def get_val_batch(data_loader, iterator):
    try:
        task_data = iterator.next()
    except StopIteration:
        iterator = iter(data_loader)
        task_data = iterator.next()

    inputs, labels = task_data
    return inputs, labels, iterator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--inner_lr', default=0.01, type=float)
    parser.add_argument('--epochs', default=350, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--meta_step', default=80, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--outer_batch_size', default=30, type=int)
    parser.add_argument('--name_of_notebook', default='meta', type=str)
    parser.add_argument('--dataset_type', default='severe_imbalance', type=str)
    parser.add_argument('--dataset_create', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--seed', default=222, type=int)
    parser.add_argument('--comet_key', default=None, type=str)

    args = parser.parse_args()

    experiment = Experiment(api_key=args.comet_key,
                            project_name="Meta_Const")


    ########## prepare dataset ######

    if args.dataset_type == 'severe_imbalance':
        fractions = np.ones(10) / 1000
    elif args.dataset_type == 'imbalance':
        fractions = 0.001 + np.random.rand(10) / 100

    max_class = random.randint(0, 10)
    fractions[max_class] = 1.0
    if args.dataset_create:
        train_imbalance_create(fractions, args.dataset_type)

    trainloader, train_outer_dataloader, testloader = get_inner_simple_loaders(args)

    ############################################

    ######### setting seeds ##############

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #####################################


    ############ setting training regime ###########

    model = ResNet18().cuda()
    model = torch.nn.DataParallel(model)

    opt_params = model.parameters()
    optimizer = torch.optim.SGD(opt_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_element = nn.CrossEntropyLoss(reduction = 'none').cuda()

    trainiter = iter(trainloader)
    train_outer_iter = iter(train_outer_dataloader)
    testiter = iter(testloader)

    Test_accs = []
    Train_accs = []
    batch_losses = []

    name_of_notebook = args.name_of_notebook + '_' + args.dataset_type
    save_file_folder = './' + name_of_notebook + '/'
    create_folder(save_file_folder)

    ##################################################

    with experiment.train():

        for epoch in range(args.start_epoch, args.epochs):

            experiment.log_current_epoch(epoch)
            Actual_params = []
            model.train()

            inner_loss = 0.0
            outer_loss = 0.0

            inner_accuracy = 0.0
            outer_accuracy = 0.0

            test_accuracy = 0.0

            mat = np.zeros((10, 10), dtype=int)
            inner_loss_element = np.zeros(10, dtype=int)
            outer_loss_element = np.zeros(10, dtype=int)
            test_loss_element = np.zeros(10, dtype=int)

            mat_test = np.zeros((10, 10), dtype=int)

            Actual_params = []
            for name, param in model.named_parameters():
                Actual_params.append(param.data)

            for i, data in enumerate(train_outer_dataloader, 0):

                inputs, labels, trainiter = get_val_batch(trainloader, trainiter)
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                # the fast weights are always θ as we start this loop that is for each batch
                # REPLACE THE MODEL WITH THE ORIGINAL PARAMS

                p = 0
                for name, param in model.named_parameters():
                    param.data = Actual_params[p]
                    p += 1

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss_element = criterion_element(outputs, labels)
                grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                fast_weights = list(map(lambda p: p[1] - args.inner_lr * p[0], zip(grad, model.parameters())))

                # REPLACE THE MODEL WITH THE NEW PARAMS
                p = 0
                for name, param in model.named_parameters():
                    param.data = fast_weights[p]
                    p += 1

                # the fast weights are now θ'
                inner_loss += loss.item()
                max_vals, max_indices = torch.max(outputs, 1)
                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                inner_accuracy += 100 * correct

                for j in range(labels.shape[0]):
                    l = labels[j].item()
                    inner_loss_element[l] += loss_element[j]

                del inputs
                del labels
                del outputs
                del grad

                ####################

                # the fast weights θ' are used to calculate the loss

                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss_element = criterion_element(outputs, labels)

                # accumulate the loss
                batch_losses.append(loss)

                for j in range(labels.shape[0]):
                    l = labels[j].item()
                    outer_loss_element[l] += loss_element[j]

                outer_loss += loss.item()
                max_vals, max_indices = torch.max(outputs, 1)
                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                outer_accuracy += 100 * correct



                # How often should i update the model as limitted gpu
                # even if there are some batches left, we will use it in the next epoch

                if len(batch_losses) > args.meta_step:
                    # print(i)
                    # now we collected the losses on all the batches of 1 epoch lets mean them and update our model

                    # REPLACE THE MODEL WITH THE ORIGINAL PARAMS AS WE WILL NOW GRADS WITH RESPECT TO ORIGINAL PARAMS
                    p = 0
                    for name, param in model.named_parameters():
                        param.data = Actual_params[p]
                        p += 1

                    meta_batch_loss = torch.stack(batch_losses).mean()
                    model.train()
                    optimizer.zero_grad()
                    meta_batch_loss.backward()
                    optimizer.step()
                    del batch_losses

                    # NOW STORE IT SO TILL WE UPDATE THE ORIGINAL PARAMS
                    p = 0
                    for name, param in model.named_parameters():
                        Actual_params[p] = param.data
                        p += 1

                    batch_losses = []

                del outputs
                del fast_weights

            p = 0
            for name, param in model.named_parameters():
                param.data = Actual_params[p]
                p += 1

            scheduler.step()

            #### creating a copy to remove the doubt of mishandling of weights
            #### also to make sure, in now way there is effect of testing data on the original model
            model_test = copy.deepcopy(model)
            model_test = model_test.cuda()

            if args.save:
                ############# save ###########

                save_file = save_file_folder + 'safe' + '.pth'

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_test.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, save_file)


            model.eval()
            model_test.eval()
            ### find the test accuracy ####
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                outputs = model_test(inputs)
                loss_element = criterion_element(outputs, labels)

                max_vals, max_indices = torch.max(outputs, 1)

                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                test_accuracy += 100 * correct

                for j in range(labels.shape[0]):
                    l = labels[j].item()
                    p = max_indices[j].item()
                    mat_test[l][p] += 1
                    test_loss_element[l] += loss_element[j]

            test_accuracy /= len(testloader)

            ### find the train accuracy after each epoch ####
            model_train_accuracy = 0.0
            for i, data in enumerate(train_outer_dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)
                loss_element = criterion_element(outputs, labels)
                max_vals, max_indices = torch.max(outputs, 1)
                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                model_train_accuracy += 100 * correct

                for j in range(labels.shape[0]):
                    l = labels[j].item()
                    p = max_indices[j].item()
                    mat[l][p] += 1

            model_train_accuracy /= len(train_outer_dataloader)



            del model_test

            ############ end save ##############
            inner_loss = inner_loss / len(train_outer_dataloader)
            inner_accuracy = inner_accuracy / len(train_outer_dataloader)
            outer_loss = outer_loss / len(train_outer_dataloader)
            outer_accuracy = outer_accuracy / len(train_outer_dataloader)
            print("##################################################################################################")
            print("Epoch " + str(epoch))
            print(inner_loss)
            print(inner_accuracy)
            print(model_train_accuracy)
            print("#########")
            print(outer_loss)
            print(outer_accuracy)

            print('#########')
            print("Test : ", test_accuracy)

            print('################################################')

            Test_accs.append(test_accuracy)
            Train_accs.append(inner_accuracy)

            ###### write the comet ml ########
            experiment.log_metric("Loss Inner", inner_loss, step=epoch)
            experiment.log_metric("Accuracy Inner", inner_accuracy, step=epoch)
            experiment.log_metric("Accuracy Train Model", model_train_accuracy, step=epoch)

            experiment.log_metric("Loss Outer", outer_loss, step=epoch)
            experiment.log_metric("Accuracy Outer", outer_accuracy, step=epoch)

            experiment.log_metric("Accuracy Test", test_accuracy, step=epoch)


            for l in range(len(mat)):
                corr = mat[l][l]
                total = 0
                for p in range(len(mat[l])):
                    total += mat[l][p]
                acc = corr/total
                experiment.log_metric("Accuracy Train " + str(l), acc, step=epoch)


            for l in range(len(mat_test)):
                corr = mat_test[l][l]
                total = 0
                for p in range(len(mat_test[l])):
                    total += mat_test[l][p]
                acc = corr/total
                experiment.log_metric("Accuracy Test " + str(l), acc, step=epoch)

            for l in range(10):
                experiment.log_metric("Loss Total Inner " + str(l), inner_loss_element[l], step=epoch)
                experiment.log_metric("Loss Total Outer " + str(l), outer_loss_element[l], step=epoch)
                experiment.log_metric("Loss Total Test " + str(l), test_loss_element[l], step=epoch)

    model_test = copy.deepcopy(model)
    model_test = model_test.cuda()
    save_file = save_file_folder + str(epoch) + '.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_test.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, save_file)