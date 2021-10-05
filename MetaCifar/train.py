from comet_ml import Experiment
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import copy
from torch.autograd import Variable
from models import ResNet18
from dataset import *
from meta_weight_net_model import ResNet18Meta, VNet
import torch.nn.functional as F

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


def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_model():
    model = ResNet18Meta(num_classes=10)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


def CB_loss(labels, logits, samples_per_cls, no_of_classes, beta):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    labels_one_hot = labels
    labels_one_hot = F.one_hot(labels_one_hot.to(torch.int64), no_of_classes).float()

    weights = torch.tensor(weights).float().cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)

    #weights = weights.repeat(1,no_of_classes)

    #print(logits)
    #print(labels)
    #print(weights)

    criterion = nn.CrossEntropyLoss(reduction = 'none').cuda()
    cb_loss = criterion(input=logits, target=labels)
    cb_loss = cb_loss * weights
    return cb_loss.sum()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--inner_lr', default=0.01, type=float)
    parser.add_argument('--epochs', default=350, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--meta_step', default=80, type=int)
    parser.add_argument('--meta_count', default=1, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--outer_batch_size', default=30, type=int)
    parser.add_argument('--name_of_notebook', default='meta', type=str)
    parser.add_argument('--dataset_type', default='severe_imbalance', type=str)
    parser.add_argument('--dataset_create', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--seed', default=222, type=int)
    parser.add_argument('--comet_key', default=None, type=str)
    parser.add_argument('--method', default='metabalance', type=str)
    parser.add_argument('--loss_reweight_beta', default=0.9999, type=float)


    args = parser.parse_args()
    print(args)

    if args.method == 'metabalance':

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
        for epoch in range(args.start_epoch, args.epochs):

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
            print("Inner")
            print(inner_loss)
            print(inner_accuracy)
            print(model_train_accuracy)
            print("#########")
            print("Outer")
            print(outer_loss)
            print(outer_accuracy)
            print('#########')
            print("Test : ", test_accuracy)

            print('################################################')

            Test_accs.append(test_accuracy)
            Train_accs.append(inner_accuracy)


            for l in range(len(mat)):
                corr = mat[l][l]
                total = 0
                for p in range(len(mat[l])):
                    total += mat[l][p]
                acc = corr/total
                print("Accuracy Train " + str(l), acc)


            for l in range(len(mat_test)):
                corr = mat_test[l][l]
                total = 0
                for p in range(len(mat_test[l])):
                    total += mat_test[l][p]
                acc = corr/total
                print("Accuracy Test " + str(l), acc)

        model_test = copy.deepcopy(model)
        model_test = model_test.cuda()
        save_file = save_file_folder + " " + args.method + " " + str(epoch) + " " + str(args.seed) + '.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_test.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, save_file)


###################################################

    elif args.method == 'metaweightnet':

        if args.dataset_type == 'severe_imbalance':
            fractions = np.ones(10) / 1000
        elif args.dataset_type == 'imbalance':
            fractions = 0.001 + np.random.rand(10) / 100

        max_class = random.randint(0, 10)
        fractions[max_class] = 1.0

        root = './root_' + args.dataset_type + "_" + str(args.meta_count) + '/'
        isData = False #os.path.isfile(root)

        if args.dataset_create and (not isData):
            train_imbalance_mwn_create(fractions, args.meta_count, args.dataset_type)

        trainloader, train_meta_loader, testloader = get_mwn_loaders(args)

        model = build_model()
        vnet = VNet(1, 100, 1).cuda()

        optimizer_model = torch.optim.SGD(model.params(), args.lr,
                                          momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3,
                                          weight_decay=1e-4)

        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer_model, epoch)
            train_loss = 0
            meta_loss = 0
            train_accuracy = 0
            meta_accuracy = 0
            test_accuracy = 0

            model.train()
            train_meta_loader_iter = iter(train_meta_loader)
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                model.train()
                inputs, targets = inputs.cuda(), targets.cuda()
                meta_model = build_model().cuda()
                meta_model.load_state_dict(model.state_dict())
                outputs = meta_model(inputs)

                cost = F.cross_entropy(outputs, targets, reduce=False)
                cost_v = torch.reshape(cost, (len(cost), 1))
                v_lambda = vnet(cost_v.data)
                l_f_meta = torch.sum(cost_v * v_lambda) / len(cost_v)

                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
                meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))
                meta_model.update_params(lr_inner=meta_lr, source_params=grads)

                max_vals, max_indices = torch.max(outputs, 1)
                correct = (max_indices == targets).sum().data.cpu().numpy() / max_indices.size()[0]
                train_accuracy += 100 * correct

                del grads


                try:
                    inputs_val, targets_val = next(train_meta_loader_iter)
                except StopIteration:
                    train_meta_loader_iter = iter(train_meta_loader)
                    inputs_val, targets_val = next(train_meta_loader_iter)
                inputs_val, targets_val = inputs_val.cuda(), targets_val.cuda()
                y_g_hat = meta_model(inputs_val)
                l_g_meta = F.cross_entropy(y_g_hat, targets_val)

                optimizer_vnet.zero_grad()
                l_g_meta.backward()
                optimizer_vnet.step()

                outputs = model(inputs)
                cost_w = F.cross_entropy(outputs, targets, reduce=False)
                cost_v = torch.reshape(cost_w, (len(cost_w), 1))

                with torch.no_grad():
                    w_new = vnet(cost_v)

                loss = torch.sum(cost_v * w_new) / len(cost_v)

                optimizer_model.zero_grad()
                loss.backward()
                optimizer_model.step()

                max_vals, max_indices = torch.max(y_g_hat, 1)
                correct = (max_indices == targets_val).sum().data.cpu().numpy() / max_indices.size()[0]
                meta_accuracy += 100 * correct

                train_loss += loss.item()
                meta_loss += l_g_meta.item()

            train_accuracy /= len(trainloader)
            meta_accuracy /= len(trainloader)

            model.eval()
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)

                max_vals, max_indices = torch.max(outputs, 1)
                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                test_accuracy += 100 * correct

            test_accuracy /= len(testloader)
            print(epoch, train_accuracy, meta_accuracy, test_accuracy)

    elif args.method == 'simple':

        if args.dataset_type == 'severe_imbalance':
            fractions = np.ones(10) / 1000
        elif args.dataset_type == 'imbalance':
            fractions = 0.001 + np.random.rand(10) / 100

        max_class = random.randint(0, 10)
        fractions[max_class] = 1.0
        if args.dataset_create:
            train_imbalance_create(fractions, args.dataset_type)

        trainloader, _, testloader = get_inner_simple_loaders(args)

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

        ##################################################


        for epoch in range(args.start_epoch, args.epochs):
            model.train()

            train_loss = 0.0
            train_accuracy = 0.0
            test_accuracy = 0.0

            for i, data in enumerate(trainloader, 0):

                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                max_vals, max_indices = torch.max(outputs, 1)
                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                train_accuracy += 100 * correct

            scheduler.step()
            train_accuracy /= len(trainloader)
            train_loss /= len(trainloader)

            model.eval()
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)

                max_vals, max_indices = torch.max(outputs, 1)
                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                test_accuracy += 100 * correct


            test_accuracy /= len(testloader)
            print(epoch)
            print(train_loss, train_accuracy, test_accuracy)


    elif args.method == 'loss_reweight':

        if args.dataset_type == 'severe_imbalance':
            fractions = np.ones(10) / 1000
        elif args.dataset_type == 'imbalance':
            fractions = 0.001 + np.random.rand(10) / 100

        max_class = random.randint(0, 10)
        fractions[max_class] = 1.0
        if args.dataset_create:
            train_imbalance_create(fractions, args.dataset_type)

        trainloader, _, testloader = get_inner_simple_loaders(args)

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

        ##################################################
        no_of_classes = 10
        beta = args.loss_reweight_beta
        samples_per_cls = fractions


        for epoch in range(args.start_epoch, args.epochs):
            model.train()

            train_loss = 0.0
            train_accuracy = 0.0
            test_accuracy = 0.0

            for i, data in enumerate(trainloader, 0):

                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)
                loss = CB_loss(labels, outputs, samples_per_cls, no_of_classes, beta) #criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                max_vals, max_indices = torch.max(outputs, 1)
                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                train_accuracy += 100 * correct

            scheduler.step()
            train_accuracy /= len(trainloader)
            train_loss /= len(trainloader)

            model.eval()
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)

                max_vals, max_indices = torch.max(outputs, 1)
                correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
                test_accuracy += 100 * correct


            test_accuracy /= len(testloader)
            print(epoch)
            print(train_loss, train_accuracy, test_accuracy)
