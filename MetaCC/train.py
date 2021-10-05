from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, matthews_corrcoef, roc_auc_score
import torch
import torch.nn as nn
from utils import *
from meta_weight_net_model import *
import numpy as np


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(29, 16),
            nn.ReLU(),
            nn.Linear(16, 24),
            nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(24, 20),
            nn.ReLU(),
            nn.Linear(20, 24),
            nn.ReLU(),
            nn.Linear(24, 1),

            )

    def forward(self, x):
        x = self.model(x)
        return x

def train(inner_lr, meta_batch_update_factor, train_loader, train_loader_outer, test_loader, args):
    batch_losses = []

    model = SimpleNet()
    model = model.cuda()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    trainiter = iter(train_loader)

    ROC_AUC = []

    for epoch in range(args.epochs):

        train_loss = 0.0
        train_acc = 0.0

        inner_loss = 0.0
        inner_acc = 0.0

        test_loss = 0.0
        test_acc = 0.0

        Actual_params = []
        for name, param in model.named_parameters():
            Actual_params.append(param.data)

        for data in train_loader_outer:

            p = 0
            for name, param in model.named_parameters():
                param.data = Actual_params[p]
                p += 1

            x, y, trainiter = get_val_batch(train_loader, trainiter)
            x = x.cuda()
            y = y.cuda()

            y = y.unsqueeze(1)
            y_ = model(x.float())
            loss = criterion(y_, y)

            grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            fast_weights = list(map(lambda p: p[1] - inner_lr * p[0], zip(grad, model.parameters())))

            # REPLACE THE MODEL WITH THE NEW PARAMS
            p = 0
            for name, param in model.named_parameters():
                param.data = fast_weights[p]
                p += 1

            inner_loss += loss.item()
            acc = binary_acc(y_, y)
            inner_acc += acc

            ############################################################

            x, y = data
            x = x.cuda()
            y = y.cuda()

            y = y.unsqueeze(1)
            y_ = model(x.float())
            loss = criterion(y_, y) + 0.01 * loss

            batch_losses.append(loss)

            train_loss += loss.item()
            acc = binary_acc(y_, y)
            train_acc += acc

            if len(batch_losses) > meta_batch_update_factor:
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

        p = 0
        for name, param in model.named_parameters():
            param.data = Actual_params[p]
            p += 1

        y_test = []
        y_pred = []

        for data in test_loader:
            x, y = data
            x = x.cuda()
            y = y.cuda()

            y = y.unsqueeze(1)
            y_ = model(x.float())
            loss = criterion(y_, y)

            y_test.append(y)
            y_pred.append(y_)

            test_loss += loss.item()
            acc = binary_acc(y_, y)
            test_acc += acc

        y_test = torch.cat(y_test).cuda()
        y_pred = torch.cat(y_pred).cuda()

        y_test = y_test.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        roc_auc = roc_auc_score(y_test, y_pred)

        ROC_AUC.append(roc_auc)

    return ROC_AUC



def train_separate(inner_lr, meta_batch_update_factor, train_loader, train_loader_outer, test_loader, args):
    batch_losses = []

    model = SimpleNet()
    model = model.cuda()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    trainouteriter = iter(train_loader_outer)

    ROC_AUC = []

    for epoch in range(args.epochs):

        train_loss = 0.0
        train_acc = 0.0

        inner_loss = 0.0
        inner_acc = 0.0

        test_loss = 0.0
        test_acc = 0.0

        Actual_params = []
        for name, param in model.named_parameters():
            Actual_params.append(param.data)

        for data in train_loader:

            p = 0
            for name, param in model.named_parameters():
                param.data = Actual_params[p]
                p += 1

            x, y = data
            x = x.cuda()
            y = y.cuda()
            y = y.unsqueeze(1)

            y_ = model(x.float())
            loss = criterion(y_, y)

            grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            fast_weights = list(map(lambda p: p[1] - inner_lr * p[0], zip(grad, model.parameters())))

            # REPLACE THE MODEL WITH THE NEW PARAMS
            p = 0
            for name, param in model.named_parameters():
                param.data = fast_weights[p]
                p += 1

            inner_loss += loss.item()
            acc = binary_acc(y_, y)
            inner_acc += acc

            ############################################################

            x, y, trainouteriter = get_val_batch(train_loader_outer, trainouteriter)
            x = x.cuda()
            y = y.cuda()
            y_ = model(x.float())
            loss = criterion(y_, y) + 0.01 * loss

            batch_losses.append(loss)

            train_loss += loss.item()
            acc = binary_acc(y_, y)
            train_acc += acc

            if len(batch_losses) > meta_batch_update_factor:
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

        p = 0
        for name, param in model.named_parameters():
            param.data = Actual_params[p]
            p += 1

        y_test = []
        y_pred = []

        for data in test_loader:
            x, y = data
            x = x.cuda()
            y = y.cuda()

            y = y.unsqueeze(1)
            y_ = model(x.float())
            loss = criterion(y_, y)

            y_test.append(y)
            y_pred.append(y_)

            test_loss += loss.item()
            acc = binary_acc(y_, y)
            test_acc += acc

        y_test = torch.cat(y_test).cuda()
        y_pred = torch.cat(y_pred).cuda()

        y_test = y_test.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        roc_auc = roc_auc_score(y_test, y_pred)

        ROC_AUC.append(roc_auc)

    return ROC_AUC


def CB_loss(labels, logits, samples_per_cls, no_of_classes, beta):


    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    labels_one_hot = labels.squeeze(1)
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

    criterion = nn.BCEWithLogitsLoss(weight=weights)
    cb_loss = criterion(input=logits, target=labels)
    return cb_loss


def train_loss_reweight(train_loader, test_loader, args):

    model = SimpleNet()
    model = model.cuda()

    ### get samples per class
    total_data = 0
    pos_data = 0
    for data in train_loader:
        x, y = data
        total_data += y.shape[0]
        pos_data += y.sum()

    neg_data = total_data - pos_data
    samples_per_cls = [neg_data, pos_data]
    print(samples_per_cls)
    no_of_classes = 2
    beta = args.loss_reweight_beta

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=True)
    ROC_AUC = []

    for epoch in range(args.epochs):
        test_acc = 0.0

        for data in train_loader:
            x, y = data
            x = x.cuda()
            y = y.cuda()

            y = y.unsqueeze(1)
            y_ = model(x.float())
            loss = CB_loss(y, y_, samples_per_cls, no_of_classes, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_test = []
        y_pred = []

        for data in test_loader:
            x, y = data
            x = x.cuda()
            y = y.cuda()

            y = y.unsqueeze(1)
            y_ = model(x.float())

            y_test.append(y)
            y_pred.append(y_)

            acc = binary_acc(y_, y)
            test_acc += acc

        y_test = torch.cat(y_test).cuda()
        y_pred = torch.cat(y_pred).cuda()

        y_test = y_test.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        roc_auc = roc_auc_score(y_test, y_pred)

        ROC_AUC.append(roc_auc)

    return ROC_AUC


def train_baselines(train_loader, test_loader, args):

    model = SimpleNet()
    model = model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=True)
    ROC_AUC = []

    for epoch in range(args.epochs):

        test_loss = 0.0
        test_acc = 0.0

        for data in train_loader:
            x, y = data
            x = x.cuda()
            y = y.cuda()

            y = y.unsqueeze(1)
            y_ = model(x.float())
            loss = criterion(y_, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_test = []
        y_pred = []

        for data in test_loader:
            x, y = data
            x = x.cuda()
            y = y.cuda()

            y = y.unsqueeze(1)
            y_ = model(x.float())
            loss = criterion(y_, y)

            y_test.append(y)
            y_pred.append(y_)

            test_loss += loss.item()
            acc = binary_acc(y_, y)
            test_acc += acc

        y_test = torch.cat(y_test).cuda()
        y_pred = torch.cat(y_pred).cuda()

        y_test = y_test.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        roc_auc = roc_auc_score(y_test, y_pred)

        ROC_AUC.append(roc_auc)

    return ROC_AUC



def build_model():
    model = SimpleNetCC()

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

def train_meta_weight_net(train_loader, train_meta_loader, test_loader, args):

    model = build_model()
    vnet = VNet(1, 100, 1).cuda()

    criterion = nn.BCEWithLogitsLoss(reduce=False)
    optimizer_model = torch.optim.SGD(model.params(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
    optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3,
                                      weight_decay=1e-4)

    ROC_AUC = []
    ROC_AUC_META = []

    for epoch in range(args.epochs):

        model.train()
        train_meta_loader_iter = iter(train_meta_loader)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.unsqueeze(1).cuda()
            meta_model = build_model().cuda()
            meta_model.load_state_dict(model.state_dict())
            outputs = meta_model(inputs)

            cost = criterion(outputs, targets)
            cost_v = torch.reshape(cost, (len(cost), 1))
            v_lambda = vnet(cost_v.data)
            l_f_meta = torch.sum(cost_v * v_lambda) / len(cost_v)
            meta_model.zero_grad()
            grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
            meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))  # For ResNet32
            meta_model.update_params(lr_inner=meta_lr, source_params=grads)
            del grads

            try:
                inputs_val, targets_val = next(train_meta_loader_iter)
            except StopIteration:
                train_meta_loader_iter = iter(train_meta_loader)
                inputs_val, targets_val = next(train_meta_loader_iter)
            # print(targets_val.shape)
            inputs_val, targets_val = inputs_val.cuda(), targets_val.cuda()
            y_g_hat = meta_model(inputs_val)
            # print(y_g_hat.shape, targets_val.shape)
            l_g_meta = criterion(y_g_hat, targets_val).mean()

            optimizer_vnet.zero_grad()
            l_g_meta.backward()
            optimizer_vnet.step()

            outputs = model(inputs)
            cost_w = criterion(outputs, targets)
            cost_v = torch.reshape(cost_w, (len(cost_w), 1))

            with torch.no_grad():
                w_new = vnet(cost_v)

            loss = cost_w.mean()  # torch.sum(cost_v * w_new)/len(cost_v)

            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()

        y_test = []
        y_pred = []

        for data in test_loader:
            x, y = data
            x = x.cuda()
            y = y.cuda()

            y = y.unsqueeze(1)
            y_ = model(x.float())

            y_test.append(y)
            y_pred.append(y_)

        y_test = torch.cat(y_test).cuda()
        y_pred = torch.cat(y_pred).cuda()

        y_test = y_test.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        roc_auc = roc_auc_score(y_test, y_pred)

        ROC_AUC.append(roc_auc)

        y_test = []
        y_pred = []
        test_loss = 0.0
        test_acc = 0.0
        model.eval()

        for data in train_meta_loader:
            x, y = data
            x = x.cuda()
            y = y.cuda()

            y_ = model(x.float())
            loss = criterion(y_, y).mean()

            y_test.append(y)
            y_pred.append(y_)

            test_loss += loss.item()
            acc = binary_acc(y_, y)
            test_acc += acc

        y_test = torch.cat(y_test).cuda()
        y_pred = torch.cat(y_pred).cuda()
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        y_test = y_test.detach().cpu().numpy()
        y_pred_tag = y_pred_tag.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        roc_auc = roc_auc_score(y_test, y_pred)

        ROC_AUC_META.append(roc_auc)

    return ROC_AUC, ROC_AUC_META