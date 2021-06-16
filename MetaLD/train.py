from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, matthews_corrcoef, roc_auc_score
import torch
import torch.nn as nn
from utils import *


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(12, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
            )

    def forward(self, x):
        x = self.model(x)
        return x


def train(inner_lr, meta_batch_update_factor, train_loader, train_loader_outer, test_loader, args):
    batch_losses = []

    model = SimpleNet()
    model = model.cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=True)
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
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        y_test = y_test.detach().cpu().numpy()
        y_pred_tag = y_pred_tag.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        roc_auc = roc_auc_score(y_test, y_pred)

        ROC_AUC.append(roc_auc)

    return ROC_AUC