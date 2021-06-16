from comet_ml import Experiment
import argparse
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from backbone.model_resnet import ResNet_152, ResNet_18
from head.metrics import CosFace
from loss.focal import FocalLoss
from util.utils import separate_resnet_bn_paras, warm_up_lr, \
    schedule_lr, AverageMeter, accuracy
from util.fairness_utils import evaluate
from util.data_loader_utils import prepare_data
import numpy as np
import random
from util.utils import get_val_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(1337)

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
random.seed(222)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
import errno

def create_folder(path):
  try:
    os.mkdir(path)
  except OSError as exc:
      if exc.errno != errno.EEXIST:
          raise
      pass



class Model(nn.Module):
    def __init__(self, backbone_dict, num_class, args):
        super(Model, self).__init__()
        self.backbone = backbone_dict[args.backbone_name]
        self.head = CosFace(in_features=args.embedding_size, out_features=num_class, device_id=args.gpu_id)

    def forward(self, inputs, labels):
        features = self.backbone(inputs)
        outputs = self.head(features, labels)

        return features, outputs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_train_root', default='loc')
    parser.add_argument('--data_val_root', default='loc')
    parser.add_argument('--data_test_root', default='loc')
    parser.add_argument('--demographics', default='CelebA_demographics.txt')
    parser.add_argument('--checkpoints_root', default='checkpoints_fairness')
    parser.add_argument('--non', default='check')
    parser.add_argument('--backbone_name', default='ResNet_18')
    parser.add_argument('--head_name', default='CosFace')
    parser.add_argument('--train_loss', default='Focal', type=str)

    parser.add_argument('--meta_update', default=8, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--outer_batch_size', default=32, type=int)
    parser.add_argument('--input_size', default=[112,112], type=int)
    parser.add_argument('--embedding_size', default=512, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--mean', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--std', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--stages', default=[50, 100, 150], type=list)
    parser.add_argument('--num_workers', default=16, type=int)

    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--inner_lr', default=0.01, type=float)
    parser.add_argument('--leak', default=0.00, type=float)
    parser.add_argument('--num_epoch', default=200, type=int)
    parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='gpu id')

    parser.add_argument('--outer_loader', default='Both', type=str)
    parser.add_argument('--modify_identities', action='store_true')
    parser.add_argument('--modify_data', action='store_true')
    parser.add_argument('--proportion', default=0.1, type=float)
    parser.add_argument('--modify_gender', default='men', type=str)
    parser.add_argument('--comet_key', default=None, type=str)

    args = parser.parse_args()

    inner_lr = args.inner_lr
    leak = args.leak
    ####################################################################################################################################
    # ======= data, model and test data =======#

    experiment = Experiment(api_key=args.comet_key,
                            project_name="Meta_Face")

    dataloaders, num_class = prepare_data(args)


    backbone_dict = {'ResNet_18': ResNet_18(args.input_size), 'ResNet_152': ResNet_152(args.input_size)}
    model = Model(backbone_dict, num_class, args)
    train_criterion = FocalLoss(elementwise=True).cuda()

    ####################################################################################################################
    # ======= optimizer =======#


    model_paras_only_bn, model_paras_wo_bn = separate_resnet_bn_paras(model)

    optimizer = optim.SGD([{'params': model_paras_wo_bn, 'weight_decay': args.weight_decay},
                               {'params': model_paras_only_bn}], lr=args.lr, momentum=args.momentum)

    model = model.cuda()
    model = nn.DataParallel(model)

    ####################################################################################################################################
    # ======= train & validation & save checkpoint =======#
    num_epoch_warm_up = args.num_epoch // 25  # use the first 1/25 epochs to warm up
    num_batch_warm_up = len(dataloaders['train']) * num_epoch_warm_up  # use the first 1/25 epochs to warm up

    ####################################################################################################################################
    # ======= training =======#
    epoch = 0
    batch = 0

    if args.outer_loader == 'Both':
        womeniter = iter(dataloaders['women'])
        meniter = iter(dataloaders['men'])
    else:
        outeriter = iter(dataloaders['outer'])

    batch_losses = []

    meta_update = args.meta_update

    with experiment.train():

        while epoch <= args.num_epoch:

            experiment.log_current_epoch(epoch)
            model.train()
            meters = {}
            meters['loss'] = AverageMeter()
            meters['top5'] = AverageMeter()

            meters['inner_loss'] = AverageMeter()
            meters['inner_top5'] = AverageMeter()

            if epoch in args.stages:  # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
                schedule_lr(optimizer)

            ActualModelParams = []
            for name, param in model.named_parameters():
                ActualModelParams.append(param.data)

            if batch + 1 <= num_batch_warm_up:

                print('Warmup') # Warmup without meta-balance

                for inputs, labels, young, male in tqdm(iter(dataloaders['naive_train'])):

                    if batch + 1 <= num_batch_warm_up:  # adjust LR for each training batch during warm up
                        warm_up_lr(batch + 1, num_batch_warm_up, args.lr, optimizer)


                    inputs, labels = inputs.to(device), labels.to(device).long()
                    features, outputs = model(inputs, labels)


                    loss = train_criterion(outputs, labels).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                    meters['loss'].update(loss.data.item(), inputs.size(0))
                    meters['top5'].update(prec5.data.item(), inputs.size(0))

                    meters['inner_loss'].update(loss.data.item(), inputs.size(0))
                    meters['inner_top5'].update(prec5.data.item(), inputs.size(0))

                    batch += 1

            else:
                print('No Warmup') # once the warmup is done

                for inputs, labels, young, male in tqdm(iter(dataloaders['naive_train'])): # make sure to use everything

                    if batch + 1 <= num_batch_warm_up:  # adjust LR for each training batch during warm up
                        warm_up_lr(batch + 1, num_batch_warm_up, args.lr, optimizer)

                    # print(labels)
                    ############### inner loop ############

                    p = 0
                    for name, param in model.named_parameters():
                        param.data = ActualModelParams[p]
                        p += 1

                    inputs, labels = inputs.cuda(), labels.cuda().long()
                    features, outputs = model(inputs, labels)
                    loss = train_criterion(outputs, labels).mean()

                    inner_prec1, inner_prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                    meters['inner_loss'].update(loss.data.item(), inputs.size(0))
                    meters['inner_top5'].update(inner_prec5.data.item(), inputs.size(0))

                    grad_model = torch.autograd.grad(loss, model.parameters(), create_graph=True)

                    fast_weights_model = list(map(lambda p: p[1] - inner_lr * p[0], zip(grad_model, model.parameters())))

                    p = 0
                    for name, param in model.named_parameters():
                        param.data = fast_weights_model[p]
                        p += 1

                    ################# outer ##############


                    if args.outer_loader == 'Both':
                        inputs, labels, young, male, womeniter = get_val_batch(dataloaders['women'], womeniter)
                        inputs, labels = inputs.cuda(), labels.cuda().long()
                        features, outputs = model(inputs, labels)
                        loss_women = train_criterion(outputs, labels).mean()

                        inputs, labels, young, male, meniter = get_val_batch(dataloaders['men'], meniter)
                        inputs, labels = inputs.cuda(), labels.cuda().long()
                        features, outputs = model(inputs, labels)
                        loss_men = train_criterion(outputs, labels).mean()

                        loss = loss_women + loss_men + leak * loss
                        batch_losses.append(loss)

                    else:
                        inputs, labels, young, male, outeriter = get_val_batch(dataloaders['outer'], outeriter)

                        inputs, labels = inputs.cuda(), labels.cuda().long()
                        features, outputs = model(inputs, labels)
                        loss = train_criterion(outputs, labels).mean() + leak * loss
                        batch_losses.append(loss)



                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                    meters['loss'].update(loss.data.item(), inputs.size(0))
                    meters['top5'].update(prec5.data.item(), inputs.size(0))

                    batch += 1

                    # after the losses are accumulated we can update the model
                    if len(batch_losses) > meta_update :

                        p = 0
                        for name, param in model.named_parameters():
                            param.data = ActualModelParams[p]
                            p += 1

                        meta_batch_loss = torch.stack(batch_losses).mean()
                        optimizer.zero_grad()
                        meta_batch_loss.backward()
                        optimizer.step()

                        batch_losses = []

                        ActualModelParams = []  # save_weights(head)
                        for name, param in model.named_parameters():
                            ActualModelParams.append(param.data)




            backbone = model.module.backbone
            head = model.module.head
            backbone = backbone.cuda()
            head = head.cuda()
            model.eval()
            backbone.eval()  # set to training mode
            head.eval()
            experiment.log_metric("Training Loss", meters['loss'].avg, step=epoch)
            experiment.log_metric("Training Acc5", meters['top5'].avg, step=epoch)

            experiment.log_metric("Training Inner Loss", meters['inner_loss'].avg, step=epoch)
            experiment.log_metric("Training Inner Acc5", meters['inner_top5'].avg, step=epoch)


            '''For train data compute only multilabel accuracy'''
            loss_overall_train, loss_male_train, loss_female_train, acc_overall_train, acc_male_train, acc_female_train, \
            _,_,_ = evaluate(dataloaders['train'], train_criterion, backbone,head, args.embedding_size,  k_accuracy = False, multilabel_accuracy = True)

            '''For test data compute only k-neighbors accuracy'''
            _, _, _, _, _, _, acc_k_overall_test, acc_k_male_test, acc_k_female_test = evaluate(dataloaders['test'], train_criterion, backbone,
                                                                         head, args.embedding_size,  k_accuracy = True, multilabel_accuracy = False)

            ''' Log here everything that you want like that
            experiment.log_metric("Loss valset Overall", loss_overall_val, step=batch)
            
            
            '''

            experiment.log_metric("Acc k testset Overall", acc_k_overall_test, step=epoch)
            experiment.log_metric("Acc k Male testset Overall", acc_k_male_test, step=epoch)
            experiment.log_metric("Acc k Female testset Overall", acc_k_female_test, step=epoch)

            # save checkpoints per epoch

            if (epoch % 20) == 0:

                name_of_notebook = args.non
                save_file_folder = './' + name_of_notebook + '/'
                create_folder(save_file_folder)

                save_file = save_file_folder + str(epoch) + '.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_file)

            epoch += 1

