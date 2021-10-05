

import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
import json
from utils import prepare_data_meta_balance
from train import *
from scipy.stats import sem
from scipy import mean, std

np.random.seed(2)

parser = argparse.ArgumentParser()
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--inner_lr', default=0.01, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--meta_step', default=80, type=int)
#
parser.add_argument('--test_size', default=0.2, type=float)
parser.add_argument('--meta_size', default=0.2, type=float) #for meta_weight_net
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--outer_batch_size', default=16, type=int)
parser.add_argument('--data_loc', default='loan_data.csv', type=str)
parser.add_argument('--method', default='meta_balance', type=str)
parser.add_argument('--inner_method', default=None, type=str)
parser.add_argument('--outer_method', default=None, type=str)
parser.add_argument('--runs', default=1, type=int)

parser.add_argument('--loss_reweight_beta', default=0.9999, type=float)

args = parser.parse_args()
print("#####################################################")
print(args)

ROC_AUC = []

if args.method == 'meta_balance':
    if args.inner_method is None:
        inner_methods = 'Simple,SMOTE,BorderlineSMOTE,SVMSMOTE,ADASYN,RandomOverSampler,ClusterCentroids,RandomUnderSampler,NearMiss,AllKNN,SMOTEENN'.split(',')
        outer_methods = 'Simple,SMOTE,BorderlineSMOTE,SVMSMOTE,ADASYN,RandomOverSampler,ClusterCentroids,RandomUnderSampler,NearMiss,AllKNN,SMOTEENN'.split(',')
    else:
        inner_methods = args.inner_method.split(',')
        outer_methods = args.outer_method.split(',')

    ROC_AUC_inner = []
    ROC_AUC_inner_max = []
    All_Final = []

    for inner_method in inner_methods:
        ROC_AUC_inner_outer = []
        ROC_AUC_inner_outer_max = []

        for outer_method in outer_methods:
            print(inner_method, outer_method)
            train_loader, train_loader_outer, test_loader = prepare_data_meta_balance(inner_method, outer_method, args)

            for i in range(args.runs): #just the number of times you want to run the same thing

              inner_lr, meta_batch_update_factor = args.inner_lr, args.meta_step
              roc_auc = train_meta_balance(inner_lr, meta_batch_update_factor, train_loader, train_loader_outer, test_loader, args)
              ROC_AUC.append(roc_auc[len(roc_auc)-1])
              print(roc_auc[len(roc_auc)-1])
              if args.runs > 1:
                  All_Final.append(roc_auc[len(roc_auc)-1])

            ROC_AUC_inner_outer.append(roc_auc[len(roc_auc)-1])
            ROC_AUC_inner_outer_max.append(max(roc_auc))

        ROC_AUC_inner.append(ROC_AUC_inner_outer)
        ROC_AUC_inner_max.append(ROC_AUC_inner_outer_max)


    print("ROC_AUC_inner")
    for r in ROC_AUC_inner:
        print(r)

    print("ROC_AUC_inner_max")
    for r in ROC_AUC_inner_max:
        print(r)

    print(All_Final)
    print(mean(All_Final), sem(All_Final))

elif args.method == 'meta_weight_net':
    print(args.method)
    train_loader, train_meta_loader, test_loader = prepare_data_meta_weight_net(args)
    All_Final = []

    for i in range (args.runs): #just the number of times you want to run the same thing
        roc_auc, roc_auc_meta = train_meta_weight_net(train_loader, train_meta_loader, test_loader, args)
        final = roc_auc[len(roc_auc) - 1]
        max_i = roc_auc_meta.index(max(roc_auc_meta))
        print(final, max_i, roc_auc[max_i])
        All_Final.append(final)

    print(All_Final)
    print(mean(All_Final), sem(All_Final))

elif args.method == 'old_baselines':
    if args.inner_method is None:
        methods = 'Simple,SMOTE,BorderlineSMOTE,SVMSMOTE,ADASYN,RandomOverSampler,ClusterCentroids,RandomUnderSampler,NearMiss,AllKNN,SMOTEENN'.split(',')
    else:
        methods = args.inner_method.split(',')
    All_Final = []
    for method in methods:
        print(method)
        train_loader, test_loader = prepare_baseline(method, args)
        for i in range(args.runs):  # just the number of times you want to run the same thing
            roc_auc = train_baselines(train_loader, test_loader, args)
            ROC_AUC.append(roc_auc[len(roc_auc) - 1])
            print(roc_auc[len(roc_auc) - 1])
            All_Final.append(roc_auc[len(roc_auc) - 1])

        print(All_Final)
        print(mean(All_Final), sem(All_Final))

elif args.method == 'loss_reweight':
    train_loader, test_loader = prepare_baseline("Simple", args)
    All_Final = []
    for i in range(args.runs):  # just the number of times you want to run the same thing
        roc_auc = train_loss_reweight(train_loader, test_loader, args)
        ROC_AUC.append(roc_auc[len(roc_auc) - 1])
        print(roc_auc[len(roc_auc) - 1])
        All_Final.append(roc_auc[len(roc_auc) - 1])
    print(All_Final)
    print(mean(All_Final), sem(All_Final))


def save_dict(dict, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(dict))

save_dict(ROC_AUC, '2_layer_25_meta.txt')



