

import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
import json
from utils import prepare_data
from train import *

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
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--outer_batch_size', default=16, type=int)
parser.add_argument('--data_loc', default='loan_data.csv', type=str)

args = parser.parse_args()

ROC_AUC = []

inner_method = 'Simple'
outer_method = 'RandomUnderSampler'

print(inner_method, outer_method)
train_loader, train_loader_outer, test_loader = prepare_data(inner_method, outer_method, args)

for i in range (1): #just the number of times you want to run the same thing

  inner_lr, meta_batch_update_factor = args.inner_lr, args.meta_step
  roc_auc = train(inner_lr, meta_batch_update_factor, train_loader, train_loader_outer, test_loader, args)
  ROC_AUC.append(roc_auc[len(roc_auc)-1])
  print(roc_auc[len(roc_auc)-1])

    
def save_dict(dict, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(dict))
    
save_dict(ROC_AUC, '2_layer_25_meta.txt')



