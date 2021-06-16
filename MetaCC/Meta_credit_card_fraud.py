# -*- coding: utf-8 -*-
"""loan_all_meta_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14qOkh7vy2IllURmEnD49VXpOqugHziCz
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import json
from train import *
import argparse
from utils import prepare_data

np.random.seed(2)

parser = argparse.ArgumentParser()
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--inner_lr', default=0.01, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--meta_step', default=80, type=int)

parser.add_argument('--test_size', default=0.2, type=float)
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--outer_batch_size', default=16, type=int)

args = parser.parse_args()

ROC_AUC = []
ROC_AUC_all = []
train_loader, train_loader_outer, test_loader = prepare_data('Simple', 'RandomUnderSampler', args)
  
for i in range (1):
  
  np.random.seed(i)
  inner_lr, meta_batch_update_factor = args.inner_lr, args.meta_step #hyper parameters
  roc_auc = train(inner_lr, meta_batch_update_factor, train_loader, train_loader_outer, test_loader, args)
  ra = roc_auc[len(roc_auc) - 1]
  ROC_AUC.append(ra)
  ROC_AUC_all.append(roc_auc)
  print(ra)

  
  
print(ROC_AUC)
    
def save_dict(dict, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(dict))
    
save_dict(ROC_AUC, 'credit_meta_final.txt')
save_dict(ROC_AUC_all, 'credit_meta_final_all.txt')