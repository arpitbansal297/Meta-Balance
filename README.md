This repository is the official PyTorch implementation of Meta-Balance. Find the paper on arxiv

# MetaBalance: High-Performance Neural Networksfor Class-Imbalanced Data


## MetaCifar

Cifar10 dataset is downloaded by the code itself.
Both the Severe and Moderate Class Imbalance is simulated by the code as well.

```
cd MetaCifar
```

Severely Imbalanced Cifar10 data
```
python3 train.py 
--dataset_create 
--dataset_type 'severe_imbalance' 
--comet_key 'key' 
```

Moderately Imbalanced Cifar10 data
```
python3 train.py 
--dataset_create 
--dataset_type 'imbalance' 
--comet_key 'key'
```

## MetaFace
Need to download the CelebA dataset from [this link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
The Training and Testing splits are further explained in the paper.
```
cd MetaFace
python3 train.py 
--modify_data 
--modify_gender 'women' 
--proportion 0.1 
--data_train_root '/loc/to/training/data' 
--data_test_root '/loc/to/testing/data' 
--comet_key 'key'
```

## MetaCC
Download the Loan Default datset from [this link](https://www.kaggle.com/c/1056lab-credit-card-fraud-detection) inside MetaCC.
```
cd MetaCC 
python3 Meta_credit_card_fraud.py 
```

## MetaLD
Download the Loan Default datset from [this link](https://www.kaggle.com/sarahvch/predicting-who-pays-back-loans) inside MetaLD.
```
cd MetaLD 
python3 Meta_loan_default.py 
```
