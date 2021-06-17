This repository is the official PyTorch implementation of Meta-Balance. Find the paper on arxiv

# MetaBalance: High-Performance Neural Networksfor Class-Imbalanced Data


## MetaCifar
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
```
cd MetaCC 
python3 Meta_credit_card_fraud.py 
```

## MetaLD
```
cd MetaLD 
python3 Meta_loan_default.py 
```