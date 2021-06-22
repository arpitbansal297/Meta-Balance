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
