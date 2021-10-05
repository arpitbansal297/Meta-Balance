## MetaFace
Need to download the CelebA dataset from [this link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
The Training and Testing splits are further explained in the paper.
```
python3 train.py 
--modify_data 
--modify_gender 'women' 
--proportion 0.1 
--data_train_root '/loc/to/training/data' 
--data_test_root '/loc/to/testing/data' 
--comet_key 'key'
```
