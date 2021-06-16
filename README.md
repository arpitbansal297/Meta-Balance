# Meta-Balance


# MetaCifar
cd MetaCifar <br />
python3 train.py --dataset_create --dataset_type 'severe_imbalance' --comet_key 'key' <br />
python3 train.py --dataset_type 'severe_imbalance' --comet_key 'key' <br />


python3 train.py --dataset_create --dataset_type 'imbalance' --comet_key 'key' <br />
python3 train.py --dataset_type 'imbalance' --comet_key 'key' <br />

# MetaFace
cd MetaFace <br />
python3 train.py --modify_data --modify_gender 'women' --proportion 0.1 --data_train_root '/loc/to/training/data' --data_test_root '/loc/to/testing/data' --comet_key 'key' <br />


# MetaCC
cd MetaCC <br />
python3 Meta_credit_card_fraud.py <br />


# MetaLD
cd MetaLD <br />
python3 Meta_loan_default.py <br />