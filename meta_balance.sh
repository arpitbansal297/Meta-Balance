########################## Cifar #######################
python -W ignore train.py --dataset_create --dataset_type severe_imbalance --method metabalance
python -W ignore train.py --dataset_create --dataset_type severe_imbalance --method metaweightnet --lr 0.02 --meta_count 1
python -W ignore train.py --dataset_create --dataset_type severe_imbalance --method simple
python -W ignore train.py --dataset_create --dataset_type severe_imbalance --method loss_reweight --loss_reweight_beta 0.99 --lr 0.02

python -W ignore train.py --dataset_create --dataset_type imbalance --method metabalance
python -W ignore train.py --dataset_create --dataset_type imbalance --method metaweightnet --lr 0.05 --meta_count 1
python -W ignore train.py --dataset_create --dataset_type imbalance --method simple
python -W ignore train.py --dataset_create --dataset_type imbalance --method loss_reweight --loss_reweight_beta 0.99 --lr 0.02



############################# Loan Default #########################3
python -W ignore Meta_loan_default.py --method meta_balance --lr 0.02 --inner_method SVMSMOTE --outer_method RandomUnderSampler
python -W ignore Meta_loan_default.py --method old_baselines --lr 0.05 --inner_method Simple
python -W ignore Meta_loan_default.py --method old_baselines --lr 0.1 --inner_method SMOTE
python -W ignore Meta_loan_default.py --method old_baselines --lr 0.1 --inner_method BorderlineSMOTE
python -W ignore Meta_loan_default.py --method old_baselines --lr 0.1 --inner_method SVMSMOTE
python -W ignore Meta_loan_default.py --method old_baselines --lr 0.1 --inner_method ADASYN
python -W ignore Meta_loan_default.py --method old_baselines --lr 0.1 --inner_method RandomOverSampler
python -W ignore Meta_loan_default.py --method old_baselines --lr 0.01 --inner_method ClusterCentroids
python -W ignore Meta_loan_default.py --method old_baselines --lr 0.1 --inner_method RandomUnderSampler
python -W ignore Meta_loan_default.py --method old_baselines --lr 0.05 --inner_method NearMiss
python -W ignore Meta_loan_default.py --method old_baselines --lr 0.05 --inner_method AllKNN
python -W ignore Meta_loan_default.py --method old_baselines --lr 0.05 --inner_method SMOTEENN
python -W ignore Meta_loan_default.py --method meta_weight_net --meta_size 0.01 --lr 0.01
python -W ignore Meta_loan_default.py --method loss_reweight --lr 0.1 --loss_reweight_beta 0.999



############################# credit card #########################3
python -W ignore Meta_credit_card_fraud.py --method meta_balance --lr 0.02 --inner_method RandomUnderSampler --outer_method AllKNN
python -W ignore Meta_credit_card_fraud.py --method old_baselines --lr 0.1 --inner_method Simple
python -W ignore Meta_credit_card_fraud.py --method old_baselines --lr 0.1 --inner_method SMOTE
python -W ignore Meta_credit_card_fraud.py --method old_baselines --lr 0.1 --inner_method BorderlineSMOTE
python -W ignore Meta_credit_card_fraud.py --method old_baselines --lr 0.1 --inner_method SVMSMOTE
python -W ignore Meta_credit_card_fraud.py --method old_baselines --lr 0.1 --inner_method ADASYN
python -W ignore Meta_credit_card_fraud.py --method old_baselines --lr 0.05 --inner_method RandomOverSampler
python -W ignore Meta_credit_card_fraud.py --method old_baselines --lr 0.01 --inner_method ClusterCentroids
python -W ignore Meta_credit_card_fraud.py --method old_baselines --lr 0.1 --inner_method RandomUnderSampler
python -W ignore Meta_credit_card_fraud.py --method old_baselines --lr 0.05 --inner_method NearMiss
python -W ignore Meta_credit_card_fraud.py --method old_baselines --lr 0.1 --inner_method AllKNN
python -W ignore Meta_credit_card_fraud.py --method old_baselines --lr 0.02 --inner_method SMOTEENN
python -W ignore Meta_credit_card_fraud.py --method meta_weight_net --meta_size 0.2 --lr 0.02
python -W ignore Meta_credit_card_fraud.py --method loss_reweight --lr 0.1 --loss_reweight_beta 0.999


####################### Facial Recognition #####################
python3 Meta_loan_default.py
