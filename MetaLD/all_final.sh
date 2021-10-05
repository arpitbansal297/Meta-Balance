python -W ignore Meta_loan_default.py --method old_baselines --runs 10 --lr 0.02 --inner_method Simple
python -W ignore Meta_loan_default.py --method old_baselines --runs 10 --lr 0.2 --inner_method SMOTE
python -W ignore Meta_loan_default.py --method old_baselines --runs 10 --lr 0.2 --inner_method BorderlineSMOTE
python -W ignore Meta_loan_default.py --method old_baselines --runs 10 --lr 0.1 --inner_method SVMSMOTE
python -W ignore Meta_loan_default.py --method old_baselines --runs 10 --lr 0.2 --inner_method ADASYN
python -W ignore Meta_loan_default.py --method old_baselines --runs 10 --lr 0.05 --inner_method RandomOverSampler
python -W ignore Meta_loan_default.py --method old_baselines --runs 10 --lr 0.01 --inner_method ClusterCentroids
python -W ignore Meta_loan_default.py --method old_baselines --runs 10 --lr 0.01 --inner_method RandomUnderSampler
python -W ignore Meta_loan_default.py --method old_baselines --runs 10 --lr 0.02 --inner_method NearMiss
python -W ignore Meta_loan_default.py --method old_baselines --runs 10 --lr 0.01 --inner_method AllKNN
python -W ignore Meta_loan_default.py --method old_baselines --runs 10 --lr 0.01 --inner_method SMOTEENN
python -W ignore Meta_loan_default.py --method meta_weight_net --runs 10 --meta_size 0.1 --lr 0.1
python -W ignore Meta_loan_default.py --method meta_balance --runs 10 --lr 0.02 --inner_method SVMSMOTE --outer_method RandomUnderSampler
