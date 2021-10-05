

def create_metabalance():
    methods = ["Simple", "ADASYN", "ClusterCentroids", "RandomUnderSampler", "AllKNN"]
    lrs = [0.01, 0.02]
    for inner_method in methods:
        for outer_method in methods:
            for lr in lrs:
                job_str = f"python -W ignore Meta_credit_card_fraud.py " \
                          f"--lr {lr} " \
                          f"--method meta_balance " \
                          f"--runs {5} " \
                          f"--inner_sampling {inner_method} " \
                          f"--outer_sampling {outer_method} "
                print(job_str)


def create_metabalance_separate():
    methods = ['Simple', 'Smote', 'ADASYN', 'RandomOverSampler', 'ClusterCentroids', 'RandomUnderSampler', 'AllKNN', 'SMOTEENN']
    lrs = [0.01, 0.02, 0.05]
    meta_sizes = [0.2, 0.1, 0.05, 0.01]
    for inner_method in methods:
        for meta_size in meta_sizes:
            for lr in lrs:
                job_str = f"python -W ignore Meta_credit_card_fraud.py " \
                          f"--lr {lr} " \
                          f"--method meta_balance_separate " \
                          f"--inner_sampling {inner_method} " \
                          f"--meta_size {meta_size} "
                print(job_str)


def create_oldbaseline():
    methods = ["Simple", "Smote", "ADASYN", "RandomOverSampler", "ClusterCentroids", "RandomUnderSampler", "AllKNN", "SMOTEENN"]
    lrs = [0.01, 0.02, 0.05, 0.1]
    for inner_method in methods:
        for lr in lrs:
            job_str = f"python -W ignore Meta_credit_card_fraud.py " \
                      f"--lr {lr} " \
                      f"--runs {5} " \
                      f"--method meta_balance " \
                      f"--inner_sampling {inner_method} "
            print(job_str)


def create_mwn():

    lrs = [0.01, 0.02, 0.05, 0.1]
    MetaSizes = [0.2, 0.1, 0.05, 0.01]
    for ms in MetaSizes:
        for lr in lrs:
            job_str = f"python -W ignore Meta_credit_card_fraud.py " \
                      f"--lr {lr} " \
                      f"--runs {5} " \
                      f"--method meta_weight_net " \
                      f"--meta_size {ms} "
            print(job_str)


if __name__ == "__main__":
    #create_mwn()
    #create_oldbaseline()
    create_metabalance()
