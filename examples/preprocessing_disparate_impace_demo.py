# import os
# import sys
# sys.path.insert(0, '../')
# import matplotlib.pyplot as plt
# import numpy as np
# from IPython.display import Markdown, display
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
#
# from aif360.algorithms.preprocessing import DisparateImpactRemover
#
# # Datasets
# from aif360.datasets import MEPSDataset19, AdultDataset, GermanDataset, BankDataset, CompasDataset
# from aif360.datasets.compas_dataset1 import CompasDataset_1
#
# # Fairness metrics
# from aif360.metrics import BinaryLabelDatasetMetric
# from aif360.metrics import ClassificationMetric
#
# import tensorflow as tf
# import numpy as np
#
# from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn
# from plot_result import Plot


import os.path
import random
import sys
import time

sys.path.insert(0, '../')
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

# Datasets
from aif360.datasets import MEPSDataset19, AdultDataset, GermanDataset, BankDataset, CompasDataset
from aif360.datasets.compas_dataset1 import CompasDataset_1
# from aif360.datasets import MEPSDataset20
# from aif360.datasets import MEPSDataset21

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

# Explainers
from aif360.explainers import MetricTextExplainer

# Scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover\

from model_training.model import Model
from model_training.network import *
from model_training.layer import *
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization
# import torch
from sklearn.metrics import accuracy_score
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_compas, get_distortion_german
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german
from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, initial_dnn2, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
from aif360.algorithms.preprocessing import DisparateImpactRemover
from plot_result import Plot

seed = 1
np.random.seed(seed)
# tf.random.set_seed(seed)
tf.random.set_random_seed(seed)
nb_classes = 2
# BATCH_SIZE = 32
# EPOCHS = 500
MAX_NUM = 1000  # 251 for causal discrimination test / 2000 for group discrimination test
names = locals()

# MODEL_DIR = "adult_original_remover.h5"
# MODEL_TRANS_DIR = "adult_remover(sex).h5"

# Adult Income dataset
# all_feature_names = ['age', 'workclass','education-num', 'occupation', 'relationship','capital-gain', 'capital-loss', 'hours-per-week']
# categorical_features = ['workclass', 'occupation', 'relationship']

# German Credit dataset
# all_feature_names = ['status', 'credit_amount', 'savings', 'employment', 'investment_as_income_percentage', 'age', 'number_of_credits', 'skill_level', 'foreign_worker']
# categorical_features = ['status', 'savings', 'employment', 'skill_level','foreign_worker']

# Compas dataset
# all_feature_names = ['age', 'age_cat',
#                     'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count',
#                     'priors_count', 'c_charge_degree', 'is_recid',
#                     'is_violent_recid', 'decile_score', 'v_decile_score', 'priors_count', 'start', 'end', 'event',
#                     'two_year_recid']
# categorical_features = ['age_cat', 'c_charge_degree']
#
# ad = CompasDataset_1(protected_attribute_names=[protected],
#         privileged_classes=[protected_object], categorical_features=categorical_features,
#         features_to_keep=all_feature_names)

# processing_name = str(os.path.basename(__file__)).split("_demo")[0]
processing_name = "DI"

def DI_metric_test(dataset_name):
    print("-->DI_metric_test")
    if dataset_name == "Adult income":
        ad = AdultDataset()
        protected_attributes = ['race', 'sex']
        privileged_groups = [{'race': 1.0}, {'sex': 1.0}]
        unprivileged_groups = [{'race': 0.0}, {'sex': 0.0}]
    elif dataset_name == "German credit":
        ad = GermanDataset()
        protected_attributes = ['sex', 'age']
        privileged_groups = [{'sex': 1.0}, {'age': 1.0}]
        unprivileged_groups = [{'sex': 0.0}, {'age': 0.0}]
    elif dataset_name == "Bank":
        ad = BankDataset()
        protected_attributes = ['age']
        privileged_groups = [{'age': 1.0}]
        unprivileged_groups = [{'age': 0.0}]
    else:
        ad = CompasDataset_1()
        protected_attributes = ['sex', 'race']
        privileged_groups = [{'sex': 1.0}, {'race': 1.0}]
        unprivileged_groups = [{'sex': 0.0}, {'race': 0.0}]

    if dataset_name == "Bank":
        name = "Bank"
        BATCH_SIZE = 128
        EPOCHS = 50
    elif dataset_name == "Adult income":
        name = "Adult"
        BATCH_SIZE = 128
        EPOCHS = 10
    elif dataset_name == "German credit":
        name = "Credit"
        BATCH_SIZE = 32
        EPOCHS = 500
    else:
        name = "Compas"
        BATCH_SIZE = 128
        EPOCHS = 100

    print("-->ad", ad)
    print(ad.feature_names)

    di = DisparateImpactRemover(repair_level=1.0)
    ad_repd = di.fit_transform(ad)

    print("-->ad_repd", ad_repd)
    print(ad_repd.feature_names)
    if ad_repd != ad:
        print('-->Error raised: ad_repd != ad')

    test, train = ad.split([0.7], shuffle=True, seed=seed)  # Adult Income: 16281, German Credit: 200, Compas:1000

    basic_time1 = time.time()
    # scaler = MinMaxScaler(copy=False)  # MinMaxScaler
    # scaler = MinMaxScaler()
    # train.features = scaler.fit_transform(train.features)
    # test.features = scaler.transform(test.features)

    train.features = train.features
    test.features = test.features

    if np.any(test.labels):
        print("-->True")
    else:
        print("-->not at least one label is True")

    indexs = [train.feature_names.index(protected_name) for protected_name in protected_attributes]
    print("-->indexs", indexs)

    # X_tr = np.delete(train.features, indexs, axis=1)
    # X_te = np.delete(test.features, indexs, axis=1)
    X_tr = train.features
    X_te = test.features
    print("-->X_tr", X_tr)
    print(list(X_tr[0]))
    X_te_dataset = test.copy()
    X_te_dataset.features = X_te
    y_tr = train.labels.ravel()
    y_te = test.labels.ravel()

    dimension = len(X_tr[0])
    orig_lmod = initial_dnn2(dimension)
    orig_lmod.fit(x=X_tr, y=y_tr, sample_weight=train.instance_weights, batch_size=BATCH_SIZE,
              epochs=EPOCHS, shuffle=False, verbose=1)

    MODEL_DIR = name + "/DI_model.h5"
    orig_lmod.save(MODEL_DIR)

    # orig_lmod = keras.models.load_model(MODEL_DIR)

    # orig_lmod = MLPClassifier(solver='adam', activation='identity', max_iter=100, alpha=1e-5,
    #                       hidden_layer_sizes=(64, 32, 16, 8, 4),
    #                       random_state=1, verbose=True)
    # orig_lmod.fit(X_tr, y_tr)

    test_pred = test.copy()
    test_pred.labels = orig_lmod.predict(X_te)
    orig_test_pred_prob = orig_lmod.predict_proba(X_te)

    basic_time2 = time.time()

    # print("----------" + "test on test data" + "----------")
    # cm = ClassificationMetric(test, test_pred, privileged_groups=privileged_groups,
    #                           unprivileged_groups=unprivileged_groups)
    # before = cm.disparate_impact()
    # # print("-->prediction accuracy on test data",accuracy_score(list(test), list(test_pred)))
    # print('Disparate impact: {:.4}'.format(before))
    # print('Acc overall: {:.4}'.format(cm.accuracy()))

    thresh_arr = np.array([0.5])
    # print("----------" + "get_metrics" + "----------")
    # orig_metrics = get_metrics(test, test_pred, privileged_groups=privileged_groups,
    #                            unprivileged_groups=unprivileged_groups)
    # describe_metrics(orig_metrics, thresh_arr)

    def new_inputs_to_dataset(new_inputs, original_dataset):
        classified_dataset = original_dataset.copy()
        classified_dataset.features = np.array(new_inputs)
        length = len(new_inputs)
        classified_dataset.instance_names = [1] * length
        classified_dataset.instance_weights = np.array([1] * length)
        classified_dataset.protected_attributes = np.array([[input[classified_dataset.protected_attribute_indexs[0]],
                                                             input[classified_dataset.protected_attribute_indexs[1]]]
                                                            for
                                                            input in new_inputs])
        return classified_dataset


    input_privileged = []
    input_unprivileged = []
    for i in range(0, len(privileged_groups)):
        privileged_group = [privileged_groups[i]]
        unprivileged_group = [unprivileged_groups[i]]
        # for group in privileged_groups:
        #     group = [group]
        new_inputs_priviledge = ad.generate_inputs(max_num=MAX_NUM, if_random=False,
                                                                           privileged_groups=privileged_group,
                                                                           if_priviledge=True)
        input_privileged += new_inputs_priviledge
        new_inputs_unpriviledge = ad.generate_inputs(max_num=MAX_NUM, if_random=False,
                                                                             privileged_groups=unprivileged_group,
                                                                             if_priviledge=True)
        input_unprivileged += new_inputs_unpriviledge

    new_inputs = input_privileged + input_unprivileged
    random.shuffle(new_inputs)
    # classified_dataset_full = new_inputs_to_dataset(new_inputs, train)
    # print("-->classified_dataset_full", classified_dataset_full)
    # classified_dataset = classified_dataset_full.copy()
    # classified_dataset.features = scaler.fit_transform(classified_dataset.features)
    # classified_dataset_features = np.delete(classified_dataset_full.features, indexs, axis=1)
    # # classified_dataset_features = classified_dataset_full.features
    # classified_dataset.features = classified_dataset_features
    # orig_classified_dataset_pred_prob = orig_lmod.predict_proba(classified_dataset.features)


    thresh_arr = np.array([0.5])

    print("----------" + "test on test data" + "----------")
    # test, test_pred
    multi_orig_metrics = metric_test_new_inputs(dataset=test,
                                                model=None,
                                                thresh_arr=None,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups,
                                                dataset_pred=orig_test_pred_prob
                                                )
    describe_metrics_new_inputs(multi_orig_metrics, thresh_arr)

    print("----------" + "multivariate group metric test" + "----------")
    multi_group_metrics = metric_test_multivariate(dataset=test,   # classified_dataset_full
                                                   model=None,
                                                   thresh_arr=None,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups,
                                                   dataset_pred=orig_test_pred_prob   # orig_classified_dataset_pred_prob
                                                   )
    # multi_group_metrics['acc'] = [accuracy_score(list(y_te), list(orig_lmod.predict_classes(X_te)))]
    print(multi_group_metrics)

    print("----------" + "causal metric test" + "----------")
    multi_causal_metrics = metric_test_causal(dataset=test,
                                                    model=orig_lmod,
                                                    thresh_arr=None,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups,
                                                    dataset_pred=orig_test_pred_prob
                                                    )
    # multi_causal_metrics['acc'] = [accuracy_score(list(y_te), list(orig_lmod.predict(X_te)))]
    print(multi_causal_metrics)

    time2 = time.time()
    print("-->DI fitting")
    di = DisparateImpactRemover(repair_level=1.0)
    train_repd = di.fit_transform(train)
    test_repd = di.fit_transform(test)

    print("-->original test", list(test.features[:10]))
    print("-->test_repd", list(test_repd.features[:10]))

    # X_tr_repd = np.delete(train_repd.features, indexs, axis=1)
    # X_te_repd = np.delete(test_repd.features, indexs, axis=1)
    X_tr_repd = train_repd.features
    x_te_repd = test_repd.features
    y_tr_repd = train_repd.labels.ravel()
    y_te_repd = test_repd.labels.ravel()

    y_tr = train.labels.ravel()
    y_te = test.labels.ravel()

    dimension = len(X_tr_repd[0])
    trans_lmod = initial_dnn2(dimension)
    trans_lmod.fit(x=X_tr_repd, y=y_tr_repd, sample_weight=train_repd.instance_weights, batch_size=BATCH_SIZE,
                  epochs=EPOCHS, shuffle=False, verbose=0)

    # trans_lmod = MLPClassifier(solver='adam', activation='identity', max_iter=100, alpha=1e-5,
    #                           hidden_layer_sizes=(64, 32, 16, 8, 4),
    #                           random_state=1, verbose=True)
    # trans_lmod.fit(X_tr_repd, y_tr_repd)

    print("-->X_tr_repd", X_tr_repd)
    print(list(X_tr_repd[0]))

    test_pred = test.copy()
    test_pred.labels = trans_lmod.predict(X_te)
    trans_test_pred_prob = trans_lmod.predict_proba(X_te)
    # trans_classified_dataset_pred_prob = trans_lmod.predict_proba(classified_dataset.features)

    time3 = time.time()

    print("----------" + "test on test data" + "----------")
    multi_orig_trans_metrics = metric_test_new_inputs(dataset=test,
                                                model=None,
                                                thresh_arr=None,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups,
                                                dataset_pred=trans_test_pred_prob
                                                )
    describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)

    print("----------" + "multivariate group metric test" + "----------")
    multi_group_trans_metrics = metric_test_multivariate(dataset=test,   # classified_dataset_full
                                                   model=None,
                                                   thresh_arr=None,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups,
                                                   dataset_pred=trans_test_pred_prob  # trans_classified_dataset_pred_prob
                                                   )
    # multi_group_trans_metrics['acc'] = [accuracy_score(list(y_te), list(trans_lmod.predict_classes(X_te)))]
    print(multi_group_trans_metrics)

    print("----------" + "causal metric test" + "----------")
    multi_causal_trans_metrics = metric_test_causal(dataset=test,
                                              model=trans_lmod,
                                              thresh_arr=None,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups,
                                              dataset_pred=trans_test_pred_prob
                                              )
    # multi_causal_trans_metrics['acc'] = [accuracy_score(list(y_te), list(trans_lmod.predict(X_te)))]
    print(multi_causal_trans_metrics)

    if len(privileged_groups) == 1:
        multi_orig_metrics = [multi_orig_metrics, multi_group_metrics]
        print("-->multi_orig_metrics", multi_orig_metrics)
        all_multi_orig_metrics = defaultdict(list)
        for to_merge in multi_orig_metrics:
            for key, value in to_merge.items():
                # print("-->value", value)
                all_multi_orig_metrics[key].append(value[0])
        print("-->all_multi_orig_metrics", all_multi_orig_metrics)
        # multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics, multi_causal_trans_metrics]
        multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics]
        all_multi_trans_metrics = defaultdict(list)
        for to_merge in multi_trans_metrics:
            for key, value in to_merge.items():
                # print("-->value", value)
                all_multi_trans_metrics[key].append(value[0])
        print("-->all_multi_trans_metrics", all_multi_trans_metrics)
        print("--all results:")
        print([dict(all_multi_orig_metrics), dict(all_multi_trans_metrics)])
        print("time for processing:", basic_time2-basic_time1+time3-time2)
        return all_multi_orig_metrics, all_multi_trans_metrics, dataset_name, processing_name


    # univariate test
    all_uni_orig_metrics = []
    all_uni_trans_metrics = []
    for index in range(0, len(protected_attributes)):
        sens_attr = protected_attributes[index]
        print("-->sens_attr", sens_attr)

        privileged_groups = [{sens_attr: 1}]
        unprivileged_groups = [{sens_attr: 0}]

        # protected = 'sex'  # sex, race, sex, age, race
        # protected_object = ['Female']  # Male, White, male, lambda x: x > 25, Female, Caucasian

        index = train.feature_names.index(sens_attr)
        # X_tr = np.delete(train.features, index, axis=1)
        # X_te = np.delete(test.features, index, axis=1)
        X_tr = train.features
        X_te = test.features
        y_tr = train.labels.ravel()

        dimension = len(X_tr[0])
        orig_lmod = initial_dnn2(dimension)
        orig_lmod.fit(x=X_tr, y=y_tr, sample_weight=train.instance_weights, batch_size=BATCH_SIZE,
                      epochs=EPOCHS, shuffle=False, verbose=0)

        # orig_lmod = MLPClassifier(solver='adam', activation='identity', max_iter=100, alpha=1e-5,
        #                           hidden_layer_sizes=(64, 32, 16, 8, 4),
        #                           random_state=1, verbose=True)
        # orig_lmod.fit(X_tr, y_tr)

        # classified_dataset = classified_dataset_full.copy()
        # classified_dataset.features = scaler.fit_transform(classified_dataset.features)
        # classified_dataset_features = np.delete(classified_dataset_full.features, index, axis=1)
        # classified_dataset.features = classified_dataset_features
        # orig_classified_dataset_pred_prob = orig_lmod.predict_proba(classified_dataset.features)

        # print("----------" + "test on test data" + "----------")
        # cm = ClassificationMetric(test, test_pred, privileged_groups=privileged_groups,
        #                           unprivileged_groups=unprivileged_groups)
        # before = cm.disparate_impact()
        # # print("-->prediction accuracy on test data",accuracy_score(list(test), list(test_pred)))
        # print('Disparate impact: {:.4}'.format(before))
        # print('Acc overall: {:.4}'.format(cm.accuracy()))
        #
        # thresh_arr = np.array([0.5])
        # orig_metrics = get_metrics(test, test_pred, privileged_groups=privileged_groups,
        #                            unprivileged_groups=unprivileged_groups)
        # describe_metrics(orig_metrics, thresh_arr)

        names["orig_" + str(sens_attr) + '_metrics'] = []
        print("----------" + "test on test data" + "----------")
        # test, test_pred
        uni_orig_metrics = metric_test_new_inputs(dataset=test,
                                                    model=None,
                                                    thresh_arr=None,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups,
                                                    dataset_pred=orig_test_pred_prob
                                                    )
        describe_metrics_new_inputs(uni_orig_metrics, thresh_arr)
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_orig_metrics)

        print("----------" + "multivariate group metric test" + "----------")
        uni_group_metrics = metric_test_multivariate(dataset=test,   # classified_dataset_full
                                                       model=None,
                                                       thresh_arr=None,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups,
                                                       dataset_pred=orig_test_pred_prob    # orig_classified_dataset_pred_prob
                                                       )
        print(uni_group_metrics)
        # uni_group_metrics['acc'] = [accuracy_score(list(y_te), list(orig_lmod.predict_classes(X_te)))]
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_group_metrics)

        print("----------" + "causal metric test" + "----------")
        uni_causal_metrics = metric_test_causal(dataset=test,
                                                  model=orig_lmod,
                                                  thresh_arr=None,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups,
                                                  dataset_pred=orig_test_pred_prob
                                                  )
        # uni_causal_metrics['acc'] = [accuracy_score(list(y_te), list(orig_lmod.predict(X_te)))]
        print(uni_causal_metrics)
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_causal_metrics)

        time4 = time.time()
        di = DisparateImpactRemover(repair_level=1.0)
        train_repd = di.fit_transform(train)
        test_repd = di.fit_transform(test)
        y_tr_repd = train_repd.labels.ravel()
        y_te_repd = test_repd.labels.ravel()

        # X_tr_repd = np.delete(train_repd.features, index, axis=1)
        # X_te_repd = np.delete(test_repd.features, index, axis=1)

        dimension = len(X_tr_repd[0])
        trans_lmod = initial_dnn2(dimension)
        trans_lmod.fit(x=X_tr_repd, y=y_tr_repd, sample_weight=train_repd.instance_weights, batch_size=BATCH_SIZE,
                       epochs=EPOCHS, shuffle=False, verbose=0)

        # trans_lmod = MLPClassifier(solver='adam', activation='identity', max_iter=100, alpha=1e-5,
        #                      hidden_layer_sizes=(64, 32, 16, 8, 4),
        #                      random_state=1, verbose=True)
        # trans_lmod.fit(X_tr_repd, y_tr_repd)

        trans_test_pred_prob = trans_lmod.predict_proba(X_te)
        # trans_classified_dataset_pred_prob = trans_lmod.predict_proba(classified_dataset.features)

        # test_repd_pred = test_repd.copy()
        # test_repd_pred.labels = lmod.predict(X_te_repd)
        # test_repd_pred.labels = lmod.predict_classes(X_te_repd)

        # print("----------" + "test on test data after disparate impatct removing" + "----------")
        # repaired_cm = ClassificationMetric(test_repd, test_repd_pred, privileged_groups=privileged_groups,
        #                                    unprivileged_groups=unprivileged_groups)
        # after = repaired_cm.disparate_impact()
        # print('Disparate impact: {:.4}'.format(after))
        # print('Acc overall: {:.4}'.format(repaired_cm.accuracy()))
        #
        # print("-->for transformed test data")
        # thresh_arr = np.array([0.5])
        # repd_metrics = get_metrics(test_repd, test_repd_pred, privileged_groups=privileged_groups,
        #                            unprivileged_groups=unprivileged_groups)
        # lr_orig_best_ind = np.argmax(repd_metrics['bal_acc'])
        # describe_metrics(repd_metrics, thresh_arr)

        time5 = time.time()
        names["trans_" + str(sens_attr) + '_metrics'] = []
        print("----------" + "test on test data" + "----------")
        # test, test_pred
        uni_trans_metrics = metric_test_new_inputs(dataset=test,
                                                  model=None,
                                                  thresh_arr=None,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups,
                                                  dataset_pred=trans_test_pred_prob
                                                  )
        describe_metrics_new_inputs(uni_orig_metrics, thresh_arr)
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_trans_metrics)

        print("----------" + "multivariate group metric test" + "----------")
        uni_group_trans_metrics = metric_test_multivariate(dataset=test,  # classified_dataset_full
                                                     model=None,
                                                     thresh_arr=None,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups,
                                                     dataset_pred=trans_test_pred_prob  # trans_classified_dataset_pred_prob
                                                     )
        # uni_group_trans_metrics['acc'] = [accuracy_score(list(y_te), list(orig_lmod.predict_classes(X_te)))]
        print(uni_group_trans_metrics)
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_group_trans_metrics)

        print("----------" + "causal metric test" + "----------")
        uni_causal_trans_metrics = metric_test_causal(dataset=test,
                                                model=trans_lmod,
                                                thresh_arr=None,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups,
                                                dataset_pred=trans_test_pred_prob
                                                )
        # uni_causal_trans_metrics['acc'] = [accuracy_score(list(y_te), list(orig_lmod.predict(X_te)))]
        print(uni_causal_trans_metrics)
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_causal_trans_metrics)

        input_dicts = names["orig_" + str(sens_attr) + '_metrics']
        names['uni_' + str(sens_attr) + '_metrics'] = defaultdict(list)
        for to_merge in input_dicts:
            for key, value in to_merge.items():
                names['uni_' + str(sens_attr) + '_metrics'][key].append(value[0])
        all_uni_orig_metrics.append(names['uni_' + str(sens_attr) + '_metrics'])

        input_dicts = names["trans_" + str(sens_attr) + '_metrics']
        names['uni_' + str(sens_attr) + '_metrics'] = defaultdict(list)
        for to_merge in input_dicts:
            for key, value in to_merge.items():
                # print("-->value", value)
                names['uni_' + str(sens_attr) + '_metrics'][key].append(value[0])
        all_uni_trans_metrics.append(names['uni_' + str(sens_attr) + '_metrics'])
        print("time for protected attribute", sens_attr, basic_time2-basic_time1+time5 - time4)

    print("time for multivariate attributes", basic_time2-basic_time1+ time3 - time2)
    print("-->uni_race_orig_metrics", all_uni_orig_metrics[0])
    print("-->uni_race_trans_metrics", all_uni_trans_metrics[0])
    print("-->uni_sex_orig_metrics", all_uni_orig_metrics[1])
    print("-->uni_sex_trans_metrics", all_uni_trans_metrics[1])
    # multi_orig_metrics = [multi_orig_metrics, multi_group_metrics, multi_causal_metrics]
    multi_orig_metrics = [multi_orig_metrics, multi_group_metrics]
    print("-->multi_orig_metrics", multi_orig_metrics)
    all_multi_orig_metrics = defaultdict(list)
    for to_merge in multi_orig_metrics:
        for key, value in to_merge.items():
            # print("-->value", value)
            all_multi_orig_metrics[key].append(value[0])
    print("-->all_multi_orig_metrics", all_multi_orig_metrics)
    # multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics, multi_causal_trans_metrics]
    multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics]
    all_multi_trans_metrics = defaultdict(list)
    for to_merge in multi_trans_metrics:
        for key, value in to_merge.items():
            # print("-->value", value)
            all_multi_trans_metrics[key].append(value[0])
    print("-->all_multi_trans_metrics", all_multi_trans_metrics)

    print("-->all results:")
    print([dict(all_uni_orig_metrics[0]), dict(all_uni_trans_metrics[0]), dict(all_uni_orig_metrics[1]),
           dict(all_uni_trans_metrics[1]),
           dict(all_multi_orig_metrics), dict(all_multi_trans_metrics)])

    from plot_result import Plot
    sens_attrs = ad.protected_attribute_names
    Plot = Plot(dataset_name=dataset_name, sens_attr=sens_attrs, processing_name=processing_name)
    # Plot.plot_acc_metric(lr_orig_metrics, lr_transf_metrics, "stat_par_diff")
    multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
    # def plot_abs_acc_multi_metric(self, orig_uni_metrics1, improved_uni_metrics1, orig_uni_metrics2, improved_uni_metrics2,
    #         orig_multi_metrics, improved_multi_metrics):
    # 1 image
    Plot.plot_abs_acc_all_metric(all_uni_orig_metrics[0], all_uni_trans_metrics[0], all_uni_orig_metrics[1],
                                 all_uni_trans_metrics[1], all_multi_orig_metrics, all_multi_trans_metrics,
                                 metric_names=['stat_par_diff', 'group'])
    # # 2 images: one for group metric. one for causal metric
    # Plot.plot_abs_acc_multi_metric(all_uni_orig_metrics[0], all_uni_trans_metrics[0], all_uni_orig_metrics[1],
    #                                all_uni_trans_metrics[1], all_multi_orig_metrics, all_multi_trans_metrics)
    # # 3 images: one for 'race', one for 'sex', one for 'race,sex'
    # Plot.plot_abs_acc_individual_metric(all_uni_orig_metrics[0], all_uni_trans_metrics[0], all_uni_orig_metrics[1],
    #                                     all_uni_trans_metrics[1], all_multi_orig_metrics, all_multi_trans_metrics)

    return all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics, all_multi_trans_metrics, \
           dataset_name, sens_attrs, processing_name













# dataset_name = "Adult income"

def metric_test(dataset_name):
    if dataset_name == "Adult income":
        ad = AdultDataset()
        protected_attributes = ['race', 'sex']
        privileged_groups = [{'race': 1.0}, {'sex': 1.0}]
        unprivileged_groups = [{'race': 0.0}, {'sex': 0.0}]
    elif dataset_name == "German credit":
        ad = GermanDataset()
        protected_attributes = ['sex', 'age']
        all_privileged_groups = {'sex': [{'sex': 1}], 'age': [{'age': 1}]}
        all_unprivileged_groups = {'sex': [{'sex': 0}], 'age': [{'age': 0}]}
    else:
        ad = CompasDataset_1()
        protected_attributes = ['sex', 'race']
        all_privileged_groups = {'sex': [{'sex': 1}], 'race': [{'race': 1}]}
        all_unprivileged_groups = {'sex': [{'sex': 0}], 'race': [{'race': 0}]}

    print("-->ad", ad)
    print(ad.feature_names)

    di = DisparateImpactRemover(repair_level=0.)
    ad_repd = di.fit_transform(ad)

    print("-->ad_repd", ad_repd)
    print(ad_repd.feature_names)
    if ad_repd != ad:
        print('-->Error raised: ad_repd != ad')

    test, train = ad.split([0.7], shuffle=True, seed=seed)  # Adult Income: 16281, German Credit: 200, Compas:1000

    scaler = MinMaxScaler(copy=False)  # MinMaxSclaer
    ad.features = scaler.fit_transform(ad.features)
    train.features = scaler.fit_transform(train.features)
    test.features = scaler.transform(test.features)

    if np.any(test.labels):
        print("-->True")
    else:
        print("-->not at least one label is True")

    protected_attributes = ad.protected_attribute_names




    di = DisparateImpactRemover(repair_level=0.)
    ad_repd = di.fit_transform(ad)

    for index in range(0, len(protected_attributes)):
        protected = protected_attributes[index]
        print("-->sens_attr", protected)

        # protected = 'sex'  # sex, race, sex, age, race
        # protected_object = ['Female']  # Male, White, male, lambda x: x > 25, Female, Caucasian

        index = train.feature_names.index(protected)
        X_tr = np.delete(train.features, index, axis=1)
        X_te = np.delete(test.features, index, axis=1)
        y_tr = train.labels.ravel()

        di = DisparateImpactRemover(repair_level=1.0)
        train_repd = di.fit_transform(train)
        test_repd = di.fit_transform(test)

        if np.all(train_repd.protected_attributes == train.protected_attributes):
            print("-->True")
        else:
            print("-->np.all(train_repd.protected_attributes != train.protected_attributes)")

        privileged_groups = [{protected: 1}]
        unprivileged_groups = [{protected: 0}]

        # dimension = len(X_tr[0])
        # lmod = initial_dnn(dimension)
        # lmod.fit(x=X_tr,y=y_tr, batch_size=BATCH_SIZE, epochs=EPOCHS)
        # lmod.save(MODEL_DIR)
        # lmod.load_weights(MODEL_DIR)

        # lmod = LogisticRegression(class_weight='balanced')
        # lmod = SVM(class_weight='balanced')
        # lmod = MLPClassifier()

        # lmod = MLPClassifier(solver='sgd', activation='identity', max_iter=500, alpha=1e-5,
        #                      hidden_layer_sizes=(64, 32, 16, 8, 4),
        #                      random_state=1, verbose=True)
        lmod = MLPClassifier(solver='adam', activation='identity', max_iter=100, alpha=1e-5,
                                                            hidden_layer_sizes=(64, 32, 16, 8, 4),
                                                            random_state=1, verbose=True)

        lmod.fit(X_tr, y_tr)
        # lmod.save(MODEL_DIR)

        test_pred = test.copy()
        test_pred.labels = lmod.predict(X_te)
        # test_pred.labels = lmod.predict_classes(X_te)

        print("----------" + "test on test data" + "----------")
        cm = ClassificationMetric(test, test_pred, privileged_groups=privileged_groups,
                                  unprivileged_groups=unprivileged_groups)
        before = cm.disparate_impact()
        # print("-->prediction accuracy on test data",accuracy_score(list(test), list(test_pred)))
        print('Disparate impact: {:.4}'.format(before))
        print('Acc overall: {:.4}'.format(cm.accuracy()))

        # thresh_arr = np.linspace(0.01, 0.5, 50)
        thresh_arr = np.array([0.5])
        # orig_metrics = metric_test1(dataset=X_te,
        #                             model=lmod,
        #                             thresh_arr=thresh_arr,
        #                             unprivileged_groups=unprivileged_groups,
        #                             privileged_groups=privileged_groups)
        orig_metrics = get_metrics(test, test_pred, privileged_groups=privileged_groups,
                                   unprivileged_groups=unprivileged_groups)
        lr_orig_best_ind = np.argmax(orig_metrics['bal_acc'])

        describe_metrics(orig_metrics, thresh_arr)

        X_tr_repd = np.delete(train_repd.features, index, axis=1)
        X_te_repd = np.delete(test_repd.features, index, axis=1)
        y_tr_repd = train_repd.labels.ravel()
        if (y_tr == y_tr_repd).all():
            print("-->True")
        else:
            print("-->t_tr != t_tr_repd.all")

        # dimension = len(X_tr[0])
        # lmod = initial_dnn(dimension)
        # lmod.fit(x=X_tr_repd,y=y_tr_repd, batch_size=BATCH_SIZE, epochs=EPOCHS)
        # lmod.save(MODEL_DIR)
        # lmod.load_weights(MODEL_DIR)

        lmod = MLPClassifier(solver='adam', activation='identity', max_iter=100, alpha=1e-5,
                             hidden_layer_sizes=(64, 32, 16, 8, 4),
                             random_state=1, verbose=True)
        lmod.fit(X_tr_repd, y_tr_repd)
        # lmod.save(MODEL_DIR)

        test_repd_pred = test_repd.copy()
        test_repd_pred.labels = lmod.predict(X_te_repd)
        # test_repd_pred.labels = lmod.predict_classes(X_te_repd)

        print("----------" + "test on test data after disparate impatct removing" + "----------")
        repaired_cm = ClassificationMetric(test_repd, test_repd_pred, privileged_groups=privileged_groups,
                                           unprivileged_groups=unprivileged_groups)
        after = repaired_cm.disparate_impact()
        print('Disparate impact: {:.4}'.format(after))
        print('Acc overall: {:.4}'.format(repaired_cm.accuracy()))

        print("-->for transformed test data")
        # thresh_arr = np.linspace(0.01, 0.5, 50)
        thresh_arr = np.array([0.5])
        # repd_metrics = metric_test1(dataset=X_te_repd,
        #                             model=lmod,
        #                             thresh_arr=thresh_arr,
        #                             unprivileged_groups=unprivileged_groups,
        #                             privileged_groups=privileged_groups)

        repd_metrics = get_metrics(test_repd, test_repd_pred, privileged_groups=privileged_groups,
                                   unprivileged_groups=unprivileged_groups)
        lr_orig_best_ind = np.argmax(repd_metrics['bal_acc'])
        describe_metrics(repd_metrics, thresh_arr)

        # print("-->for original test data")
        # test_pred1 = test.copy()
        # test_pred1.labels = lmod.predict(X_te)
        # val_metrics1 = get_metrics(test, test_pred1, privileged_groups=privileged_groups,
        #                            unprivileged_groups=unprivileged_groups)
        # describe_metrics(val_metrics1, thresh_arr)

        # if after > before:
        #     print("-->True")
        #     print("-->difference:", after - before)
        # else:
        #     print("-->after < before")
        # if abs(1 - after) <= 0.2:
        #     print("-->True")
        # else:
        #     print("-->after > 0.8")

        plot_class = Plot(dataset_name=dataset_name, sens_attr=protected, processing_name=processing_name)
        # Plot.plot_acc_metric(lr_orig_metrics, lr_transf_metrics, "stat_par_diff")
        multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
        plot_class.plot_acc_multi_metric(orig_metrics, repd_metrics, multi_metric_names)

dataset_name = "Adult income"
# metric_test(dataset_name)
DI_metric_test(dataset_name)

