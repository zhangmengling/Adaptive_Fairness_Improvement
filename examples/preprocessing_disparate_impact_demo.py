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
tf.random.set_seed(seed)
# tf.random.set_random_seed(seed)
nb_classes = 2
# BATCH_SIZE = 32
# EPOCHS = 500
MAX_NUM = 1000  # 251 for causal discrimination test / 2000 for group discrimination test
names = locals()

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
        EPOCHS = 1000
    elif dataset_name == "Adult income":
        name = "Adult"
        BATCH_SIZE = 128
        EPOCHS = 1000
    elif dataset_name == "German credit":
        name = "Credit"
        BATCH_SIZE = 32
        EPOCHS = 500
    else:
        name = "Compas"
        BATCH_SIZE = 128
        EPOCHS = 1000

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

    train.features = train.features
    test.features = test.features

    if np.any(test.labels):
        print("-->True")
    else:
        print("-->not at least one label is True")

    indexs = [train.feature_names.index(protected_name) for protected_name in protected_attributes]
    print("-->indexs", indexs)

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

    test_pred = test.copy()
    test_pred.labels = orig_lmod.predict(X_te)
    orig_test_pred_prob = orig_lmod.predict(X_te)

    basic_time2 = time.time()

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
    multi_group_metrics['acc'] = [accuracy_score(list(y_te), list(np.argmax(orig_lmod.predict(X_te), axis=1)))]
    print(multi_group_metrics)

    print("----------" + "causal metric test" + "----------")
    multi_causal_metrics = metric_test_causal(dataset=test,
                                                    model=orig_lmod,
                                                    thresh_arr=None,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups,
                                                    dataset_pred=orig_test_pred_prob
                                                    )
    print(multi_causal_metrics)

    time2 = time.time()
    print("-->DI fitting")
    di = DisparateImpactRemover(repair_level=1.0, sensitive_attribute=protected_attributes)
    train_repd = di.fit_transform(train)

    X_tr_repd = train_repd.features
    y_tr_repd = train_repd.labels.ravel()

    dimension = len(X_tr_repd[0])
    trans_lmod = initial_dnn2(dimension)
    trans_lmod.fit(x=X_tr_repd, y=y_tr_repd, sample_weight=train_repd.instance_weights, batch_size=BATCH_SIZE,
                  epochs=EPOCHS, shuffle=False, verbose=0)

    print("-->X_tr_repd", X_tr_repd)
    print(list(X_tr_repd[0]))

    test_pred = test.copy()
    test_pred.labels = trans_lmod.predict(X_te)
    trans_test_pred_prob = trans_lmod.predict(X_te)

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
    multi_group_trans_metrics['acc'] = [accuracy_score(list(y_te), list(np.argmax(trans_lmod.predict(X_te), axis=1)))]
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
        all_multi_orig_metrics = defaultdict(list)
        for to_merge in multi_orig_metrics:
            for key, value in to_merge.items():
                all_multi_orig_metrics[key].append(value[0])
        # print("-->all_multi_orig_metrics", all_multi_orig_metrics)
        # multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics, multi_causal_trans_metrics]
        multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics]
        all_multi_trans_metrics = defaultdict(list)
        for to_merge in multi_trans_metrics:
            for key, value in to_merge.items():
                # print("-->value", value)
                all_multi_trans_metrics[key].append(value[0])
        return all_multi_orig_metrics, all_multi_trans_metrics, dataset_name, processing_name


    # univariate test
    all_uni_orig_metrics = []
    all_uni_trans_metrics = []
    for index in range(0, len(protected_attributes)):
        sens_attr = protected_attributes[index]
        print("-->sens_attr", sens_attr)

        privileged_groups = [{sens_attr: 1}]
        unprivileged_groups = [{sens_attr: 0}]

        index = train.feature_names.index(sens_attr)
        X_tr = train.features
        X_te = test.features
        y_tr = train.labels.ravel()

        dimension = len(X_tr[0])
        orig_lmod = initial_dnn2(dimension)
        orig_lmod.fit(x=X_tr, y=y_tr, sample_weight=train.instance_weights, batch_size=BATCH_SIZE,
                      epochs=EPOCHS, shuffle=False, verbose=0)

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
        uni_group_metrics['acc'] = [accuracy_score(list(y_te), list(np.argmax(orig_lmod.predict(X_te), axis=1)))]
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
        di = DisparateImpactRemover(repair_level=1.0, sensitive_attribute='')
        train_repd = di.fit_transform(train)
        test_repd = di.fit_transform(test)
        y_tr_repd = train_repd.labels.ravel()
        y_te_repd = test_repd.labels.ravel()

        dimension = len(X_tr_repd[0])
        trans_lmod = initial_dnn2(dimension)
        trans_lmod.fit(x=X_tr_repd, y=y_tr_repd, sample_weight=train_repd.instance_weights, batch_size=BATCH_SIZE,
                       epochs=EPOCHS, shuffle=False, verbose=0)

        trans_test_pred_prob = trans_lmod.predict(X_te)

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
        uni_group_trans_metrics['acc'] = [accuracy_score(list(y_te), list(np.argmax(orig_lmod.predict(X_te), axis=1)))]
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
        uni_causal_trans_metrics['acc'] = [accuracy_score(list(y_te), list(np.argmax(orig_lmod.predict(X_te), axis=1)))]
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


dataset_name = "Adult income"
DI_metric_test(dataset_name)

