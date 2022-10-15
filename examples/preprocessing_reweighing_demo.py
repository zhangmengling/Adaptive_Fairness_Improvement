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
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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

from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, initial_dnn2, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
names = locals()
# processing_name = str(os.path.basename(__file__)).split("_demo")[0]
processing_name = "RW"
"""
global variables
"""
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
# tf.random.seed(seed)
nb_classes = 2
# BATCH_SIZE = 16  #32, 128
# EPOCHS = 500   # 500, 1000
MAX_NUM = 1000  # 251 for causal discrimination test / 2000 for group discrimination test

def RW_metric_test(dataset_name):
    overall_starttime = time.time()
    print("-->RW_metric_test")
    if dataset_name == "Adult income":
        function = "AdultDataset"
        protected_attributes = ['race', 'sex']
        privileged_groups = [{'race': 1.0}, {'sex': 1.0}]
        unprivileged_groups = [{'race': 0.0}, {'sex': 0.0}]
    elif dataset_name == "German credit":
        function = "GermanDataset"
        protected_attributes = ['sex', 'age']
        privileged_groups = [{'sex': 1.0}, {'age': 1.0}]
        unprivileged_groups = [{'sex': 0.0}, {'age': 0.0}]
    elif dataset_name == "Bank":
        function = "BankDataset"
        protected_attributes = ['age']
        privileged_groups = [{'age': 1.0}]
        unprivileged_groups = [{'age': 0.0}]
    else:
        function = "CompasDataset_1"
        protected_attributes = ['sex', 'race']
        privileged_groups = [{'sex': 1.0}, {'race': 1.0}]
        unprivileged_groups = [{'sex': 0.0}, {'race': 0.0}]
        # all_privileged_groups = {'sex': [{'sex': 1}], 'race': [{'race': 1}]}
        # all_unprivileged_groups = {'sex': [{'sex': 0}], 'race': [{'race': 0}]}

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

    (train,
     test) = eval(function)().split([0.7], shuffle=True, seed=seed)

    dataset = test.features
    labels =test.labels.ravel()
    print("-->dataset", dataset)
    print("-->labels:", labels)

    print("-->length:", len(dataset))

    import statistics
    bound = []
    mean_deviation = []
    for i in range(0, len(dataset[0])):
        feature = list(np.array(dataset)[:,i])
        mean_deviation.append((statistics.mean(feature), statistics.stdev(feature)))
        bound.append((min(feature), max(feature)))
    print("-->bound", bound)
    print("-->mean_deviation",mean_deviation)


    data_dir = name + "/data_test/"
    for i in range(0, len(dataset)):
        data = list(dataset[i])
        data_path = data_dir + 'data' + str(i) + ".txt"
        print(data_path)
        f = open(data_path, "w")
        f.writelines(str(data))
        f.close()
    # data_dir = name + "/data/"
    # for i in range(0, 100):
    #     data = list(dataset[i])
    #     data_path = data_dir + 'data' + str(i) + ".txt"
    #     print(data_path)
    #     f = open(data_path, "w")
    #     f.writelines(str(data))
    #     f.close()

    label_path = name + "/data_test/" + "labels.txt"
    f = open(label_path, "w")
    f.writelines(str(list(labels)))
    f.close()

    # label_path = name + "/data/" + "labels.txt"
    # f = open(label_path, "w")
    # f.writelines(str(list(labels)[:100]))
    # f.close()



    (dataset_orig_panel19_train,
     dataset_orig_panel19_test) = eval(function)().split([0.7], shuffle=True, seed=seed)

    print("-->dataset protected attributes", protected_attributes)
    print("-->dataset_orig_panel19_train", dataset_orig_panel19_train.feature_names)

    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_panel19_train,
                                                                unprivileged_groups=unprivileged_groups,
                                                                privileged_groups=privileged_groups)
    print(
        "Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
    print("disparate impace = %f" % metric_orig_train.disparate_impact())
    print("statistical_parity_difference = %f" % metric_orig_train.statistical_parity_difference())

    dataset = dataset_orig_panel19_train
    labels = dataset.labels.ravel()

    # model = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,
    #                                                        hidden_layer_sizes=(64, 32, 16, 8, 4),
    #                                                        random_state=1, verbose=True)
    # model.fit(dataset.features, labels)

    dimension = len(dataset.features[0])
    model = initial_dnn2(dimension)
    model.fit(x=dataset.features, y=labels, sample_weight=dataset.instance_weights, batch_size=BATCH_SIZE,
              epochs=EPOCHS, shuffle=False, verbose=1)

    # dataset_orig_test_pred = dataset_orig_panel19_test.copy(deepcopy=True)
    # cm_pred_test = ClassificationMetric(dataset_orig_panel19_test, dataset_orig_test_pred,
    #                                     unprivileged_groups=unprivileged_groups,
    #                                     privileged_groups=privileged_groups)
    # print("Difference in GFPR between unprivileged and privileged groups")
    # print(cm_pred_test.difference(cm_pred_test.generalized_false_positive_rate))
    # print("Difference in GFNR between unprivileged and privileged groups")
    # print(cm_pred_test.difference(cm_pred_test.generalized_false_negative_rate))
    # print("-->accuracy:", cm_pred_test.accuracy())

    print("-->weights")
    i = 0
    for layer in model.layers:
        print("-->layer", layer)
        parameters = layer.get_weights()
        print("-->parameters:", parameters)
        # print("-->", list(parameters[0]))
        # print("-->", list(parameters[1]))

        if parameters != []:
            print(np.array(parameters[0]).shape)
            i += 1
            weights_dir = name + "/weights/w" + str(i) + ".txt"
            weight = [list(w) for w in list(parameters[0])]

            new_weight = []
            weight = np.array(weight)
            for j in range(0, list(weight.shape)[1]):
                new_weight.append(list(weight[:,j]))


            f = open(weights_dir, "w")
            f.writelines(str(new_weight))
            f.close()

            bias_dir = name + "/bias/b" + str(i) + ".txt"
            f = open(bias_dir, "w")
            f.writelines(str(list(parameters[1])))
            f.close()
        else:
            continue

    MODEL_DIR = name + "/model.h5"
    model.save(MODEL_DIR)

    # model.load_weights(MODEL_DIR)
    # model.load_model(MODEL_DIR)
    # model = keras.models.load_model(MODEL_DIR)

    lr_orig_panel19 = model

    print("-->prediction accuracy on dataset", accuracy_score(list(dataset.labels.ravel()),
                                                              list(np.argmax(lr_orig_panel19.predict(dataset.features), axis=1))))
    y_test = dataset_orig_panel19_test.labels.ravel()
    y_pred = np.argmax(lr_orig_panel19.predict(dataset_orig_panel19_test.features), axis=1)
    # y_pred = lr_orig_panel19.predict(dataset_orig_panel19_test.features)
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))

    print("-->unprivileged_groups", unprivileged_groups)
    print("-->privileged_groups", privileged_groups)


    time1 = time.time()
    thresh_arr = np.array([0.5])

    print("----------" + "test on test data" + "----------")
    multi_orig_metrics = metric_test_new_inputs(dataset=dataset_orig_panel19_test,
                                                model=lr_orig_panel19,
                                                thresh_arr=thresh_arr,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups
                                                )
    describe_metrics_new_inputs(multi_orig_metrics, thresh_arr)
    print(multi_orig_metrics)

    print("----------" + "multivariate group metric test" + "----------")
    multi_group_metrics = metric_test_multivariate(dataset=dataset_orig_panel19_test,
                                                   model=lr_orig_panel19,
                                                   thresh_arr=thresh_arr,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups
                                                   )
    print(multi_group_metrics)

    print("----------" + "causal metric test" + "----------")
    multi_causal_metrics = metric_test_causal(dataset=dataset_orig_panel19_test,
                                              model=lr_orig_panel19,
                                              thresh_arr=thresh_arr,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups
                                              )
    multi_causal_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
    print(multi_causal_metrics)

    time2 = time.time()

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_panel19_train = RW.fit_transform(dataset_orig_panel19_train)
    dataset = dataset_transf_panel19_train
    dimension = len(dataset.features[0])
    model = initial_dnn2(dimension)
    model.fit(x=dataset.features, y=labels, sample_weight=dataset.instance_weights, batch_size=BATCH_SIZE,
              epochs=EPOCHS, shuffle=False, verbose=0)
    lr_trans_panel19 = model

    weights1 = dataset.instance_weights
    dataset1 = dataset

    time3 = time.time()
    print("----------" + "test on test data" + "----------")
    multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_panel19_test,
                                                      model=lr_trans_panel19,
                                                      thresh_arr=thresh_arr,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups
                                                      )
    describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)
    print(multi_orig_trans_metrics)

    print("----------" + "multivariate group metric test" + "----------")
    multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_panel19_test,
                                                         model=lr_trans_panel19,
                                                         thresh_arr=thresh_arr,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups
                                                         )
    print(multi_group_trans_metrics)

    print("----------" + "causal metric test" + "----------")
    multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_panel19_test,
                                                    model=lr_trans_panel19,
                                                    thresh_arr=thresh_arr,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups
                                                    )
    accuracy = accuracy_score(list(dataset_orig_panel19_test.labels.ravel()),
                                                        list(np.argmax(lr_trans_panel19.predict(
                                                            dataset_orig_panel19_test.features), axis=1)))
    print("-->accuracy", accuracy)
    multi_causal_trans_metrics['acc'] = [accuracy]
    print(multi_causal_trans_metrics)

    if len(privileged_groups) == 1:
        multi_orig_metrics = [multi_orig_metrics, multi_group_metrics, multi_causal_metrics]
        print("-->multi_orig_metrics", multi_orig_metrics)
        all_multi_orig_metrics = defaultdict(list)
        for to_merge in multi_orig_metrics:
            for key, value in to_merge.items():
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
        print("time for processing:", time3-time2)
        return all_multi_orig_metrics, all_multi_trans_metrics, dataset_name, processing_name


    # univariate test
    all_uni_orig_metrics = []
    all_uni_trans_metrics = []

    sens_inds = [0, 1]
    for sens_ind in sens_inds:
        sens_attr = protected_attributes[sens_ind]
        print("-->sensitive attribute", sens_attr)
        names["orig_" + str(sens_attr) + '_metrics'] = []
        privileged_groups = [{sens_attr: v} for v in
                             dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]
        unprivileged_groups = [{sens_attr: v} for v in
                               dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
        print("-->unprivileged_groups", unprivileged_groups)
        print("-->privileged_groups", privileged_groups)

        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_panel19_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        print(
            "Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
        print("disparate impace = %f" % metric_orig_train.disparate_impact())
        print("statistical_parity_difference = %f" % metric_orig_train.statistical_parity_difference())

        thresh_arr = np.array([0.5])

        print("----------" + "test on test data" + "----------")
        uni_orig_metrics = metric_test_new_inputs(dataset=dataset_orig_panel19_test,
                                                  model=lr_orig_panel19,
                                                  thresh_arr=thresh_arr,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups
                                                  )
        describe_metrics_new_inputs(uni_orig_metrics, thresh_arr)
        uni_orig_metrics['acc'] = [accuracy_score(list(dataset_orig_panel19_test.labels.ravel()),
                                                  list(np.argmax(lr_orig_panel19.predict(
                                                      dataset_orig_panel19_test.features), axis=1)))]
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_orig_metrics)

        print("----------" + "univariate group metric test" + "----------")
        uni_group_metric = metric_test_multivariate(dataset=dataset_orig_panel19_test,
                                                    model=lr_orig_panel19,
                                                    thresh_arr=thresh_arr,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups
                                                    )
        print(uni_group_metric)
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_group_metric)

        print("----------" + "univariate causal metric test" + "----------")
        uni_causal_metric = metric_test_causal(dataset=dataset_orig_panel19_test,
                                               model=lr_orig_panel19,
                                               thresh_arr=thresh_arr,
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups
                                               )
        print(uni_causal_metric)
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_causal_metric)

        time4 = time.time()
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        dataset_transf_panel19_train = RW.fit_transform(dataset_orig_panel19_train)

        print("---------- Learning a classifier on data transformed by reweighing ----------")

        dataset = dataset_transf_panel19_train
        dimension = len(dataset.features[0])
        model = initial_dnn2(dimension)
        model.fit(x=dataset.features, y=dataset.labels.ravel(), sample_weight=dataset.instance_weights,
                  batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=0)

        if dataset1.features.all() == dataset.features.all():
            print("-->equal dataset")
        else:
            print(list(dataset1.features))
            print(list(dataset.features))

        if dataset1.labels.ravel().all() == dataset.features.ravel().all():
            print("-->equal labels")
        else:
            print(list(dataset1.labels.ravel()))
            print(list(dataset.features.ravel()))

        lr_transf_panel19 = model

        print("-->prediction accuracy on dataset", accuracy_score(list(dataset.labels.ravel()),
                                                                  list(np.argmax(lr_transf_panel19.predict(
                                                                          dataset.features), axis=1))))
        y_test = dataset_orig_panel19_test.labels.ravel()
        y_pred = np.argmax(lr_transf_panel19.predict(dataset_orig_panel19_test.features), axis=1)
        print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))

        time5 = time.time()
        # thresh_arr = np.linspace(0.01, 0.5, 50)
        thresh_arr = np.array([0.5])

        print("---------- Testing model after reweighing ----------")
        names["trans_" + str(sens_attr) + '_metrics'] = []

        print("----------" + "test on test data" + "----------")
        uni_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_panel19_test,
                                                        model=lr_transf_panel19,
                                                        thresh_arr=thresh_arr,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups
                                                        )
        describe_metrics_new_inputs(uni_orig_trans_metrics, thresh_arr)
        accuracy = accuracy_score(list(dataset_orig_panel19_test.labels.ravel()),
                                                        list(np.argmax(lr_transf_panel19.predict(
                                                            dataset_orig_panel19_test.features), axis=1)))
        print("-->accuracy", accuracy)
        uni_orig_trans_metrics['acc'] = [accuracy]
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_orig_trans_metrics)

        print("----------" + "univariate group metric test" + "----------")
        uni_group_trans_metric = metric_test_multivariate(dataset=dataset_orig_panel19_test,
                                                          model=lr_transf_panel19,
                                                          thresh_arr=thresh_arr,
                                                          unprivileged_groups=unprivileged_groups,
                                                          privileged_groups=privileged_groups
                                                          )
        print(uni_group_trans_metric)
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_group_trans_metric)

        print("----------" + "univariate causal metric test" + "----------")
        uni_causal_trans_metric = metric_test_causal(dataset=dataset_orig_panel19_test,
                                                     model=lr_transf_panel19,
                                                     thresh_arr=thresh_arr,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups
                                                     )
        print(uni_causal_trans_metric)
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_causal_trans_metric)

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

        print("time for protected attribute", sens_attr, time5 - time4)

    print("time for multivariate attributes", time3 - time2)
    print("-->results")
    print(all_uni_orig_metrics[0])
    print(all_uni_trans_metrics[0])
    print(all_uni_orig_metrics[1])
    print(all_uni_trans_metrics[1])
    multi_orig_metrics = [multi_orig_metrics, multi_group_metrics, multi_causal_metrics]
    all_multi_orig_metrics = defaultdict(list)
    for to_merge in multi_orig_metrics:
        for key, value in to_merge.items():
            # print("-->value", value)
            all_multi_orig_metrics[key].append(value[0])
    print(all_multi_orig_metrics)
    multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics, multi_causal_trans_metrics]
    all_multi_trans_metrics = defaultdict(list)
    for to_merge in multi_trans_metrics:
        for key, value in to_merge.items():
            # print("-->value", value)
            all_multi_trans_metrics[key].append(value[0])
    print(all_multi_trans_metrics)

    print("-->all results:")
    print([dict(all_uni_orig_metrics[0]), dict(all_uni_trans_metrics[0]), dict(all_uni_orig_metrics[1]), dict(all_uni_trans_metrics[1]),
           dict(all_multi_orig_metrics), dict(all_multi_trans_metrics)])

    from plot_result import Plot
    sens_attrs = protected_attributes
    Plot = Plot(dataset_name=dataset_name, sens_attr=sens_attrs, processing_name=processing_name)
    Plot.plot_abs_acc_all_metric(all_uni_orig_metrics[0], all_uni_trans_metrics[0], all_uni_orig_metrics[1],
                                 all_uni_trans_metrics[1], all_multi_orig_metrics, all_multi_trans_metrics)
    # # 2 images: one for group metric. one for causal metric
    # Plot.plot_abs_acc_multi_metric(all_uni_orig_metrics[0], all_uni_trans_metrics[0], all_uni_orig_metrics[1],
    #                                all_uni_trans_metrics[1], all_multi_orig_metrics, all_multi_trans_metrics)
    # # 3 images: one for 'race', one for 'sex', one for 'race,sex'
    # Plot.plot_abs_acc_individual_metric(all_uni_orig_metrics[0], all_uni_trans_metrics[0], all_uni_orig_metrics[1],
    #                                     all_uni_trans_metrics[1], all_multi_orig_metrics, all_multi_trans_metrics)

    return all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics, all_multi_trans_metrics, \
           dataset_name, sens_attrs, processing_name

dataset_name = "Adult income"
print(RW_metric_test(dataset_name))

