import os.path
import sys
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

from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs

processing_name = str(os.path.basename(__file__)).split("_demo")[0]

"""
global variables
"""
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
nb_classes = 2
BATCH_SIZE = 128
EPOCHS = 1000   # 500
MAX_NUM = 5000
# MODEL_DIR = "german_credit_original.h5"
# MODEL_TRANS_DIR = "german_credit_weight(sex).h5"

dataset_name = "Adult income"
# sens_ind = 1

(dataset_orig_panel19_train,
 dataset_orig_panel19_test) = AdultDataset().split([0.7], shuffle=True, seed=seed)  # CompasDataset_1

# scale_orig = StandardScaler()
# X_train = scale_orig.fit_transform(dataset_orig_panel19_train.features)
# y_train = dataset_orig_panel19_train.labels.ravel()

# print("-->dataset_orig_panel19_train", dataset_orig_panel19_train)
print("-->dataset protected attributes", dataset_orig_panel19_train.protected_attribute_names)

# privileged_groups = [{sens_attr: v} for v in
#                      dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]

# sens_attr = dataset_orig_panel19_train.protected_attribute_names[sens_ind]

# unprivileged_groups = [{sens_attr: v} for v in
#                        dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
# print("-->unprivileged_groups", unprivileged_groups)
# privileged_groups = [{sens_attr: v} for v in
#                      dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]
# print("-->privileged_groups", privileged_groups)

# Metric for the original training dataset
# metric_orig_panel19_train = BinaryLabelDatasetMetric(
#         dataset_orig_panel19_train,
#         unprivileged_groups=unprivileged_groups,
#         privileged_groups=privileged_groups)
# explainer_orig_panel19_train = MetricTextExplainer(metric_orig_panel19_train)
# print(explainer_orig_panel19_train.disparate_impact())

### 3.2. Learning a classifier on original data

#### 3.2.1. Training model on original data

dataset = dataset_orig_panel19_train
# model = make_pipeline(StandardScaler(),
#                       LogisticRegression(solver='liblinear', random_state=1))
# fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
#
# lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

# labels = np.array([x-1 for x in list(dataset.labels.ravel())])
labels = dataset.labels.ravel()


dimension = len(dataset.features[0])
model = initial_dnn(dimension)
sample_weight=dataset.instance_weights,
model.fit(x=dataset.features,y=labels, sample_weight=dataset.instance_weights, batch_size=BATCH_SIZE, epochs=EPOCHS)
# model.save(MODEL_DIR)

# model.load_weights(MODEL_DIR)

lr_orig_panel19 = model

# print("-->dataset labels")
# print(list(dataset.labels.ravel()))
# print(list(lr_orig_panel19.predict_classes(dataset.features)))

print("-->prediction accuracy on dataset", accuracy_score(list(dataset.labels.ravel()),
                                                        list(lr_orig_panel19.predict_classes(dataset.features))))
y_test = dataset_orig_panel19_test.labels.ravel()
y_pred = lr_orig_panel19.predict_classes(dataset_orig_panel19_test.features)
print("-->prediction accuracy on test data",accuracy_score(list(y_test), list(y_pred)))


print("---------- Reweighing ----------")

sens_inds = [0, 1]
for sens_ind in sens_inds:
    sens_attr = dataset_orig_panel19_train.protected_attribute_names[sens_ind]
    print("-->sensitive attribute", sens_attr)
    unprivileged_groups = [{sens_attr: v} for v in
                           dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
    print("-->unprivileged_groups", unprivileged_groups)
    privileged_groups = [{sens_attr: v} for v in
                         dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]











    new_inputs_priviledge = dataset_orig_panel19_train.generate_inputs(max_num=MAX_NUM, if_random=False,
                                                            privileged_groups=privileged_groups, if_priviledge=True)
    new_inputs_unpriviledge = dataset_orig_panel19_train.generate_inputs(max_num=MAX_NUM, if_random=False,
                                                                       privileged_groups=privileged_groups,
                                                                       if_priviledge=False)
    def new_inputs_to_dataset(new_inputs_priviledge, new_inputs_unpriviledge):
        new_inputs = new_inputs_priviledge + new_inputs_unpriviledge
        # classified_dataset, no_matter_dataset = AdultDataset().split(np.array([MAX_NUM*2]), shuffle=False)[0]
        classified_dataset = dataset_orig_panel19_train.copy()
        classified_dataset.features = np.array(new_inputs)
        classified_dataset.instance_names = [1]*(MAX_NUM*2)
        classified_dataset.instance_weights = np.array([1] * (MAX_NUM * 2))
        classified_dataset.protected_attributes = np.array([[input[classified_dataset.protected_attribute_indexs[0]],
                                                          input[classified_dataset.protected_attribute_indexs[0]]] for input in new_inputs])
        return classified_dataset

    classified_dataset = new_inputs_to_dataset(new_inputs_priviledge, new_inputs_unpriviledge)
    print(classified_dataset)
    # print("-->features", classified_dataset.features)
    # print("lables", classified_dataset.labels)
    # print("-->privileged_groups", privileged_groups)

    thresh_arr = np.array([0.5])
    print("----------" + "test on test data" + "----------")
    lr_orig_metrics = metric_test_new_inputs(dataset=classified_dataset,
                                   model=lr_orig_panel19,
                                   thresh_arr=thresh_arr,
                                   unprivileged_groups=unprivileged_groups,
                                   privileged_groups=privileged_groups
                                   )
    describe_metrics_new_inputs(lr_orig_metrics, thresh_arr)


    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_panel19_train = RW.fit_transform(dataset_orig_panel19_train)

    print("---------- Learning a classifier on data transformed by reweighing ----------")

    #### 4.2.1. Training model after reweighing

    dataset = dataset_transf_panel19_train

    # clf = MLPClassifier(solver='sgd', activation='identity', max_iter=10, alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 8, 4),
    #                     random_state=1, verbose=True)

    dimension = len(dataset.features[0])
    model = initial_dnn(dimension)
    model.fit(x=dataset.features, y=labels, sample_weight=dataset.instance_weights,
              batch_size=BATCH_SIZE, epochs=EPOCHS)
    # model.save(MODEL_TRANS_DIR)

    # model.load_weights(MODEL_TRANS_DIR)

    lr_transf_panel19 = model
    # print("-->dataset labels")
    # print(list(dataset.labels.ravel()))
    # print(list(lr_transf_panel19.predict_classes(dataset.features)))

    print("-->prediction accuracy on dataset", accuracy_score(list(dataset.labels.ravel()),
                                                              list(
                                                                  lr_transf_panel19.predict_classes(dataset.features))))
    y_test = dataset_orig_panel19_test.labels.ravel()
    y_pred = lr_transf_panel19.predict_classes(dataset_orig_panel19_test.features)
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))

    print("---------- Validating model after reweighing ----------")

    # thresh_arr = np.linspace(0.01, 0.5, 50)
    thresh_arr = np.array([0.5])

    print("---------- Testing model after reweighing ----------")

    lr_transf_metrics = metric_test_new_inputs(dataset=classified_dataset,
                                     model=lr_transf_panel19,
                                     thresh_arr=thresh_arr,
                                     unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups)

    describe_metrics_new_inputs(lr_transf_metrics, thresh_arr)








"""
    #### 3.2.2. testing model on original data
    # thresh_arr = np.linspace(0.01, 0.5, 50)
    thresh_arr = np.array([0.5])
    print("----------" + "test on test data" + "----------")
    lr_orig_metrics = metric_test1(dataset=dataset_orig_panel19_test,
                                   model=lr_orig_panel19,
                                   thresh_arr=thresh_arr,
                                   unprivileged_groups=unprivileged_groups,
                                   privileged_groups=privileged_groups
                                   )
    describe_metrics(lr_orig_metrics, thresh_arr)

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_panel19_train = RW.fit_transform(dataset_orig_panel19_train)

    # metric_transf_panel19_train = BinaryLabelDatasetMetric(
    #         dataset_transf_panel19_train,
    #         unprivileged_groups=unprivileged_groups,
    #         privileged_groups=privileged_groups)
    # explainer_transf_panel19_train = MetricTextExplainer(metric_transf_panel19_train)
    # print(explainer_transf_panel19_train.disparate_impact())
    # print(explainer_transf_panel19_train.statistical_parity_difference())

    print("---------- Learning a classifier on data transformed by reweighing ----------")

    #### 4.2.1. Training model after reweighing

    dataset = dataset_transf_panel19_train

    # clf = MLPClassifier(solver='sgd', activation='identity', max_iter=10, alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 8, 4),
    #                     random_state=1, verbose=True)

    dimension = len(dataset.features[0])
    model = initial_dnn(dimension)
    model.fit(x=dataset.features, y=labels, sample_weight=dataset.instance_weights,
              batch_size=BATCH_SIZE, epochs=EPOCHS)
    # model.save(MODEL_TRANS_DIR)

    # model.load_weights(MODEL_TRANS_DIR)

    lr_transf_panel19 = model
    # print("-->dataset labels")
    # print(list(dataset.labels.ravel()))
    # print(list(lr_transf_panel19.predict_classes(dataset.features)))

    print("-->prediction accuracy on dataset", accuracy_score(list(dataset.labels.ravel()),
                                                              list(
                                                                  lr_transf_panel19.predict_classes(dataset.features))))
    y_test = dataset_orig_panel19_test.labels.ravel()
    y_pred = lr_transf_panel19.predict_classes(dataset_orig_panel19_test.features)
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))

    print("---------- Validating model after reweighing ----------")

    # thresh_arr = np.linspace(0.01, 0.5, 50)
    thresh_arr = np.array([0.5])

    print("---------- Testing model after reweighing ----------")

    lr_transf_metrics = metric_test1(dataset=dataset_orig_panel19_test,
                                     model=lr_transf_panel19,
                                     thresh_arr=thresh_arr,
                                     unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups)

    describe_metrics(lr_transf_metrics, thresh_arr)
"""
"""
    from plot_result import Plot

    Plot = Plot(dataset_name=dataset_name, sens_attr=sens_attr, processing_name=processing_name)
    # Plot.plot_acc_metric(lr_orig_metrics, lr_transf_metrics, "stat_par_diff")
    multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
    Plot.plot_acc_multi_metric(lr_orig_metrics, lr_transf_metrics, multi_metric_names)
"""




