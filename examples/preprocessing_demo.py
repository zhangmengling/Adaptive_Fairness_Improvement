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

processing_name = str(os.path.basename(__file__)).split("_demo")[0]

"""
global variables
"""
np.random.seed(1)
tf.random.set_random_seed(1)
nb_classes = 2
BATCH_SIZE = 32
EPOCHS = 500
# MODEL_DIR = "census_income_original.h5"
# MODEL_TRANS_DIR = "census_income_weight(sex).h5"

dataset_name = "German credit"
(dataset_orig_panel19_train,
 dataset_orig_panel19_val,
 dataset_orig_panel19_test) = GermanDataset().split([0.5, 0.8], shuffle=True)  # CompasDataset_1

print("-->dataset_orig_panel19_train", dataset_orig_panel19_train)
print(dataset_orig_panel19_train.protected_attribute_names)

#########  use sensitive index to fix the sensitive considered   #########
sens_ind = 1
sens_attr = dataset_orig_panel19_train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in
                       dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
print("-->unprivileged_groups", unprivileged_groups)
privileged_groups = [{sens_attr: v} for v in
                     dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]
print("-->privileged_groups", privileged_groups)

def describe(train=None, val=None, test=None):
    if train is not None:
        display(Markdown("#### Training Dataset shape"))
        print(train.features.shape)
    if val is not None:
        display(Markdown("#### Validation Dataset shape"))
        print(val.features.shape)
    display(Markdown("#### Test Dataset shape"))
    print(test.features.shape)
    display(Markdown("#### Favorable and unfavorable labels"))
    print(test.favorable_label, test.unfavorable_label)
    display(Markdown("#### Protected attribute names"))
    print(test.protected_attribute_names)
    display(Markdown("#### Privileged and unprivileged protected attribute values"))
    print(test.privileged_protected_attributes,
          test.unprivileged_protected_attributes)
    display(Markdown("#### Dataset feature names"))
    print(test.feature_names)

metric_orig_panel19_train = BinaryLabelDatasetMetric(
        dataset_orig_panel19_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
explainer_orig_panel19_train = MetricTextExplainer(metric_orig_panel19_train)

print(explainer_orig_panel19_train.disparate_impact())



### 3.2. Learning a Logistic Regression (LR) classifier on original data

#### 3.2.1. Training LR model on original data


dataset = dataset_orig_panel19_train
# model = make_pipeline(StandardScaler(),
#                       LogisticRegression(solver='liblinear', random_state=1))
# fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
#
# lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

# labels = np.array([x-1 for x in list(dataset.labels.ravel())])
labels = dataset.labels.ravel()
# original label: dataset.labels.ravel()

def initial_dnn(dim):
    model = Sequential()
    # model.add(Input(shape=x_train.shape))
    ## need to change the input shape of each datsete
    ## adult_income: 98; german_credit: 58
    # kernel_initializer = 'random_uniform',bias_initializer = 'zeros',
    model.add(InputLayer(input_shape=(dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    loss = tf.keras.losses.sparse_categorical_crossentropy
    metrics = tf.keras.metrics.categorical_accuracy
    model.compile(loss=loss, metrics=[metrics], optimizer='adam')
    return model

# def initial_dnn(dim):
#     model = Sequential()
#     # model.add(Input(shape=x_train.shape))
#     ## need to change the input shape of each datsete
#     ## adult_income:98; german_credit:58; bank:57; compas:401
#     model.add(InputLayer(input_shape=(dim,)))
#     model.add(Dense(64))
#     model.add(LeakyReLU(alpha=0.05))
#     # model.add(BatchNormalization())
#     model.add(Dense(32))
#     model.add(LeakyReLU(alpha=0.05))
#     # model.add(BatchNormalization())
#     model.add(Dense(16))
#     model.add(LeakyReLU(alpha=0.05))
#     # model.add(BatchNormalization())
#     model.add(Dense(8))
#     model.add(LeakyReLU(alpha=0.05))
#     # model.add(BatchNormalization())
#     model.add(Dense(4))
#     model.add(LeakyReLU(alpha=0.05))
#     # model.add(BatchNormalization())
#     model.add(Flatten())
#     model.add(Dense(nb_classes, activation='softmax'))
#     loss = tf.keras.losses.sparse_categorical_crossentropy
#     # model.add(tf.keras.regularizers.l2(0.01))
#     metrics = tf.keras.metrics.categorical_accuracy
#     model.compile(loss=loss, metrics=[metrics], optimizer='adam')
#     return model

dimension = len(dataset.features[0])
model = initial_dnn(dimension)
sample_weight=dataset.instance_weights,
model.fit(x=dataset.features,y=labels, sample_weight=dataset.instance_weights, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
# model.save(MODEL_DIR)

# model = initial_dnn()
# model.load_weights(MODEL_DIR)

lr_orig_panel19 = model

print("-->dataset labels")
print(list(dataset.labels.ravel()))
print(list(lr_orig_panel19.predict_classes(dataset.features)))

print("-->prediction accuracy on dataset", accuracy_score(list(dataset.labels.ravel()),
                                                        list(lr_orig_panel19.predict_classes(dataset.features))))
y_test = dataset_orig_panel19_test.labels.ravel()
y_pred = lr_orig_panel19.predict_classes(dataset_orig_panel19_test.features)
print("-->prediction accuracy on test data",accuracy_score(list(y_test), list(y_pred)))

#### 3.2.2. Validating LR model on original data

from collections import defaultdict

def metric_test(dataset, model, thresh_arr):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            dataset, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())

    return metric_arrs


def metric_test1(dataset, model, thresh_arr):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        all_classes = np.array([0, 1])
        pos_ind = np.where(all_classes == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        # changed coding
        pos_ind = 1
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        # changed coding
        # dataset_pred.labels = model.predict_classes(dataset.features)

        metric = ClassificationMetric(
            dataset, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
        metric_arrs['tpr'].append(metric.true_positive_rate())
        metric_arrs['tnr'].append(metric.true_negative_rate())
        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    return metric_arrs


# %%

# thresh_arr = np.linspace(0.01, 0.5, 50)
thresh_arr = np.array([0.5])
print("----------" + "test on val data" + "----------")
val_metrics = metric_test1(dataset=dataset_orig_panel19_val,
                   model=lr_orig_panel19,
                   thresh_arr=thresh_arr)
lr_orig_best_ind = np.argmax(val_metrics['bal_acc'])

def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(0.5, 0.8)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    if 'DI' in y_right_name:
        ax2.set_ylim(0., 0.7)
    else:
        ax2.set_ylim(-0.25, 0.1)

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    # plt.show()


disp_imp = np.array(val_metrics['disp_imp'])
disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
# plot(thresh_arr, 'Classification Thresholds',
#      val_metrics['bal_acc'], 'Balanced Accuracy',
#      disp_imp_err, '1 - min(DI, 1/DI)')
#
# #%%
#
# plot(thresh_arr, 'Classification Thresholds',
#      val_metrics['bal_acc'], 'Balanced Accuracy',
#      val_metrics['avg_odds_diff'], 'avg. odds diff.')


def describe_metrics(metrics, thresh_arr):
    best_ind = np.argmax(metrics['bal_acc'])
    print("True positive rate: {:6.4f}".format(metrics['tpr'][best_ind]))
    print("True negative rate: {:6.4f}".format(metrics['tnr'][best_ind]))
    print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
#     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))

describe_metrics(val_metrics, thresh_arr)

print("----------" + "test on test data" + "----------")
lr_orig_metrics = metric_test1(dataset=dataset_orig_panel19_test,
                       model=lr_orig_panel19,
                       thresh_arr=[thresh_arr[lr_orig_best_ind]])
#%%

describe_metrics(lr_orig_metrics, [thresh_arr[lr_orig_best_ind]])


# dataset = dataset_orig_panel19_train
# model = make_pipeline(StandardScaler(),
#                       RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
# fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}
# rf_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)



print("---------- Reweighing ----------")

print("-->unprivileged_groups", unprivileged_groups)
print("-->privileged_groups", privileged_groups)

RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf_panel19_train = RW.fit_transform(dataset_orig_panel19_train)

print("-->dataset_transf_panel19_train", dataset_transf_panel19_train)
print(type(dataset_transf_panel19_train))
print("-->insance weights:", list(dataset_transf_panel19_train.instance_weights))


metric_transf_panel19_train = BinaryLabelDatasetMetric(
        dataset_transf_panel19_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
explainer_transf_panel19_train = MetricTextExplainer(metric_transf_panel19_train)

print(explainer_transf_panel19_train.disparate_impact())
print(explainer_transf_panel19_train.statistical_parity_difference())

num = 0
for index in range(len(dataset_orig_panel19_train.features)):
    original_data = dataset_orig_panel19_train.features[index]
    trans_data = dataset_orig_panel19_train.features[index]
    if original_data.all() != trans_data.all():
        num += 1


### 4.2. Learning a Logistic Regression (LR) classifier on data transformed by reweighing

print("---------- Learning a Logistic Regression (LR) classifier on data transformed by reweighing ----------")

#### 4.2.1. Training LR model after reweighing

#%%

dataset = dataset_transf_panel19_train

from sklearn.neural_network import MLPClassifier

# clf = MLPClassifier(solver='sgd', activation='identity', max_iter=10, alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 8, 4),
#                     random_state=1, verbose=True)

dimension = len(dataset.features[0])
model = initial_dnn(dimension)
model.fit(x=dataset.features,y=labels, sample_weight=dataset.instance_weights,
          batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
# model.save(MODEL_TRANS_DIR)

# model.load_weights(MODEL_TRANS_DIR)

# print("-->dataset labels")
# print(list(dataset.labels.ravel()))
# print(list(reconstructed_model.predict_classes(dataset.features)))

lr_transf_panel19 = model
'''
model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
# lr_transf_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)
lr_transf_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)
'''
print("-->dataset labels")
print(list(dataset.labels.ravel()))
print(list(lr_transf_panel19.predict_classes(dataset.features)))

print("-->prediction accuracy on dataset", accuracy_score(list(dataset.labels.ravel()),
                                                        list(lr_transf_panel19.predict_classes(dataset.features))))
y_test = dataset_orig_panel19_test.labels.ravel()
y_pred = lr_transf_panel19.predict_classes(dataset_orig_panel19_test.features)
print("-->prediction accuracy on test data",accuracy_score(list(y_test), list(y_pred)))

def true_postive_rate(orig_labels, pred_labels):
    true_postive = 0
    false_nagative = 0
    for i in range(len(orig_labels)):
        if orig_labels[i] == pred_labels[i] and orig_labels[i] == 1:
            true_postive += 1
        elif orig_labels[i] == 1 and pred_labels[i] == 0:
            false_nagative += 1
    return true_postive/(true_postive+false_nagative)

print("-->true_postive_rate", true_postive_rate(list(y_test), list(y_pred)))


print("---------- Validating LR model after reweighing ----------")


# thresh_arr = np.linspace(0.01, 0.5, 50)
thresh_arr = np.array([0.5])
val_metrics = metric_test1(dataset=dataset_orig_panel19_val,
                   model=lr_transf_panel19,
                   thresh_arr=thresh_arr)
lr_transf_best_ind = np.argmax(val_metrics['bal_acc'])


disp_imp = np.array(val_metrics['disp_imp'])
disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
# plot(thresh_arr, 'Classification Thresholds',
#      val_metrics['bal_acc'], 'Balanced Accuracy',
#      disp_imp_err, '1 - min(DI, 1/DI)')
#
#
# plot(thresh_arr, 'Classification Thresholds',
#      val_metrics['bal_acc'], 'Balanced Accuracy',
#      val_metrics['avg_odds_diff'], 'avg. odds diff.')

#%%

describe_metrics(val_metrics, thresh_arr)



print("---------- Testing  LR model after reweighing ----------")

#%%

lr_transf_metrics = metric_test1(dataset=dataset_orig_panel19_test,
                         model=lr_transf_panel19,
                         thresh_arr=[thresh_arr[lr_transf_best_ind]])

describe_metrics(lr_transf_metrics, [thresh_arr[lr_transf_best_ind]])

def plot_acc_metric(orig_metrics, improved_metrics, metric_name):
    orig_best_ind = np.argmax(orig_metrics['bal_acc'])
    improved_best_ind = np.argmax(improved_metrics['bal_acc'])
    orig_acc = orig_metrics['bal_acc'][orig_best_ind]
    orig_metric = abs(orig_metrics[metric_name][orig_best_ind])
    improved_acc = improved_metrics['bal_acc'][improved_best_ind]
    improved_metric = abs(improved_metrics[metric_name][improved_best_ind])
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    title = "RW-" + dataset_name + "-" + str(sens_attr)
    plt.title(title)
    plt.xlabel("accuracy")
    plt.ylabel(metric_name)
    plt.plot(orig_acc, orig_metric, 'bo', label="original mdoel")
    plt.plot(improved_acc, improved_metric, 'ro', label="improved model")
    # plt.show()
    file_path = os.path.abspath(os.path.dirname(__file__))
    file_name = file_path + "/plotting_result/" + title + ".png"
    plt.savefig(file_name)

plot_acc_metric(lr_orig_metrics ,lr_transf_metrics, "stat_par_diff")


# dataset_name = "German credit"
# from plot_result import Plot
# Plot = Plot(dataset_name=dataset_name, sens_attr=sens_attr, processing_name=processing_name)
# Plot.plot_acc_metric(lr_orig_metrics, lr_transf_metrics, "stat_par_diff")




