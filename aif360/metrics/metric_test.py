from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.datasets.compas_dataset1 import CompasDataset_1
from aif360.metrics import ClassificationMetric
from aif360.metrics.newinput_classification_metric import NewInputClassificationMetric
from aif360.metrics.causal_classification_metric import CausalClassficationMetric

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization, Activation
# import torch
from sklearn.metrics import accuracy_score
from collections import defaultdict
import matplotlib.pyplot as plt

import random
seed = 1
random.seed(seed)
np.random.seed(seed)
# tf.random.set_random_seed(seed)

nb_classes = 2

def initial_dnn2(dim):
    model = Sequential()
    model.add(InputLayer(input_shape=(dim,)))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    loss = tf.keras.losses.sparse_categorical_crossentropy
    metrics = tf.keras.metrics.categorical_accuracy
    model.compile(loss=loss, metrics=[metrics], optimizer='adam')  # adam, sgd
    return model

def initial_dnn(dim):
    model = Sequential()
    model.add(InputLayer(input_shape=(dim,)))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.05))
    # model.add(BatchNormalization())
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.05))
    # model.add(BatchNormalization())
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.05))
    # model.add(BatchNormalization())
    model.add(Dense(8))
    model.add(LeakyReLU(alpha=0.05))
    # model.add(BatchNormalization())
    model.add(Dense(4))
    model.add(LeakyReLU(alpha=0.05))
    # model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    loss = tf.keras.losses.sparse_categorical_crossentropy
    metrics = tf.keras.metrics.categorical_accuracy
    model.compile(loss=loss, metrics=[metrics], optimizer='adam')
    return model

def metric_test(dataset, model, thresh_arr, unprivileged_groups, privileged_groups):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except:
        try:
            y_val_pred_prob = model.predict(dataset.features)
        except:
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

        metric_arrs['tpr'].append(metric.true_positive_rate())
        metric_arrs['tnr'].append(metric.true_negative_rate())
        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
        metric_arrs['acc'].append(metric.accuracy())
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    return metric_arrs


def metric_test1(dataset, model, thresh_arr, unprivileged_groups, privileged_groups):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        all_classes = np.array([0, 1])
        pos_ind = np.where(all_classes == dataset.favorable_label)[0][0]
        pos_ind = 1
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        # changed coding
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
        try:
            metric_arrs['acc'].append(accuracy_score(list(dataset.labels.ravel()),list(model.predict_classes(dataset.features))))
        except:
            try:
                metric_arrs['acc'].append(accuracy_score(list(dataset.labels.ravel()), list(model.predict(dataset.features))))
            except:
                metric_arrs['acc'].append(
                    accuracy_score(list(dataset.labels.ravel()), list(model.predict(dataset)).labels))
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    return metric_arrs


def metric_test_new_inputs(dataset, model, thresh_arr, unprivileged_groups, privileged_groups, **kwargs):
    if thresh_arr == None:
        print("-->thresh_attr is None")
        dataset_pred = kwargs['dataset_pred']
        y_val_pred_prob = dataset_pred
        try:
            metric_arrs = defaultdict(list)
            metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
            metric_arrs['disp_imp'].append(metric.disparate_impact())
            metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
            metric_arrs['average_odds_difference'].append(metric.average_odds_difference())
            metric_arrs['generalized_entropy_index'].append(metric.generalized_entropy_index())
            metric_arrs['false_positive_rate'].append(metric.false_positive_rate_difference())
            metric_arrs['false_negative_rate'].append(metric.false_negative_rate_difference())
            metric_arrs['acc'].append(metric.accuracy())
            print("-------return metrics without specify threshold")
            return metric_arrs
        except:
            thresh_arr = [0.5]
    else:
        try:
            # sklearn classifier
            y_val_pred_prob = model.predict_proba(dataset.features)
        except:
            try:
                y_val_pred_prob = model.predict_proba(dataset)
            except:
                try:
                    y_val_pred_prob = model.predict(dataset.features)
                except:
                    # aif360 inprocessing algorithm
                    y_val_pred_prob = model.predict(dataset).scores

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        # changed coding
        pos_ind = 1
        try:
           y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
        except:
            y_val_pred = (np.array(y_val_pred_prob) > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred

        metric = ClassificationMetric(
            dataset, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['average_odds_difference'].append(metric.average_odds_difference())
        metric_arrs['generalized_entropy_index'].append(metric.generalized_entropy_index())
        metric_arrs['false_positive_rate'].append(metric.false_positive_rate_difference())
        metric_arrs['false_negative_rate'].append(metric.false_negative_rate_difference())
        metric_arrs['accuracy'].append(metric.accuracy())
    return metric_arrs

def metric_test_multivariate(dataset, model, thresh_arr, unprivileged_groups, privileged_groups, **kwargs):
    """
    test univariate/multivariate group discrimination score
    dataset: generated new inputs
    model: model for prediction
    thresh_arr: [0.5] for default setting
    unprivileged_groups: e.g.[{'race': 0.0}, {'sex': 0.0}] for multivariate_group_discrimination
                         e.g.['race': 0.0] for univariate_group_discrimination
    privileged_groups: e.g.[{'race': 1.0}, {'sex': 1.0}]
                       e.g.[['race': 1.0]
    """
    if thresh_arr == None:
        dataset_pred = kwargs['dataset_pred']
        y_val_pred_prob = dataset_pred
        # metric_arrs = defaultdict(list)
        # metric = CausalClassficationMetric(
        #     dataset, dataset_pred,
        #     unprivileged_groups=unprivileged_groups,
        #     privileged_groups=privileged_groups)
        # if len(privileged_groups) == 1:
        #     univariate_group_discrimination = metric.univariate_group_discrimination()
        #     print("-->univariate_group_discrimination", univariate_group_discrimination)
        #     metric_arrs['group'].append(univariate_group_discrimination)
        # else:
        #     multivariate_group_discrimination = metric.multivariate_group_discrimination()
        #     print("-->multivariate_group_discrimination", multivariate_group_discrimination)
        #     metric_arrs['group'].append(multivariate_group_discrimination)
        # print("-------return metrics without specify threshold")
        # return metric_arrs
        try:
            metric_arrs = defaultdict(list)
            metric = CausalClassficationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
            if len(privileged_groups) == 1:
                univariate_group_discrimination = metric.univariate_group_discrimination()
                print("-->univariate_group_discrimination", univariate_group_discrimination)
                metric_arrs['group'].append(univariate_group_discrimination)
            else:
                multivariate_group_discrimination = metric.multivariate_group_discrimination()
                print("-->multivariate_group_discrimination", multivariate_group_discrimination)
                metric_arrs['group'].append(multivariate_group_discrimination)
            print("-------return metrics without specify threshold")
            return metric_arrs
        except:
            thresh_arr = [0.5]
    else:
        try:
            # sklearn classifier
            y_val_pred_prob = model.predict_proba(dataset.features)
            all_classes = np.array([0, 1])
            pos_ind = np.where(all_classes == dataset.favorable_label)[0][0]
        except AttributeError:
            try:
                y_val_pred_prob = model.predict(dataset.features)
            except:
                # aif360 inprocessing algorithm
                y_val_pred_prob = model.predict(dataset).scores
                pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        # changed coding
        pos_ind = 1
        try:
            y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
        except:
            y_val_pred = (np.array(y_val_pred_prob) > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred

        # print("-->y_val_pred", y_val_pred)
        # print("-->dataset_pred.labels", dataset_pred.labels)
        # changed coding
        # dataset_pred.labels = model.predict_classes(dataset.features)

        metric = CausalClassficationMetric(
            dataset, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        if len(privileged_groups) == 1:
            univariate_group_discrimination = metric.univariate_group_discrimination()
            print("-->univariate_group_discrimination", univariate_group_discrimination)
            metric_arrs['group'].append(univariate_group_discrimination)
        else:
            multivariate_group_discrimination = metric.multivariate_group_discrimination()
            print("-->multivariate_group_discrimination", multivariate_group_discrimination)
            metric_arrs['group'].append(multivariate_group_discrimination)

    return metric_arrs

def metric_test_causal(dataset, model, thresh_arr, unprivileged_groups, privileged_groups, **kwargs):
    """
    test univariate/multivariate causal discrimination score
    dataset: generated new inputs
    model: model for prediction
    thresh_arr: [0.5] for default setting
    unprivileged_groups: e.g.[{'race': 0.0}, {'sex': 0.0}] for multivariate_group_discrimination
                         e.g.['race': 0.0] for univariate_group_discrimination
    privileged_groups: e.g.[{'race': 1.0}, {'sex': 1.0}]
                       e.g.[['race': 1.0]
    """
    if thresh_arr == None:
        dataset_pred = kwargs['dataset_pred']
        y_val_pred_prob = dataset_pred

        try:
            metric_arrs = defaultdict(list)
            metric = CausalClassficationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

            if len(privileged_groups) == 1:
                try:
                    uni_causal = metric.univariate_causal_discrimination(model=model, thresh_arr=thresh_arr,
                                                                         scaler_protected_featal=scaler_protected_features)
                except:
                    uni_causal = metric.univariate_causal_discrimination(model=model, thresh_arr=thresh_arr)
                metric_arrs['causal'].append(uni_causal)
            else:
                try:
                    multi_causal = metric.multivariate_causal_discrimination(model=model, thresh_arr=thresh_arr,
                                                                             scaler_protected_features=scaler_protected_features)
                except:
                    multi_causal = metric.multivariate_causal_discrimination(model=model, thresh_arr=thresh_arr)
                print("-->multivariate_causal_discrimination", multi_causal)
                metric_arrs['causal'].append(multi_causal)
            print("-------return metrics without specify threshold")
            return metric_arrs
        except:
            thresh_arr = [0.5]
        try:
            scaler_protected_features = kwargs['scaler_protected_features']
        except:
            print("-->no scaler protected feature in metric_test_causal")
    else:
        try:
            # sklearn classifier
            y_val_pred_prob = model.predict_proba(dataset.features)
            all_classes = np.array([0, 1])
            pos_ind = np.where(all_classes == dataset.favorable_label)[0][0]
        except AttributeError:
            try:
                y_val_pred_prob = model.predict(dataset.features)
            except:
                # aif360 inprocessing algorithm
                y_val_pred_prob = model.predict(dataset).scores
                pos_ind = 0
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        # changed coding
        pos_ind = 1
        try:
           y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
        except:
            y_val_pred = (np.array(y_val_pred_prob) > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred

        # print("-->y_val_pred", y_val_pred)
        # print("-->dataset_pred.labels", dataset_pred.labels)
        # changed coding
        # dataset_pred.labels = model.predict_classes(dataset.features)

        metric = CausalClassficationMetric(
            dataset, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        if len(privileged_groups) == 1:
            try:
                uni_causal = metric.univariate_causal_discrimination(model=model, thresh_arr=thresh_arr, scaler_protected_featal=scaler_protected_features)
            except:
                uni_causal = metric.univariate_causal_discrimination(model=model, thresh_arr=thresh_arr)
            metric_arrs['causal'].append(uni_causal)
        else:
            try:
                multi_causal = metric.multivariate_causal_discrimination(model=model, thresh_arr=thresh_arr, scaler_protected_features=scaler_protected_features)
            except:
                multi_causal = metric.multivariate_causal_discrimination(model=model, thresh_arr=thresh_arr)
            print("-->multivariate_causal_discrimination", multi_causal)
            metric_arrs['causal'].append(multi_causal)
    return metric_arrs



def describe_metrics_new_inputs(metrics, thresh_arr):
    print("-->describing metrics: ")
#     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][0], 1/metrics['disp_imp'][0])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][0]))

    print("False positive rate difference:", metrics['false_positive_rate'])
    print("False negative rate difference", metrics['false_negative_rate'])




def get_metrics(dataset, dataset_pred, privileged_groups, unprivileged_groups):
    metric_arrs = defaultdict(list)
    metric = ClassificationMetric(
        dataset, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    metric_arrs['tpr'].append(metric.true_positive_rate())
    metric_arrs['tnr'].append(metric.true_negative_rate())
    metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                   + metric.true_negative_rate()) / 2)
    metric_arrs['acc'].append(metric.accuracy())
    metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
    metric_arrs['disp_imp'].append(metric.disparate_impact())
    metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
    metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
    metric_arrs['theil_ind'].append(metric.theil_index())
    return metric_arrs

def metric_test_without_thresh(dataset, dataset_pred, unprivileged_groups, privileged_groups):
    metric = ClassificationMetric(dataset,
                                    dataset_pred,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)
    metric_arrs = defaultdict(list)
    metric_arrs['tpr'].append(metric.true_positive_rate())
    metric_arrs['tnr'].append(metric.true_negative_rate())
    metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
    metric_arrs['acc'].append(metric.accuracy())
    metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
    metric_arrs['disp_imp'].append(metric.disparate_impact())
    metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
    metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
    metric_arrs['theil_ind'].append(metric.theil_index())
    return metric_arrs


def describe(train=None, val=None, test=None):
    if train is not None:
        print(train.features.shape)
    if val is not None:
        print(val.features.shape)
    print(test.features.shape)
    print(test.favorable_label, test.unfavorable_label)
    print(test.protected_attribute_names)
    print(test.privileged_protected_attributes,
          test.unprivileged_protected_attributes)
    print(test.feature_names)

def describe_metrics(metrics, thresh_arr):
    print("-->describing metrics: ")
    best_ind = np.argmax(metrics['bal_acc'])
    print("Accuracy: {:6.4f}".format(metrics['acc'][best_ind]))
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

