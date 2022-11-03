import sys
import os
sys.path.append("../")
import numpy as np
from tqdm import tqdm
from warnings import warn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
from aif360.datasets.compas_dataset1 import CompasDataset_1
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions \
    import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.postprocessing.reject_option_classification \
    import RejectOptionClassification
from common_utils import compute_metrics
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler


from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider
from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, initial_dnn2, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
from plot_result import Plot

names = locals()

import random
seed = 1
random.seed(seed)
np.random.seed(seed)
# tf.random.set_seed(seed)
MAX_NUM = 1000

nb_classes = 2
BATCH_SIZE = 32
EPOCHS = 500

processing_name = "RO"

def RO_metric_test(dataset_name):
    print("-->RO_metric_test")
    if dataset_name == "Adult income":
        # function = "load_preproc_data_adult"
        function = "AdultDataset"
        protected_attributes = ['race', 'sex']
        privileged_groups = [{'race': 1.0}, {'sex': 1.0}]
        unprivileged_groups = [{'race': 0.0}, {'sex': 0.0}]
    elif dataset_name == "German credit":
        # function = "load_preproc_data_german"
        function = "GermanDataset"
        protected_attributes = ['sex', 'age']
        privileged_groups = [{'sex': 1.0}, {'age': 1.0}]
        unprivileged_groups = [{'sex': 0.0}, {'age': 0.0}]
    elif dataset_name == "Bank":
        function = "BankDataset"
        protected_attributes = "age"
        privileged_groups = [{'age': 1.0}]
        unprivileged_groups = [{'age': 0.0}]
    else:
        # function = "load_preproc_data_compas"
        function = "CompasDataset_1"
        protected_attributes = ['sex', 'race']
        privileged_groups = [{'sex': 1.0}, {'race': 1.0}]
        unprivileged_groups = [{'sex': 0.0}, {'race': 0.0}]
        # privileged_groups = {'sex': [{'sex': 1}], 'race': [{'race': 1}]}
        # unprivileged_groups = {'sex': [{'sex': 0}], 'race': [{'race': 0}]}

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

    dataset_orig = eval(function)()

    # Metric used (should be one of allowed_metrics)
    metric_name = "Average odds difference"

    # Upper and lower bound on the fairness metric used
    metric_ub = 0.05
    metric_lb = -0.05

    randseed = 12345679

    allowed_metrics = ["Statistical parity difference",
                       "Average odds difference",
                       "Equal opportunity difference"]
    if metric_name not in allowed_metrics:
        raise ValueError("Metric name should be one of allowed metrics")

    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed=seed)

    # scale_orig = StandardScaler()
    # X_train = scale_orig.fit_transform(dataset_orig_train.features)
    # y_train = dataset_orig_train.labels.ravel()
    X_train = dataset_orig_train.features
    y_train = dataset_orig_train.labels.ravel()

    dimension = len(X_train[0])
    orig_model = initial_dnn2(dimension)
    orig_model.fit(x=X_train, y=y_train,
                   sample_weight=dataset_orig_train.instance_weights,
                   batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=1)

    MODEL_DIR = name + "/CEO_model.h5"
    orig_model.save(MODEL_DIR)


    # orig_model = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,
    #                      hidden_layer_sizes=(64, 32, 16, 8, 4),
    #                      random_state=1, verbose=True)  # identity， relu
    # orig_model.fit(X_train, y_train)
    # y_train_pred = orig_model.predict(X_train)

    # dimension = len(X_train[0])
    # lmod = initial_dnn(dimension)
    # lmod.fit(x=X_train,y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    # y_train_pred = lmod.predict(X_train)

    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    # dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

    # dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)

    """
    fav_idx = np.where(orig_model.classes_ == dataset_orig_train.favorable_label)[0][0]
    y_train_pred_prob = orig_model.predict_proba(X_train)[:, fav_idx]

    # Prediction probs for validation and testing data
    # X_valid = scale_orig.transform(dataset_orig_valid.features)
    # y_valid_pred_prob = orig_model.predict_proba(X_valid)[:, fav_idx]
    # X_test = scale_orig.transform(dataset_orig_test.features)
    X_test = dataset_orig_test.features
    y_test_pred_prob = orig_model.predict_proba(X_test)[:, fav_idx]

    class_thresh = 0.5
    dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
    # dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)
    dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

    y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
    y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
    y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
    dataset_orig_train_pred.labels = y_train_pred

    # y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
    # y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
    # y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
    # dataset_orig_valid_pred.labels = y_valid_pred

    y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
    y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
    y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
    dataset_orig_test_pred.labels = y_test_pred
    """

    y_test = dataset_orig_test.labels.ravel()
    # y_pred = orig_model.predict_classes(dataset_orig_test.features)
    y_predict = orig_model.predict(dataset_orig_test.features)
    y_pred = np.argmax(y_predict, axis=1)
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))


    #### Results before post-processing
    thresh_arr = [0.5]
    print("----------" + "test on test data" + "----------")
    multi_orig_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                model=orig_model,
                                                thresh_arr=thresh_arr,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups
                                                )
    describe_metrics_new_inputs(multi_orig_metrics, thresh_arr)
    multi_orig_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]

    # print("----------" + "multivariate group metric test" + "----------")
    # multi_group_metrics = metric_test_multivariate(dataset=classified_dataset,
    #                                                model=orig_model,
    #                                                thresh_arr=thresh_arr,
    #                                                unprivileged_groups=unprivileged_groups,
    #                                                privileged_groups=privileged_groups
    #                                                )
    # print(multi_group_metrics)
    print("----------" + "multivariate group metric test" + "----------")
    multi_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                   model=orig_model,
                                                   thresh_arr=thresh_arr,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups
                                                   )
    print(multi_group_metrics)

    # print("----------" + "causal metric test" + "----------")
    # multi_causal_metrics = metric_test_causal(dataset=classified_dataset,
    #                                           model=orig_model,
    #                                           thresh_arr=thresh_arr,
    #                                           unprivileged_groups=unprivileged_groups,
    #                                           privileged_groups=privileged_groups
    #                                           )
    # print(multi_causal_metrics)
    print("----------" + "causal metric test" + "----------")
    multi_causal_metrics = metric_test_causal(dataset=dataset_orig_test,
                                              model=orig_model,
                                              thresh_arr=thresh_arr,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups
                                              )
    print(multi_causal_metrics)

    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups,
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                     num_class_thresh=100, num_ROC_margin=50,
                                     metric_name=metric_name,
                                     metric_ub=metric_ub, metric_lb=metric_lb)
    ROC = ROC.fit(dataset_orig_train, dataset_orig_train_pred)

    debias_y_pred = ROC.predict(dataset_orig_test).labels
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(debias_y_pred)))

    print("-->after transform validation and test data using post-processing")
    thresh_arr = [0.5]

    print("----------" + "test on test data" + "----------")
    multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                      model=ROC,
                                                      thresh_arr=thresh_arr,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups
                                                      )
    describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)
    multi_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(debias_y_pred))]

    # print("----------" + "multivariate group metric test" + "----------")
    # multi_group_trans_metrics = metric_test_multivariate(dataset=classified_dataset,
    #                                                      model=ROC,
    #                                                      thresh_arr=thresh_arr,
    #                                                      unprivileged_groups=unprivileged_groups,
    #                                                      privileged_groups=privileged_groups
    #                                                      )
    # print(multi_group_trans_metrics)
    print("----------" + "multivariate group metric test" + "----------")
    multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                         model=ROC,
                                                         thresh_arr=thresh_arr,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups
                                                         )
    print(multi_group_trans_metrics)

    # print("----------" + "causal metric test" + "----------")
    # multi_causal_trans_metrics = metric_test_causal(dataset=classified_dataset,
    #                                                 model=ROC,
    #                                                 thresh_arr=thresh_arr,
    #                                                 unprivileged_groups=unprivileged_groups,
    #                                                 privileged_groups=privileged_groups
    #                                                 )
    # print(multi_causal_trans_metrics)
    print("----------" + "causal metric test" + "----------")
    multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                    model=ROC,
                                                    thresh_arr=thresh_arr,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups
                                                    )
    print(multi_causal_trans_metrics)

    if len(privileged_groups) == 1:
        multi_orig_metrics = [multi_orig_metrics, multi_group_metrics, multi_causal_metrics]
        # print("-->multi_orig_metrics", multi_orig_metrics)
        all_multi_orig_metrics = defaultdict(list)
        for to_merge in multi_orig_metrics:
            for key, value in to_merge.items():
                # print("-->value", value)
                all_multi_orig_metrics[key].append(value[0])
        print("-->all_multi_orig_metrics", all_multi_orig_metrics)
        # multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics, multi_causal_trans_metrics]
        multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics, multi_causal_trans_metrics]
        all_multi_trans_metrics = defaultdict(list)
        for to_merge in multi_trans_metrics:
            for key, value in to_merge.items():
                # print("-->value", value)
                all_multi_trans_metrics[key].append(value[0])
        print("-->all_multi_trans_metrics", all_multi_trans_metrics)
        print("--all results:")
        print([dict(all_multi_orig_metrics), dict(all_multi_trans_metrics)])
        return all_multi_orig_metrics, all_multi_trans_metrics, dataset_name, processing_name



    # univariate test
    all_uni_orig_metrics = []
    all_uni_trans_metrics = []
    sens_inds = [0, 1]
    for sens_ind in sens_inds:
        sens_attr = protected_attributes[sens_ind]
        print("-->sens_attr", sens_attr)
        privileged_groups = [{sens_attr: v} for v in
                             dataset_orig_train.privileged_protected_attributes[sens_ind]]
        unprivileged_groups = [{sens_attr: v} for v in
                               dataset_orig_train.unprivileged_protected_attributes[sens_ind]]
        print("-->unprivileged_groups", unprivileged_groups)
        print("-->privileged_groups", privileged_groups)

        thresh_arr = np.array([0.5])

        names["orig_" + str(sens_attr) + '_metrics'] = []
        print("----------" + "test on test data" + "----------")
        uni_orig_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                  model=orig_model,
                                                  thresh_arr=thresh_arr,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups
                                                  )
        describe_metrics_new_inputs(uni_orig_metrics, thresh_arr)
        uni_orig_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_orig_metrics)

        print("----------" + "multivariate group metric test" + "----------")
        uni_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                     model=orig_model,
                                                     thresh_arr=thresh_arr,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups
                                                     )
        print(uni_group_metrics)
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_group_metrics)

        print("----------" + "causal metric test" + "----------")
        uni_causal_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                model=orig_model,
                                                thresh_arr=thresh_arr,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups
                                                )
        uni_causal_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
        print(uni_causal_metrics)
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_causal_metrics)

        ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups,
                                         low_class_thresh=0.01, high_class_thresh=0.99,
                                         num_class_thresh=100, num_ROC_margin=50,
                                         metric_name=metric_name,
                                         metric_ub=metric_ub, metric_lb=metric_lb)
        ROC = ROC.fit(dataset_orig_train, dataset_orig_train_pred)

        debias_y_pred = ROC.predict(dataset_orig_test).labels
        print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(debias_y_pred)))

        names["trans_" + str(sens_attr) + '_metrics'] = []
        print("----------" + "test on test data" + "----------")
        uni_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                        model=ROC,
                                                        thresh_arr=thresh_arr,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups
                                                        )
        describe_metrics_new_inputs(uni_orig_trans_metrics, thresh_arr)
        uni_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(debias_y_pred))]
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_orig_trans_metrics)

        print("----------" + "multivariate group metric test" + "----------")
        uni_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                           model=ROC,
                                                           thresh_arr=thresh_arr,
                                                           unprivileged_groups=unprivileged_groups,
                                                           privileged_groups=privileged_groups
                                                           )
        print(uni_group_trans_metrics)
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_group_trans_metrics)

        print("----------" + "causal metric test" + "----------")
        uni_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                      model=ROC,
                                                      thresh_arr=thresh_arr,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups
                                                      )
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

    # print("-->uni_race_orig_metrics", all_uni_orig_metrics[0])
    # print("-->uni_race_trans_metrics", all_uni_trans_metrics[0])
    # print("-->uni_sex_orig_metrics", all_uni_orig_metrics[1])
    # print("-->uni_sex_trans_metrics", all_uni_trans_metrics[1])
    multi_orig_metrics = [multi_orig_metrics, multi_group_metrics, multi_causal_metrics]
    all_multi_orig_metrics = defaultdict(list)
    for to_merge in multi_orig_metrics:
        for key, value in to_merge.items():
            # print("-->value", value)
            all_multi_orig_metrics[key].append(value[0])
    # print("-->all_multi_orig_metrics", all_multi_orig_metrics)
    multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics, multi_causal_trans_metrics]
    all_multi_trans_metrics = defaultdict(list)
    for to_merge in multi_trans_metrics:
        for key, value in to_merge.items():
            # print("-->value", value)
            all_multi_trans_metrics[key].append(value[0])
    # print("-->all_multi_trans_metrics", all_multi_trans_metrics)

    from plot_result import Plot
    sens_attrs = dataset_orig_train.protected_attribute_names
    Plot = Plot(dataset_name=dataset_name, sens_attr=sens_attrs, processing_name=processing_name)
    # 1 image
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
RO_metric_test(dataset_name)
