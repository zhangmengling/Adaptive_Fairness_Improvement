import os
import sys
sys.path.append("../")
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
from aif360.datasets.compas_dataset1 import CompasDataset_1
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
import numpy as np

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions \
    import load_preproc_data_adult, load_preproc_data_compas
from collections import defaultdict

from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import tensorflow as tf

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from aif360.metrics.metric_test import metric_test_without_thresh
from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, initial_dnn2, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from plot_result import Plot
names = locals()

# Odds equalizing post-processing algorithm
from tqdm import tqdm

import random
seed = 1
random.seed(seed)
np.random.seed(seed)
# tf.random.set_seed(seed)
MAX_NUM = 1000
processing_name = "CEO"

### Fairness metrics for original dataset


def CEO_metric_test(dataset_name):
    print("-->CEO_metric_test")
    print("-->CEO_metric_test")
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

    randseed = 12345679
    seed = 1
    np.random.seed(seed)

    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed=seed)

    # scale_orig = StandardScaler()
    # dataset_orig_train.features = scale_orig.fit_transform(dataset_orig_train.features)
    # dataset_orig_test.features = scale_orig.fit_transform(dataset_orig_test.features)

    # Placeholder for predicted and transformed datasets
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

    X_train = dataset_orig_train.features
    y_train = dataset_orig_train.labels.ravel()

    dimension = len(X_train[0])
    orig_model = initial_dnn2(dimension)
    orig_model.fit(x=X_train, y=y_train,
                   sample_weight=dataset_orig_train.instance_weights,
                   batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=1)

    MODEL_DIR = name + "/CEO_model.h5"
    orig_model.save(MODEL_DIR)

    # fav_idx = np.where(orig_model.classes_ == dataset_orig_train.favorable_label)[0][0]
    fav_idx = 1
    # y_train_pred_prob = orig_model.predict_proba(X_train)[:, fav_idx]
    y_train_pred_prob = orig_model.predict(X_train)[:, fav_idx]

    # X_test = scale_orig.transform(dataset_orig_test.features)
    X_test = dataset_orig_test.features
    # y_test_pred_prob = orig_model.predict_proba(X_test)[:, fav_idx]
    y_test_pred_prob = orig_model.predict(X_test)[:, fav_idx]

    class_thresh = 0.5
    dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
    dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

    y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
    y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
    y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
    dataset_orig_train_pred.labels = y_train_pred

    y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
    y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
    y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
    dataset_orig_test_pred.labels = y_test_pred


    y_test = dataset_orig_test.labels.ravel()
    # y_pred = orig_model.predict_classes(dataset_orig_test.features)
    y_predict = orig_model.predict(dataset_orig_test.features)
    y_pred = np.argmax(y_predict, axis=1)
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))

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

    print("----------" + "multivariate group metric test" + "----------")
    multi_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                   model=orig_model,
                                                   thresh_arr=thresh_arr,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups
                                                   )
    print(multi_group_metrics)

    print("----------" + "causal metric test" + "----------")
    multi_causal_metrics = metric_test_causal(dataset=dataset_orig_test,
                                              model=orig_model,
                                              thresh_arr=thresh_arr,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups
                                              )
    print(multi_causal_metrics)


    eo_model = CalibratedEqOddsPostprocessing(unprivileged_groups=unprivileged_groups,  # EqoddsPostprocessing
                                              privileged_groups=privileged_groups, seed=randseed)
    eo_model = eo_model.fit(dataset_orig_train, dataset_orig_train_pred)

    # dataset_transf_test_pred = eo_model.predict(dataset_orig_test_pred)

    debias_y_pred = eo_model.predict(dataset_orig_test).labels
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(debias_y_pred)))

    print("-->after transform validation and test data using post-processing")
    thresh_arr = [0.5]

    print("----------" + "test on test data" + "----------")
    multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                      model=eo_model,
                                                      thresh_arr=thresh_arr,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups
                                                      )
    describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)
    multi_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(debias_y_pred))]

    print("----------" + "multivariate group metric test" + "----------")
    multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                         model=eo_model,
                                                         thresh_arr=thresh_arr,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups
                                                         )
    print(multi_group_trans_metrics)

    print("----------" + "causal metric test" + "----------")
    multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                    model=eo_model,
                                                    thresh_arr=thresh_arr,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups
                                                    )
    print(multi_causal_trans_metrics)


    if len(privileged_groups) == 1:
        multi_orig_metrics = [multi_orig_metrics, multi_group_metrics, multi_causal_metrics]
        all_multi_orig_metrics = defaultdict(list)
        for to_merge in multi_orig_metrics:
            for key, value in to_merge.items():
                all_multi_orig_metrics[key].append(value[0])
        # multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics, multi_causal_trans_metrics]
        multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics, multi_causal_trans_metrics]
        all_multi_trans_metrics = defaultdict(list)
        for to_merge in multi_trans_metrics:
            for key, value in to_merge.items():
                # print("-->value", value)
                all_multi_trans_metrics[key].append(value[0])
        # print("-->all_multi_trans_metrics", all_multi_trans_metrics)
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

        X_train = dataset_orig_train.features
        y_train = dataset_orig_train.labels.ravel()
        dimension = len(X_train[0])
        orig_model = initial_dnn2(dimension)
        orig_model.fit(x=X_train, y=y_train,
                       sample_weight=dataset_orig_train.instance_weights,
                       batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=1)



        # fav_idx = np.where(orig_model.classes_ == dataset_orig_train.favorable_label)[0][0]
        fav_idx = 1
        # y_train_pred_prob = orig_model.predict_proba(X_train)[:, fav_idx]
        y_train_pred_prob = orig_model.predict(X_train)[:, fav_idx]

        # X_test = scale_orig.transform(dataset_orig_test.features)
        X_test = dataset_orig_test.features
        # y_test_pred_prob = orig_model.predict_proba(X_test)[:, fav_idx]
        y_test_pred_prob = orig_model.predict(X_test)[:, fav_idx]

        class_thresh = 0.5
        dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
        dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

        y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
        y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
        y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
        dataset_orig_train_pred.labels = y_train_pred

        y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
        y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
        y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
        dataset_orig_test_pred.labels = y_test_pred



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


        cpp_model = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
                                                   unprivileged_groups=unprivileged_groups,
                                                   seed=randseed)
        cpp_model = cpp_model.fit(dataset_orig_train, dataset_orig_train_pred)
        # debias_y_pred = cpp_model.predict(dataset_orig_test).labels
        # print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(debias_y_pred)))

        names["trans_" + str(sens_attr) + '_metrics'] = []
        print("----------" + "test on test data" + "----------")
        uni_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                        model=cpp_model,
                                                        thresh_arr=thresh_arr,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups
                                                        )
        describe_metrics_new_inputs(uni_orig_trans_metrics, thresh_arr)
        uni_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(debias_y_pred))]
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_orig_trans_metrics)


        print("----------" + "multivariate group metric test" + "----------")
        uni_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                           model=cpp_model,
                                                           thresh_arr=thresh_arr,
                                                           unprivileged_groups=unprivileged_groups,
                                                           privileged_groups=privileged_groups
                                                           )
        print(uni_group_trans_metrics)
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_group_trans_metrics)

        print("----------" + "causal metric test" + "----------")
        uni_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                      model=cpp_model,
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

    # print("--all results:")
    # print([dict(all_uni_orig_metrics[0]), dict(all_uni_trans_metrics[0]), dict(all_uni_orig_metrics[1]),
    #        dict(all_uni_trans_metrics[1]),
    #        dict(all_multi_orig_metrics), dict(all_multi_trans_metrics)])

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
CEO_metric_test(dataset_name)
