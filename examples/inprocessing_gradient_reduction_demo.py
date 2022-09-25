import sys
sys.path.append("../")
import os
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.datasets import AdultDataset, CompasDataset, GermanDataset
from aif360.datasets.compas_dataset1 import CompasDataset_1

from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from aif360.metrics.metric_test import metric_test_without_thresh
from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, initial_dnn2, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
from plot_result import Plot

import numpy as np

import random
seed = 1
random.seed(seed)
np.random.seed(seed)
# tf.random.set_random_seed(seed)
MAX_NUM = 1000
processing_name = "GR"

names = locals()

def GR_metric_test(dataset_name):
    print("-->GR_metric_test")
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
        # privileged_groups = {'sex': [{'sex': 1}], 'age': [{'age': 1}]}
        # unprivileged_groups = {'sex': [{'sex': 0}], 'age': [{'age': 0}]}
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
        EPOCHS = 50
    elif dataset_name == "Adult income":
        name = "Adult"
        BATCH_SIZE = 128
        EPOCHS = 1000  # 1000
    elif dataset_name == "German credit":
        name = "Credit"
        BATCH_SIZE = 32
        EPOCHS = 500
    else:
        name = "Compas"
        BATCH_SIZE = 128
        EPOCHS = 100

    dataset_orig = eval(function)()
    # dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed=seed)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

    # min_max_scaler = MaxAbsScaler()
    # dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    # dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

    X_train = dataset_orig_train.features
    y_train = dataset_orig_train.labels.ravel()

    # model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
    # fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
    # lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)


    dimension = len(X_train[0])
    orig_model = initial_dnn2(dimension)
    orig_model.fit(x=X_train, y=y_train,
                   sample_weight=dataset_orig_train.instance_weights,
                   batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=1)

    MODEL_DIR = name + "/GR_model.h5"
    orig_model.save(MODEL_DIR)

    # orig_model = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,
    #                      hidden_layer_sizes=(64, 32, 16, 8, 4),
    #                      random_state=1, verbose=True)  # identity， relu
    # orig_model.fit(X_train, y_train)

    X_test = dataset_orig_test.features
    y_test = dataset_orig_test.labels.ravel()
    # y_pred = orig_model.predict_classes(X_test)
    predict_x = orig_model.predict(X_test)
    y_pred = np.argmax(predict_x, axis=1)

    print("-->accuracy:")
    lr_acc = accuracy_score(y_test, y_pred)
    print(lr_acc)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    dataset_orig_test_pred.labels = y_pred

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
        # print("-->columns", classified_dataset.column_names)
        return classified_dataset

    # multivariate test
    input_privileged = []
    input_unprivileged = []
    for i in range(0, len(privileged_groups)):
        privileged_group = [privileged_groups[i]]
        unprivileged_group = [unprivileged_groups[i]]
        # for group in privileged_groups:
        #     group = [group]
        new_inputs_priviledge = dataset_orig_train.generate_inputs(max_num=MAX_NUM, if_random=False,
                                                                           privileged_groups=privileged_group,
                                                                           if_priviledge=True)
        # new_inputs_priviledge = dataset_orig_train.generate_inputs(max_num=MAX_NUM, if_random=True,
        #                                                            privileged_groups=privileged_group,
        #                                                            if_priviledge=True)
        input_privileged += new_inputs_priviledge
        new_inputs_unpriviledge = dataset_orig_train.generate_inputs(max_num=MAX_NUM, if_random=False,
                                                                             privileged_groups=unprivileged_group,
                                                                             if_priviledge=True)
        # new_inputs_unpriviledge = dataset_orig_train.generate_inputs(max_num=MAX_NUM, if_random=True,
        #                                                              privileged_groups=unprivileged_group,
        #                                                              if_priviledge=True)
        input_unprivileged += new_inputs_unpriviledge

    new_inputs = input_privileged + input_unprivileged
    random.shuffle(new_inputs)
    classified_dataset = new_inputs_to_dataset(new_inputs, dataset_orig_train)
    print("-->classified_dataset", classified_dataset)
    # classified_dataset.features = min_max_scaler.fit_transform(classified_dataset.features)

    thresh_arr = np.array([0.5])
    # orig_metrics = metric_test1(dataset=dataset_orig_test,
    #                             model=orig_model,
    #                             thresh_arr=thresh_arr,
    #                             privileged_groups=privileged_groups,
    #                             unprivileged_groups=unprivileged_groups)
    # lr_orig_best_ind = np.argmax(orig_metrics['bal_acc'])
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
    multi_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,   # classified_dataset, dataset_orig_test
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

    print("-->Learning exponentiated gradient reduction model")

    estimator = LogisticRegression(solver='lbfgs')
    exp_grad_red = ExponentiatedGradientReduction(estimator=estimator,
                                                  constraints="DemographicParity",   # "DemographicParity" or "EqualizedOdds"
                                                  drop_prot_attr=False)
    exp_grad_red.fit(dataset_orig_train)
    exp_grad_red_pred = exp_grad_red.predict(dataset_orig_test)

    # metric_test = ClassificationMetric(dataset_orig_test,
    #                                    exp_grad_red_pred,
    #                                    unprivileged_groups=unprivileged_groups,
    #                                    privileged_groups=privileged_groups)
    # egr_acc = metric_test.accuracy()
    # print(egr_acc)

    X_test = dataset_orig_test.features
    y_test = dataset_orig_test.labels.ravel()
    y_trans_pred = exp_grad_red.predict(dataset_orig_test).labels
    lr_acc = accuracy_score(y_test, y_trans_pred)
    print("-->prediction accuracy on test data", lr_acc)

    # thresh_arr = np.array([0.5])
    # gr_metrics = metric_test_without_thresh(dataset=dataset_orig_test,
    #                                         dataset_pred=exp_grad_red_pred,
    #                                         privileged_groups=privileged_groups,
    #                                         unprivileged_groups=unprivileged_groups)
    #
    # print("-->Validating ExponentiatedGradientReduction model on original data")
    # describe_metrics(gr_metrics, thresh_arr)

    print("----------" + "test on test data" + "----------")
    multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                      model=exp_grad_red,
                                                      thresh_arr=thresh_arr,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups
                                                      )
    describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)
    multi_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(y_trans_pred))]

    print("----------" + "multivariate group metric test" + "----------")
    multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                         model=exp_grad_red,
                                                         thresh_arr=thresh_arr,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups
                                                         )
    print(multi_group_trans_metrics)

    print("----------" + "causal metric test" + "----------")
    multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                    model=exp_grad_red,
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

        estimator = LogisticRegression(solver='lbfgs')
        exp_grad_red = ExponentiatedGradientReduction(estimator=estimator,
                                                      constraints="DemographicParity",
                                                      # "DemographicParity" or "EqualizedOdds"
                                                      drop_prot_attr=False)
        dataset_train = dataset_orig_train.copy()
        dataset_train.protected_attribute_names = [sens_attr]
        exp_grad_red.fit(dataset_train)

        y_test = dataset_orig_test.labels.ravel()
        y_trans_pred = exp_grad_red.predict(dataset_orig_test).labels
        lr_acc = accuracy_score(y_test, y_trans_pred)
        print("-->prediction accuracy on test data", lr_acc)

        names["trans_" + str(sens_attr) + '_metrics'] = []
        print("----------" + "test on test data" + "----------")
        uni_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                        model=exp_grad_red,
                                                        thresh_arr=thresh_arr,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups
                                                        )
        describe_metrics_new_inputs(uni_orig_trans_metrics, thresh_arr)
        uni_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(y_trans_pred))]
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_orig_trans_metrics)

        print("----------" + "multivariate group metric test" + "----------")
        uni_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                           model=exp_grad_red,
                                                           thresh_arr=thresh_arr,
                                                           unprivileged_groups=unprivileged_groups,
                                                           privileged_groups=privileged_groups
                                                           )
        print(uni_group_trans_metrics)
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_group_trans_metrics)

        print("----------" + "causal metric test" + "----------")
        uni_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                      model=exp_grad_red,
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

    print("-->metric results")
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
    print("-->all_multi_orig_metrics", all_multi_orig_metrics)
    multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics, multi_causal_trans_metrics]
    all_multi_trans_metrics = defaultdict(list)
    for to_merge in multi_trans_metrics:
        for key, value in to_merge.items():
            # print("-->value", value)
            all_multi_trans_metrics[key].append(value[0])
    print("-->all_multi_trans_metrics", all_multi_trans_metrics)

    print("--all results:")
    print([dict(all_uni_orig_metrics[0]), dict(all_uni_trans_metrics[0]), dict(all_uni_orig_metrics[1]),
           dict(all_uni_trans_metrics[1]),
           dict(all_multi_orig_metrics), dict(all_multi_trans_metrics)])

    from plot_result import Plot
    sens_attrs = dataset_orig_train.protected_attribute_names
    Plot = Plot(dataset_name=dataset_name, sens_attr=sens_attrs, processing_name=processing_name)
    # Plot.plot_acc_metric(lr_orig_metrics, lr_transf_metrics, "stat_par_diff")
    multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
    # def plot_abs_acc_multi_metric(self, orig_uni_metrics1, improved_uni_metrics1, orig_uni_metrics2, improved_uni_metrics2,
    #         orig_multi_metrics, improved_multi_metrics):
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
# metric_test(dataset_name)
GR_metric_test(dataset_name)











def metric_test(dataset_name):
    if dataset_name == "Adult income":
        # function = "load_preproc_data_adult"
        function = "AdultDataset"
        protected_attributes = ['sex', 'race']
        all_privileged_groups = {'sex': [{'sex': 1}], 'race': [{'race': 1}]}
        all_unprivileged_groups = {'sex': [{'sex': 0}], 'race': [{'race': 0}]}
    elif dataset_name == "German credit":
        # function = "load_preproc_data_german"
        function = "GermanDataset"
        protected_attributes = ['sex', 'age']
        all_privileged_groups = {'sex': [{'sex': 1}], 'age': [{'age': 1}]}
        all_unprivileged_groups = {'sex': [{'sex': 0}], 'age': [{'age': 0}]}
    else:
        # function = "load_preproc_data_compas"
        function = "CompasDataset_1"
        protected_attributes = ['sex', 'race']
        all_privileged_groups = {'sex': [{'sex': 1}], 'race': [{'race': 1}]}
        all_unprivileged_groups = {'sex': [{'sex': 0}], 'race': [{'race': 0}]}

    # processing_name = str(os.path.basename(__file__)).split("_demo")[0]
    processing_name = "GR"

    dataset_orig = eval(function)()
    # dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed=seed)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

    for protected_attribute in protected_attributes:
        print("-->sens_attr", protected_attribute)
        privileged_groups = all_privileged_groups[protected_attribute]
        print("-->privileged_groups", privileged_groups)
        unprivileged_groups = all_unprivileged_groups[protected_attribute]
        print("-->privileged_groups", unprivileged_groups)

        # print out some labels, names, etc.
        # display(Markdown("#### Training Dataset shape"))
        print(dataset_orig_train.features.shape)
        # display(Markdown("#### Favorable and unfavorable labels"))
        print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
        # display(Markdown("#### Protected attribute names"))
        print(dataset_orig_train.protected_attribute_names)
        # display(Markdown("#### Privileged and unprivileged protected attribute values"))
        print(dataset_orig_train.privileged_protected_attributes,
              dataset_orig_train.unprivileged_protected_attributes)
        # display(Markdown("#### Dataset feature names"))
        print(dataset_orig_train.feature_names)

        # %% md

        #### Metric for original training data

        # %%

        # Metric for the original dataset
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        # display(Markdown("#### Original training dataset"))
        print("--> Original training dataset ")
        print(
            "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
        metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups)
        print(
            "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())

        # %%

        # min_max_scaler = MaxAbsScaler()
        # dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
        # dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
        metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
        # display(Markdown("#### Scaled dataset - Verify that the scaling does not affect the group label statistics"))
        print("-->Verify that the scaling does not affect the group label statistics")
        print(
            "Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_train.mean_difference())
        metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups)
        print(
            "Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_test.mean_difference())

        # %% md

        ### Standard Logistic Regression
        ### Standard MLP Classifier

        X_train = dataset_orig_train.features
        y_train = dataset_orig_train.labels.ravel()

        # model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
        # fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
        # lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)
        #
        lmod = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,
                             hidden_layer_sizes=(64, 32, 16, 8, 4),
                             random_state=1, verbose=True)  # identity， relu
        lmod.fit(X_train, y_train)
        #
        # dimension = len(X_train[0])
        # lmod = initial_dnn(dimension)
        # lmod.fit(x=X_train,y=y_train, batch_size=128, epochs=100)

        # lmod = LogisticRegression(solver='lbfgs')
        # lmod.fit(X_train, y_train, sample_weight=dataset_orig_train.instance_weights)

        X_test = dataset_orig_test.features
        y_test = dataset_orig_test.labels.ravel()

        y_pred = lmod.predict(X_test)

        # display(Markdown("#### Accuracy"))
        print("-->accuracy:")
        lr_acc = accuracy_score(y_test, y_pred)
        print(lr_acc)

        # %%

        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_test_pred.labels = y_pred

        # thresh_arr = np.linspace(0.01, 0.5, 50)
        thresh_arr = np.array([0.5])
        orig_metrics = metric_test1(dataset=dataset_orig_test,
                                    model=lmod,
                                    thresh_arr=thresh_arr,
                                    privileged_groups=privileged_groups,
                                    unprivileged_groups=unprivileged_groups)
        lr_orig_best_ind = np.argmax(orig_metrics['bal_acc'])

        disp_imp = np.array(orig_metrics['disp_imp'])
        disp_imp_err = 1 - np.minimum(disp_imp, 1 / disp_imp)

        print("-->Validating MLP model on original data")
        describe_metrics(orig_metrics, thresh_arr)

        # positive class index
        pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
        # dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
        dataset_orig_test_pred.scores = lmod.predict(X_test)[:, pos_ind].reshape(-1, 1)

        metric_test = ClassificationMetric(dataset_orig_test,
                                           dataset_orig_test_pred,
                                           unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)
        print("-->average_odds_difference:")
        lr_aod = metric_test.average_odds_difference()
        print(lr_aod)

        # %% md

        ### Exponentiated Gradient Reduction

        # %% md

        # Choose a base model for the randomized classifer

        # %%

        estimator = LogisticRegression(solver='lbfgs')

        # estimator = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 8, 4),
        #                     random_state=1, verbose=True) #identity， relu

        # dimension = len(X_train[0])
        # estimator = initial_dnn(dimension)

        # %% md

        # Train the randomized classifier and observe test accuracy. Other options for `constraints` include "DemographicParity,"
        # "TruePositiveRateDifference", and "ErrorRateRatio."

        # %%

        np.random.seed(0)  # need for reproducibility
        exp_grad_red = ExponentiatedGradientReduction(estimator=estimator,
                                                      constraints="EqualizedOdds",
                                                      drop_prot_attr=False)
        exp_grad_red.fit(dataset_orig_train)
        exp_grad_red_pred = exp_grad_red.predict(dataset_orig_test)

        metric_test = ClassificationMetric(dataset_orig_test,
                                           exp_grad_red_pred,
                                           unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)

        display(Markdown("#### Accuracy"))
        egr_acc = metric_test.accuracy()
        print(egr_acc)

        # Check if accuracy is comparable
        assert abs(lr_acc - egr_acc) < 0.03

        display(Markdown("#### Average odds difference"))
        egr_aod = metric_test.average_odds_difference()
        print(egr_aod)

        # Check if average odds difference has improved
        # assert abs(egr_aod)<abs(lr_aod)

        # accuracy of model after gradient reduction
        X_test = dataset_orig_test.features
        y_test = dataset_orig_test.labels.ravel()
        y_pred = exp_grad_red.predict(dataset_orig_test).labels
        lr_acc = accuracy_score(y_test, y_pred)
        print("-->accuracy after gradient reduction:", lr_acc)

        # thresh_arr = np.linspace(0.01, 0.5, 50)
        thresh_arr = np.array([0.5])
        gr_metrics = metric_test_without_thresh(dataset=dataset_orig_test,
                                                dataset_pred=exp_grad_red_pred,
                                                privileged_groups=privileged_groups,
                                                unprivileged_groups=unprivileged_groups)

        # disp_imp = np.array(val_metrics['disp_imp'])
        # disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
        # print("-->disp_imp_err", disp_imp_err)

        print("-->Validating ExponentiatedGradientReduction model on original data")
        describe_metrics(gr_metrics, thresh_arr)

        Plot_class = Plot(dataset_name=dataset_name, sens_attr=protected_attribute, processing_name=processing_name)
        # Plot.plot_acc_metric(lr_orig_metrics, lr_transf_metrics, "stat_par_diff")
        multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
        Plot_class.plot_acc_multi_metric(orig_metrics, gr_metrics, multi_metric_names)






