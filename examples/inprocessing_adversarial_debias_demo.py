# Load all necessary packages
import sys
import os
sys.path.append("../")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
from aif360.datasets.compas_dataset1 import CompasDataset_1
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.neural_network import MLPClassifier

from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import numpy as np
from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, initial_dnn2, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
names = locals()
from collections import defaultdict
from plot_result import Plot


import tensorflow as tf
# import tensorflow.compat.v1 as tf

# import tensorflow._api.v2.compat.v1 as tf
tf.disable_eager_execution()

import random
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_random_seed(seed)
MAX_NUM = 1000

# processing_name = str(os.path.basename(__file__)).split("_demo")[0]
processing_name = "AD"

def AD_metric_test(dataset_name):
    print("-->AD_metric_test")
    if dataset_name == "Adult income":
        # function = "load_preproc_data_adult"
        function = "AdultDataset"
        protected_attributes = 'age'
        privileged_groups = [{'age': 1.0}]
        unprivileged_groups = [{'age': 0.0}]
        # protected_attributes = ['race', 'sex']
        # privileged_groups = [{'race': 1.0}, {'sex': 1.0}]
        # unprivileged_groups = [{'race': 0.0}, {'sex': 0.0}]
    elif dataset_name == "German credit":
        # function = "load_preproc_data_german"
        function = "GermanDataset"
        protected_attributes = ['sex', 'age']
        privileged_groups = [{'sex': 1.0}, {'age': 1.0}]
        unprivileged_groups = [{'sex': 0.0}, {'age': 0.0}]
    elif dataset_name == "Bank":
        function = "BankDataset"
        protected_attributes = 'age'
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
        EPOCHS = 1000
    elif dataset_name == "German credit":
        name = "Credit"
        BATCH_SIZE = 16
        EPOCHS = 500
    else:
        name = "Compas"
        BATCH_SIZE = 128
        EPOCHS = 100

    dataset_orig = eval(function)()
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed=seed)

    # min_max_scaler = MaxAbsScaler()
    # dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    # dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

    dimension = len(dataset_orig_train.features[0])
    plain_model = initial_dnn2(dimension)
    plain_model.fit(x=dataset_orig_train.features, y=dataset_orig_train.labels.ravel(),
                   sample_weight=dataset_orig_train.instance_weights,
                   batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=1)

    MODEL_DIR = name + "/AD_model.h5"
    plain_model.save(MODEL_DIR)

    # plain_model = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,
    #                             hidden_layer_sizes=(64, 32, 16, 8, 4),
    #                             random_state=1, verbose=True)
    # plain_model.fit(dataset_orig_train.features, dataset_orig_train.labels.ravel())


    y_test = dataset_orig_test.labels.ravel()
    y_pred = plain_model.predict_classes(dataset_orig_test.features)
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))

    print("-->dataset_orig_test.features", dataset_orig_test.features)

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
    # input_privileged = []
    # input_unprivileged = []
    # for i in range(0, len(privileged_groups)):
    #     privileged_group = [privileged_groups[i]]
    #     unprivileged_group = [unprivileged_groups[i]]
    #     # for group in privileged_groups:
    #     #     group = [group]
    #     new_inputs_priviledge = dataset_orig_train.generate_inputs(max_num=MAX_NUM, if_random=False,
    #                                                                        privileged_groups=privileged_group,
    #                                                                        if_priviledge=True)
    #     input_privileged += new_inputs_priviledge
    #     new_inputs_unpriviledge = dataset_orig_train.generate_inputs(max_num=MAX_NUM, if_random=False,
    #                                                                          privileged_groups=unprivileged_group,
    #                                                                          if_priviledge=True)
    #     input_unprivileged += new_inputs_unpriviledge
    #
    # new_inputs = input_privileged + input_unprivileged
    # random.shuffle(new_inputs)
    # classified_dataset = new_inputs_to_dataset(new_inputs, dataset_orig_train)
    # print("-->classified_dataset", classified_dataset)
    # classified_dataset.features = min_max_scaler.fit_transform(classified_dataset.features)

    #### prepare when doing Scaler preprocessing
    # dataset_nodebiasing_test = dataset_orig_test.copy()
    # dataset_nodebiasing_test.labels = plain_model.predict(dataset_orig_test.features)

    thresh_arr = np.array([0.5])
    # print("----------" + "test on test data" + "----------")
    # lr_orig_metrics = metric_test1(dataset=dataset_orig_panel19_test,
    #                                    model=lr_orig_panel19,
    #                                    thresh_arr=thresh_arr,
    #                                    unprivileged_groups=unprivileged_groups,
    #                                    privileged_groups=privileged_groups
    #                                    )
    # describe_metrics(lr_orig_metrics, thresh_arr)

    print("----------" + "test on test data" + "----------")
    multi_orig_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                model=plain_model,
                                                thresh_arr=thresh_arr,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups
                                                )
    describe_metrics_new_inputs(multi_orig_metrics, thresh_arr)
    multi_orig_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]

    print("----------" + "multivariate group metric test" + "----------")
    multi_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,   #classified_dataset
                                                   model=plain_model,
                                                   thresh_arr=thresh_arr,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups
                                                   )
    print(multi_group_metrics)

    print("----------" + "causal metric test" + "----------")
    multi_causal_metrics = metric_test_causal(dataset=dataset_orig_test,
                                              model=plain_model,
                                              thresh_arr=thresh_arr,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups
                                              )
    print(multi_causal_metrics)

    sess = tf.Session()
    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                          unprivileged_groups=unprivileged_groups,
                                          scope_name='debiased_classifier',
                                          debias=True,
                                          sess=sess)
    debiased_model.fit(dataset_orig_train)

    # Apply the debiased model to test data
    dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

    # y_test = dataset_orig_test.labels.ravel()
    debias_y_pred = debiased_model.predict(dataset_orig_test).labels
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(debias_y_pred)))

    print("----------" + "test on test data" + "----------")
    multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                model=debiased_model,
                                                thresh_arr=thresh_arr,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups
                                                )
    describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)
    multi_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(debias_y_pred))]

    print("----------" + "multivariate group metric test" + "----------")
    multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                   model=debiased_model,
                                                   thresh_arr=thresh_arr,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups
                                                   )
    print(multi_group_trans_metrics)

    print("----------" + "causal metric test" + "----------")
    multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
                                              model=debiased_model,
                                              thresh_arr=thresh_arr,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups
                                              )
    print(multi_causal_trans_metrics)

    if len(privileged_groups) == 1:
        multi_orig_metrics = [multi_orig_metrics, multi_group_metrics, multi_causal_metrics]
        print("-->multi_orig_metrics", multi_orig_metrics)
        all_multi_orig_metrics = defaultdict(list)
        for to_merge in multi_orig_metrics:
            for key, value in to_merge.items():
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
        print([dict(all_multi_orig_metrics), dict(all_multi_trans_metrics)])

        return all_multi_orig_metrics, all_multi_trans_metrics, dataset_name, processing_name


    # univariate test
    all_uni_orig_metrics = []
    all_uni_trans_metrics = []
    sens_inds = [0, 1]
    for sens_ind in sens_inds:
        sens_attr = dataset_orig_train.protected_attribute_names[sens_ind]
        print("-->sens_attr", sens_attr)
        privileged_groups = [{sens_attr: v} for v in
                             dataset_orig_train.privileged_protected_attributes[sens_ind]]
        unprivileged_groups = [{sens_attr: v} for v in
                               dataset_orig_train.unprivileged_protected_attributes[sens_ind]]
        print("-->unprivileged_groups", unprivileged_groups)
        print("-->privileged_groups", privileged_groups)

        thresh_arr = np.array([0.5])

        dimension = len(dataset_orig_train.features[0])
        plain_model = initial_dnn2(dimension)
        plain_model.fit(x=dataset_orig_train.features, y=dataset_orig_train.labels.ravel(),
                        sample_weight=dataset_orig_train.instance_weights,
                        batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=1)

        y_test = dataset_orig_test.labels.ravel()
        y_pred = plain_model.predict_classes(dataset_orig_test.features)
        print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))

        names["orig_" + str(sens_attr) + '_metrics'] = []
        print("----------" + "test on test data" + "----------")
        uni_orig_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                    model=plain_model,
                                                    thresh_arr=thresh_arr,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups
                                                    )
        describe_metrics_new_inputs(uni_orig_metrics, thresh_arr)
        uni_orig_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_orig_metrics)

        print("----------" + "multivariate group metric test" + "----------")
        uni_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                       model=plain_model,
                                                       thresh_arr=thresh_arr,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups
                                                       )
        print(uni_group_metrics)
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_group_metrics)

        print("----------" + "causal metric test" + "----------")
        uni_causal_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                  model=plain_model,
                                                  thresh_arr=thresh_arr,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups
                                                  )
        uni_causal_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
        print(uni_causal_metrics)
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_causal_metrics)

        sess = tf.Session()
        sess.close()
        tf.reset_default_graph()
        sess = tf.Session()
        debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                              unprivileged_groups=unprivileged_groups,
                                              scope_name='debiased_classifier',
                                              debias=True,
                                              sess=sess)
        debiased_model.fit(dataset_orig_train)

        debias_y_pred = debiased_model.predict(dataset_orig_test).labels
        accuracy = accuracy_score(list(y_test), list(debias_y_pred))
        print("-->prediction accuracy on test data", accuracy)

        names["trans_" + str(sens_attr) + '_metrics'] = []
        print("----------" + "test on test data" + "----------")
        uni_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                          model=debiased_model,
                                                          thresh_arr=thresh_arr,
                                                          unprivileged_groups=unprivileged_groups,
                                                          privileged_groups=privileged_groups
                                                          )
        describe_metrics_new_inputs(uni_orig_trans_metrics, thresh_arr)
        uni_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(debias_y_pred))]
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_orig_trans_metrics)

        print("----------" + "multivariate group metric test" + "----------")
        uni_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                             model=debiased_model,
                                                             thresh_arr=thresh_arr,
                                                             unprivileged_groups=unprivileged_groups,
                                                             privileged_groups=privileged_groups
                                                             )
        print(uni_group_trans_metrics)
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_group_trans_metrics)

        print("----------" + "causal metric test" + "----------")
        uni_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                        model=debiased_model,
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

    print("-->uni_race_orig_metrics", all_uni_orig_metrics[0])
    print("-->uni_race_trans_metrics", all_uni_trans_metrics[0])
    print("-->uni_sex_orig_metrics", all_uni_orig_metrics[1])
    print("-->uni_sex_trans_metrics", all_uni_trans_metrics[1])
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

    dataset_orig = eval(function)()
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed=seed)

    for protected_attribute in protected_attributes:
        print("-->sens_attr", protected_attribute)
        privileged_groups = all_privileged_groups[protected_attribute]
        print("-->privileged_groups", privileged_groups)
        unprivileged_groups = all_unprivileged_groups[protected_attribute]
        print("-->privileged_groups", unprivileged_groups)

        # print out some labels, names, etc.
        display(Markdown("#### Training Dataset shape"))
        print(dataset_orig_train.features.shape)
        display(Markdown("#### Favorable and unfavorable labels"))
        print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
        display(Markdown("#### Protected attribute names"))
        print(dataset_orig_train.protected_attribute_names)
        display(Markdown("#### Privileged and unprivileged protected attribute values"))
        print(dataset_orig_train.privileged_protected_attributes,
              dataset_orig_train.unprivileged_protected_attributes)
        display(Markdown("#### Dataset feature names"))
        print(dataset_orig_train.feature_names)

        #### Metric for original training data

        # Metric for the original dataset
        print("-->Metrics for the original dataset:")
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original training dataset"))
        print("Train set: statistical parity diff = %f" % metric_orig_train.mean_difference())
        metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups)
        print("Test set: statistical parity diff = %f" % metric_orig_test.mean_difference())

        # %%
        # 数据归一化
        min_max_scaler = MaxAbsScaler()
        dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
        dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

        metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
        print("-->Metrics of the original dataset after min_max_scaler:")
        display(Markdown("#### Scaled dataset - Verify that the scaling does not affect the group label statistics"))
        # Difference in mean outcomes between unprivileged and privileged groups
        print("Train set: statistical parity diff = %f" % metric_scaled_train.mean_difference())
        metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups)
        print("Test set: statistical parity diff = %f" % metric_scaled_test.mean_difference())

        # %% md

        ### Learn plan classifier without debiasing

        # %%

        # Load post-processing algorithm that equalizes the odds
        # Learn parameters with debias set to False
        sess = tf.Session()
        # plain_model = AdversarialDebiasing(privileged_groups=privileged_groups,
        #                                    unprivileged_groups=unprivileged_groups,
        #                                    scope_name='plain_classifier',
        #                                    debias=False,
        #                                    sess=sess)

        # plain_model = MLPClassifier(solver='sgd', activation='identity', max_iter=500, alpha=1e-5,
        #                             hidden_layer_sizes=(64, 32, 16, 8, 4),
        #                             random_state=1, verbose=True)
        plain_model = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,
                             hidden_layer_sizes=(64, 32, 16, 8, 4),
                             random_state=1, verbose=True)

        plain_model.fit(dataset_orig_train.features, dataset_orig_train.labels.ravel())

        dataset_nodebiasing_test = dataset_orig_test.copy()
        dataset_nodebiasing_test.labels = plain_model.predict_classes(dataset_orig_test.features)

        dataset_nodebiasing_train = dataset_orig_train.copy()
        dataset_nodebiasing_train.labels = plain_model.predict_classes(dataset_orig_train.features)

        # Apply the plain model to test data
        # dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
        # dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)

        # %%

        # Metrics for the dataset from plain model (without debiasing)
        print("-->Metrics for plain model (without debiasing)")
        display(Markdown("#### Plain model - without debiasing - dataset metrics"))
        metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_nodebiasing_train,
                                                                    unprivileged_groups=unprivileged_groups,
                                                                    privileged_groups=privileged_groups)

        print("Train set: statistical parity diff = %f" % metric_dataset_nodebiasing_train.mean_difference())

        metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_nodebiasing_test,
                                                                   unprivileged_groups=unprivileged_groups,
                                                                   privileged_groups=privileged_groups)

        print("Test set: statistical parity diff = %f" % metric_dataset_nodebiasing_test.mean_difference())

        display(Markdown("#### Plain model - without debiasing - classification metrics"))
        # classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test,
        #                                                           dataset_nodebiasing_test,
        #                                                           unprivileged_groups=unprivileged_groups,
        #                                                           privileged_groups=privileged_groups)
        # print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
        # TPR = classified_metric_nodebiasing_test.true_positive_rate()
        # TNR = classified_metric_nodebiasing_test.true_negative_rate()
        # bal_acc_nodebiasing_test = 0.5 * (TPR + TNR)
        # print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
        # print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
        # print(
        #     "Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
        # print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
        # print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())

        thresh_arr = np.array([0.5])
        orig_metrics = get_metrics(dataset_orig_test, dataset_nodebiasing_test, privileged_groups=privileged_groups,
                                   unprivileged_groups=unprivileged_groups)
        describe_metrics(orig_metrics, thresh_arr)

        # %% md

        ### Apply in-processing algorithm based on adversarial learning

        # %%

        sess.close()
        tf.reset_default_graph()
        sess = tf.Session()

        # %%

        # Learn parameters with debias set to True
        debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                              unprivileged_groups=unprivileged_groups,
                                              scope_name='debiased_classifier',
                                              debias=True,
                                              sess=sess)
        debiased_model.fit(dataset_orig_train)

        # Apply the plain model to test data
        dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
        dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

        # %%

        # # Metrics for the dataset from plain model (without debiasing)
        # display(Markdown("#### Plain model - without debiasing - dataset metrics"))
        # print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_train.mean_difference())
        # print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_nodebiasing_test.mean_difference())

        # Metrics for the dataset from model with debiasing
        print("-->Metrics for the model with debiasing:")
        display(Markdown("#### Model - with debiasing - dataset metrics"))
        metric_dataset_debiasing_train = BinaryLabelDatasetMetric(dataset_debiasing_train,
                                                                  unprivileged_groups=unprivileged_groups,
                                                                  privileged_groups=privileged_groups)

        print("Train set: statistical parity diff = %f" % metric_dataset_debiasing_train.mean_difference())

        metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test,
                                                                 unprivileged_groups=unprivileged_groups,
                                                                 privileged_groups=privileged_groups)

        print("Test set: statistical parity diff = %f" % metric_dataset_debiasing_test.mean_difference())

        # display(Markdown("#### Plain model - without debiasing - classification metrics"))
        # print("Test set: Classification accuracy = %f" % classified_metric_nodebiasing_test.accuracy())
        # TPR = classified_metric_nodebiasing_test.true_positive_rate()
        # TNR = classified_metric_nodebiasing_test.true_negative_rate()
        # bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
        # print("Test set: Balanced classification accuracy = %f" % bal_acc_nodebiasing_test)
        # print("Test set: Disparate impact = %f" % classified_metric_nodebiasing_test.disparate_impact())
        # print("Test set: Equal opportunity difference = %f" % classified_metric_nodebiasing_test.equal_opportunity_difference())
        # print("Test set: Average odds difference = %f" % classified_metric_nodebiasing_test.average_odds_difference())
        # print("Test set: Theil_index = %f" % classified_metric_nodebiasing_test.theil_index())
        #

        display(Markdown("#### Model - with debiasing - classification metrics"))
        # classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test,
        #                                                         dataset_debiasing_test,
        #                                                         unprivileged_groups=unprivileged_groups,
        #                                                         privileged_groups=privileged_groups)
        # print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
        # TPR = classified_metric_debiasing_test.true_positive_rate()
        # TNR = classified_metric_debiasing_test.true_negative_rate()
        # bal_acc_debiasing_test = 0.5 * (TPR + TNR)
        # print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
        # print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
        # print(
        #     "Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
        # print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
        # print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())

        thresh_arr = np.array([0.5])
        debias_metrics = get_metrics(dataset_orig_test, dataset_debiasing_test, privileged_groups=privileged_groups,
                                     unprivileged_groups=unprivileged_groups)
        describe_metrics(debias_metrics, thresh_arr)

        Plot_class = Plot(dataset_name=dataset_name, sens_attr=protected_attribute, processing_name=processing_name)
        # Plot.plot_acc_metric(lr_orig_metrics, lr_transf_metrics, "stat_par_diff")
        multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
        Plot_class.plot_acc_multi_metric(orig_metrics, debias_metrics, multi_metric_names)

dataset_name = "German credit"
AD_metric_test(dataset_name)
# metric_test(dataset_name)