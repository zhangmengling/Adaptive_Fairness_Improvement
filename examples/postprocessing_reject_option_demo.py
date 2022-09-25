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
        EPOCHS = 100

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

    scale_orig = MaxAbsScaler()
    dataset_orig_train.features = scale_orig.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features = scale_orig.fit_transform(dataset_orig_test.features)

    # Logistic regression classifier and predi1ctions
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
    y_pred = orig_model.predict_classes(dataset_orig_test.features)
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))
    
    def new_inputs_to_dataset(new_inputs, original_dataset):
        classified_dataset = original_dataset.copy()
        classified_dataset.features = np.array(new_inputs)
        length = len(new_inputs)
        classified_dataset.instance_names = [1] * length
        classified_dataset.scores = np.array([0] * length)
        classified_dataset.labels = np.array([0] * length)
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
        input_privileged += new_inputs_priviledge
        new_inputs_unpriviledge = dataset_orig_train.generate_inputs(max_num=MAX_NUM, if_random=False,
                                                                     privileged_groups=unprivileged_group,
                                                                     if_priviledge=True)
        input_unprivileged += new_inputs_unpriviledge

    new_inputs = input_privileged + input_unprivileged
    random.shuffle(new_inputs)
    # classified_dataset = new_inputs_to_dataset(new_inputs, dataset_orig_train)
    # print("-->classified_dataset", classified_dataset)

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

    processing_name = str(os.path.basename(__file__)).split("_demo")[0]

    dataset_orig = eval(function)()

    # Metric used (should be one of allowed_metrics)
    metric_name = "Equal opportunity difference"

    # Upper and lower bound on the fairness metric used
    metric_ub = 0.05
    metric_lb = -0.05

    randseed = 12345679
    seed = 1
    np.random.seed(seed)

    # Verify metric name
    allowed_metrics = ["Statistical parity difference",
                       "Average odds difference",
                       "Equal opportunity difference"]
    if metric_name not in allowed_metrics:
        raise ValueError("Metric name should be one of allowed metrics")

    # %% md

    #### Split into train, test and validation

    # %%

    # Get the dataset and split into train and test
    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True, seed=seed)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True, seed=seed)

    #### Clean up training data and display properties of the data

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

    for protected_attribute in protected_attributes:
        print("-->sens_attr", protected_attribute)
        privileged_groups = all_privileged_groups[protected_attribute]
        print("-->privileged_groups", privileged_groups)
        unprivileged_groups = all_unprivileged_groups[protected_attribute]
        print("-->privileged_groups", unprivileged_groups)

        #### Metric for original training data
        metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
        display(Markdown("#### Original training dataset"))
        print(
            "Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

        # %% md

        ### Train classifier on original data

        # %%

        # Logistic regression classifier and predi1ctions
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(dataset_orig_train.features)
        y_train = dataset_orig_train.labels.ravel()

        # lmod = LogisticRegression()
        # lmod.fit(X_train, y_train)
        # y_train_pred = lmod.predict(X_train)

        lmod = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,
                             hidden_layer_sizes=(64, 32, 16, 8, 4),
                             random_state=1, verbose=True)  # identity， relu
        lmod.fit(X_train, y_train)
        y_train_pred = lmod.predict(X_train)

        # dimension = len(X_train[0])
        # lmod = initial_dnn(dimension)
        # lmod.fit(x=X_train,y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
        # y_train_pred = lmod.predict(X_train)

        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

        dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)

        fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
        y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]

        # Prediction probs for validation and testing data
        X_valid = scale_orig.transform(dataset_orig_valid.features)
        y_valid_pred_prob = lmod.predict_proba(X_valid)[:, fav_idx]

        X_test = scale_orig.transform(dataset_orig_test.features)
        y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]

        class_thresh = 0.5
        dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
        dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)
        dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

        y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
        y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
        y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
        dataset_orig_train_pred.labels = y_train_pred

        y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
        y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
        y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
        dataset_orig_valid_pred.labels = y_valid_pred

        y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
        y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
        y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
        dataset_orig_test_pred.labels = y_test_pred

        # thresh_arr = np.linspace(0.01, 0.5, 50)
        thresh_arr = [0.5]
        train_metrics = metric_test_without_thresh(dataset=dataset_orig_train,
                                                   dataset_pred=dataset_orig_train_pred,
                                                   privileged_groups=privileged_groups,
                                                   unprivileged_groups=unprivileged_groups)
        print("-->validating MLP model on original training data")
        describe_metrics(train_metrics, thresh_arr)

        val_metrics = metric_test_without_thresh(dataset=dataset_orig_valid,
                                                 dataset_pred=dataset_orig_valid_pred,
                                                 privileged_groups=privileged_groups,
                                                 unprivileged_groups=unprivileged_groups)
        print("-->validating MLP model on original validating data")
        describe_metrics(val_metrics, thresh_arr)

        orig_test_metrics = metric_test_without_thresh(dataset=dataset_orig_test,
                                                       dataset_pred=dataset_orig_test_pred,
                                                       privileged_groups=privileged_groups,
                                                       unprivileged_groups=unprivileged_groups)
        print("-->validating MLP model on original testing data")
        describe_metrics(orig_test_metrics, thresh_arr)

        # thresh_arr = np.linspace(0.01, 0.5, 50)
        # val_metrics = metric_test1(dataset=dataset_orig_valid,
        #                    model=lmod,
        #                    thresh_arr=thresh_arr)
        # print("-->validating LR model on original validating data")
        # describe_metrics(val_metrics, thresh_arr)
        # test_metrics = metric_test1(dataset=dataset_orig_test,
        #                    model=lmod,
        #                    thresh_arr=thresh_arr)
        # print("-->validating LR model on original validating data")
        # describe_metrics(test_metrics, thresh_arr)
        #

        # positive class index
        pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
        dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
        dataset_orig_train_pred.labels = y_train_pred

        # %% md

        #### Obtain scores for validation and test sets

        # %%

        scale_orig.fit_transform(dataset_orig_train.features)

        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
        y_valid = dataset_orig_valid_pred.labels
        dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:, pos_ind].reshape(-1, 1)

        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        X_test = scale_orig.transform(dataset_orig_test_pred.features)
        y_test = dataset_orig_test_pred.labels
        dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

        ### Find the optimal parameters from the validation set

        #### Best threshold for classification only (no fairness)

        # num_thresh = 100
        # ba_arr = np.zeros(num_thresh)
        # class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
        # # class_thresh_arr = [0.5]
        # for idx, class_thresh in enumerate(class_thresh_arr):
        #     fav_inds = dataset_orig_valid_pred.scores > class_thresh
        #     dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        #     dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
        #
        #     classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
        #                                                         dataset_orig_valid_pred,
        #                                                         unprivileged_groups=unprivileged_groups,
        #                                                         privileged_groups=privileged_groups)
        #
        #     ba_arr[idx] = 0.5 * (classified_metric_orig_valid.true_positive_rate() \
        #                          + classified_metric_orig_valid.true_negative_rate())
        #
        # best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
        # best_class_thresh = class_thresh_arr[best_ind]

        # print("Best balanced accuracy (no fairness constraints) = %.4f" % np.max(ba_arr))
        # print("Optimal classification threshold (no fairness constraints) = %.4f" % best_class_thresh)

        #### Estimate optimal parameters for the ROC method

        ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                         privileged_groups=privileged_groups,
                                         low_class_thresh=0.01, high_class_thresh=0.99,
                                         num_class_thresh=100, num_ROC_margin=50,
                                         metric_name=metric_name,
                                         metric_ub=metric_ub, metric_lb=metric_lb)
        ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)

        # %%

        # print("Optimal classification threshold (with fairness constraints) = %.4f" % ROC.classification_threshold)
        # print("Optimal ROC margin = %.4f" % ROC.ROC_margin)

        ### Predictions from Validation Set

        # Metrics for the test set

        # best_class_thresh = 0.5
        # fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
        # dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        # dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

        # display(Markdown("#### Validation set"))
        # display(Markdown("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy"))

        # metric_valid_bef = compute_metrics(dataset_orig_valid, dataset_orig_valid_pred,
        #                                    unprivileged_groups, privileged_groups)

        # Transform the validation set
        dataset_transf_valid_pred = ROC.predict(dataset_orig_valid_pred)
        # display(Markdown("#### Validation set"))
        # display(Markdown("##### Transformed predictions - With fairness constraints"))
        # metric_valid_aft = compute_metrics(dataset_orig_valid, dataset_transf_valid_pred,
        #                                    unprivileged_groups, privileged_groups)

        # %%

        # Testing: Check if the metric optimized has not become worse
        # assert np.abs(metric_valid_aft[metric_name]) <= np.abs(metric_valid_bef[metric_name])

        ### Predictions from Test Set

        # Metrics for the test set
        # fav_inds = dataset_orig_test_pred.scores > best_class_thresh
        # dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        # dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

        # display(Markdown("#### Test set"))
        # display(Markdown("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy"))

        # metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
        #                                   unprivileged_groups, privileged_groups)

        # Metrics for the transformed test set
        dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)

        # display(Markdown("#### Test set"))
        # display(Markdown("##### Transformed predictions - With fairness constraints"))
        # metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
        #                                   unprivileged_groups, privileged_groups)

        print("-->after transform validation and test data using post-processing")
        # thresh_arr = np.linspace(0.01, 0.5, 50)
        thresh_arr = [0.5]
        val_metrics = metric_test_without_thresh(dataset=dataset_orig_valid,
                                                 dataset_pred=dataset_transf_valid_pred,
                                                 privileged_groups=privileged_groups,
                                                 unprivileged_groups=unprivileged_groups)
        print("-->validating on transformed validating data ")
        describe_metrics(val_metrics, thresh_arr)

        ro_metrics = metric_test_without_thresh(dataset=dataset_orig_test,
                                                dataset_pred=dataset_transf_test_pred,
                                                privileged_groups=privileged_groups,
                                                unprivileged_groups=unprivileged_groups)
        print("-->validating on transformed testing data ")
        describe_metrics(ro_metrics, thresh_arr)

        Plot_class = Plot(dataset_name=dataset_name, sens_attr=protected_attribute, processing_name=processing_name)
        # Plot.plot_acc_metric(lr_orig_metrics, lr_transf_metrics, "stat_par_diff")
        multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
        Plot_class.plot_acc_multi_metric(orig_test_metrics, ro_metrics, multi_metric_names)


dataset_name = "Compas"
RO_metric_test(dataset_name)


"""
# Thresholds
all_thresh = np.linspace(0.01, 0.99, 25)
display(Markdown("#### Classification thresholds used for validation and parameter selection"))

bef_avg_odds_diff_test = []
bef_avg_odds_diff_valid = []
aft_avg_odds_diff_test = []
aft_avg_odds_diff_valid = []
bef_bal_acc_valid = []
bef_bal_acc_test = []
aft_bal_acc_valid = []
aft_bal_acc_test = []
for thresh in tqdm(all_thresh):
    dataset_orig_valid_pred_thresh = dataset_orig_valid_pred.copy(deepcopy=True)
    dataset_orig_test_pred_thresh = dataset_orig_test_pred.copy(deepcopy=True)
    dataset_transf_valid_pred_thresh = dataset_transf_valid_pred.copy(deepcopy=True)
    dataset_transf_test_pred_thresh = dataset_transf_test_pred.copy(deepcopy=True)

    # Labels for the datasets from scores
    y_temp = np.zeros_like(dataset_orig_valid_pred_thresh.labels)
    y_temp[dataset_orig_valid_pred_thresh.scores >= thresh] = dataset_orig_valid_pred_thresh.favorable_label
    y_temp[~(dataset_orig_valid_pred_thresh.scores >= thresh)] = dataset_orig_valid_pred_thresh.unfavorable_label
    dataset_orig_valid_pred_thresh.labels = y_temp

    y_temp = np.zeros_like(dataset_orig_test_pred_thresh.labels)
    y_temp[dataset_orig_test_pred_thresh.scores >= thresh] = dataset_orig_test_pred_thresh.favorable_label
    y_temp[~(dataset_orig_test_pred_thresh.scores >= thresh)] = dataset_orig_test_pred_thresh.unfavorable_label
    dataset_orig_test_pred_thresh.labels = y_temp

    y_temp = np.zeros_like(dataset_transf_valid_pred_thresh.labels)
    y_temp[dataset_transf_valid_pred_thresh.scores >= thresh] = dataset_transf_valid_pred_thresh.favorable_label
    y_temp[~(dataset_transf_valid_pred_thresh.scores >= thresh)] = dataset_transf_valid_pred_thresh.unfavorable_label
    dataset_transf_valid_pred_thresh.labels = y_temp

    y_temp = np.zeros_like(dataset_transf_test_pred_thresh.labels)
    y_temp[dataset_transf_test_pred_thresh.scores >= thresh] = dataset_transf_test_pred_thresh.favorable_label
    y_temp[~(dataset_transf_test_pred_thresh.scores >= thresh)] = dataset_transf_test_pred_thresh.unfavorable_label
    dataset_transf_test_pred_thresh.labels = y_temp

    # Metrics for original validation data
    classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                        dataset_orig_valid_pred_thresh,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups)
    bef_avg_odds_diff_valid.append(classified_metric_orig_valid.equal_opportunity_difference())

    bef_bal_acc_valid.append(0.5 * (classified_metric_orig_valid.true_positive_rate() +
                                    classified_metric_orig_valid.true_negative_rate()))

    classified_metric_orig_test = ClassificationMetric(dataset_orig_test,
                                                       dataset_orig_test_pred_thresh,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
    bef_avg_odds_diff_test.append(classified_metric_orig_test.equal_opportunity_difference())
    bef_bal_acc_test.append(0.5 * (classified_metric_orig_test.true_positive_rate() +
                                   classified_metric_orig_test.true_negative_rate()))

    # Metrics for transf validing data
    classified_metric_transf_valid = ClassificationMetric(
        dataset_orig_valid,
        dataset_transf_valid_pred_thresh,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    aft_avg_odds_diff_valid.append(classified_metric_transf_valid.equal_opportunity_difference())
    aft_bal_acc_valid.append(0.5 * (classified_metric_transf_valid.true_positive_rate() +
                                    classified_metric_transf_valid.true_negative_rate()))

    # Metrics for transf validation data
    classified_metric_transf_test = ClassificationMetric(dataset_orig_test,
                                                         dataset_transf_test_pred_thresh,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)
    aft_avg_odds_diff_test.append(classified_metric_transf_test.equal_opportunity_difference())
    aft_bal_acc_test.append(0.5 * (classified_metric_transf_test.true_positive_rate() +
                                   classified_metric_transf_test.true_negative_rate()))

# %%

bef_bal_acc_valid = np.array(bef_bal_acc_valid)
bef_avg_odds_diff_valid = np.array(bef_avg_odds_diff_valid)

aft_bal_acc_valid = np.array(aft_bal_acc_valid)
aft_avg_odds_diff_valid = np.array(aft_avg_odds_diff_valid)

fig, ax1 = plt.subplots(figsize=(13, 7))
ax1.plot(all_thresh, bef_bal_acc_valid, color='b')
ax1.plot(all_thresh, aft_bal_acc_valid, color='b', linestyle='dashed')
ax1.set_title('Original and Postprocessed validation data', fontsize=16, fontweight='bold')
ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)

ax2 = ax1.twinx()
ax2.plot(all_thresh, np.abs(bef_avg_odds_diff_valid), color='r')
ax2.plot(all_thresh, np.abs(aft_avg_odds_diff_valid), color='r', linestyle='dashed')
ax2.set_ylabel('abs(Equal opportunity diff)', color='r', fontsize=16, fontweight='bold')
ax2.yaxis.set_tick_params(labelsize=14)
ax2.grid(True)
fig.legend(["Balanced Acc. - Orig.", "Balanced Acc. - Postproc.",
            "Equal opp. diff. - Orig.", "Equal opp. diff. - Postproc.", ],
           fontsize=16)



bef_bal_acc_test = np.array(bef_bal_acc_test)
bef_avg_odds_diff_test = np.array(bef_avg_odds_diff_test)

aft_bal_acc_test = np.array(aft_bal_acc_test)
aft_avg_odds_diff_test = np.array(aft_avg_odds_diff_test)

fig, ax1 = plt.subplots(figsize=(13, 7))
ax1.plot(all_thresh, bef_bal_acc_test, color='b')
ax1.plot(all_thresh, aft_bal_acc_test, color='b', linestyle='dashed')
ax1.set_title('Original and Postprocessed testing data', fontsize=16, fontweight='bold')
ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)

ax2 = ax1.twinx()
ax2.plot(all_thresh, np.abs(bef_avg_odds_diff_test), color='r')
ax2.plot(all_thresh, np.abs(aft_avg_odds_diff_test), color='r', linestyle='dashed')
ax2.set_ylabel('abs(Equal opportunity diff)', color='r', fontsize=16, fontweight='bold')
ax2.yaxis.set_tick_params(labelsize=14)
ax2.grid(True)
fig.legend(["Balanced Acc. - Orig.", "Balanced Acc. - Postproc.",
            "Equal opp. diff. - Orig.", "Equal opp. diff. - Postproc."],
           fontsize=16)
plt.show()
"""