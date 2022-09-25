


import os
import sys
sys.path.insert(0, '../')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from sklearn.neural_network import MLPClassifier

# Datasets
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

from aif360.datasets import AdultDataset, CompasDataset, GermanDataset, BankDataset
from aif360.datasets.compas_dataset1 import CompasDataset_1

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

# Explainers
from aif360.explainers import MetricTextExplainer

# Scalers
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

# LIME
from aif360.datasets.lime_encoder import LimeEncoder
import lime
from lime.lime_tabular import LimeTabularExplainer

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization
from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, initial_dnn2, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
names = locals()
from collections import defaultdict
from plot_result import Plot

import random
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_random_seed(seed)
MAX_NUM = 1000
processing_name = "PR"

def PR_metric_test(dataset_name):
    print("-->PR_metric_test")
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
        EPOCHS = 50
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
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed=seed)

    # min_max_scaler = MaxAbsScaler()
    # dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    # dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

    dataset = dataset_orig_train

    dimension = len(dataset.features[0])
    orig_model = initial_dnn2(dimension)
    orig_model.fit(x=dataset.features, y=dataset.labels.ravel(),
                    sample_weight=dataset.instance_weights,
                    batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=1)

    MODEL_DIR = name + "/PR_model.h5"
    orig_model.save(MODEL_DIR)

    # orig_model = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,
    #                       hidden_layer_sizes=(64, 32, 16, 8, 4),
    #                       random_state=1, verbose=True)  # identity， relu
    # orig_model.fit(dataset.features, dataset.labels.ravel())

    y_test = dataset_orig_test.labels.ravel()
    y_pred = orig_model.predict_classes(dataset_orig_test.features)
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))

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
        input_privileged += new_inputs_priviledge
        new_inputs_unpriviledge = dataset_orig_train.generate_inputs(max_num=MAX_NUM, if_random=False,
                                                                             privileged_groups=unprivileged_group,
                                                                             if_priviledge=True)
        input_unprivileged += new_inputs_unpriviledge

    new_inputs = input_privileged + input_unprivileged
    random.shuffle(new_inputs)
    # classified_dataset = new_inputs_to_dataset(new_inputs, dataset_orig_train)
    # print("-->classified_dataset", classified_dataset)
    # classified_dataset.features = min_max_scaler.fit_transform(classified_dataset.features)

    thresh_arr = np.array([0.5])
    # orig_metrics = metric_test1(dataset=dataset_orig_test,
    #                             model=lr_orig_panel19,
    #                             thresh_arr=thresh_arr,
    #                             privileged_groups=privileged_groups,
    #                             unprivileged_groups=unprivileged_groups)
    # print("-->Testing MLP model on original data")
    # describe_metrics(orig_metrics, thresh_arr)

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
    multi_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,  # classified_dataset
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

    print("-->Learning Prejudice Remover (PR) model")

    eta_list = list(range(0, 26))
    metrics = []
    for eta in eta_list:
        eta = float(eta)
        debiased_model = PrejudiceRemover(sensitive_attr=protected_attributes, eta=eta)
        pr_model = debiased_model.fit(dataset_orig_train)
        metric = metric_test_new_inputs(dataset=dataset_orig_test,
                                                          model=pr_model,
                                                          thresh_arr=thresh_arr,
                                                          unprivileged_groups=unprivileged_groups,
                                                          privileged_groups=privileged_groups
                                                          )
        metrics.append(metric['stat_par_diff'])

    all_metrics = [abs(m[0]) for m in metrics]
    eta = eta_list[all_metrics.index(min(all_metrics))]

    debiased_model = PrejudiceRemover(sensitive_attr=protected_attributes, eta=eta)  # eta=25.0
    pr_model = debiased_model.fit(dataset_orig_train)

    debias_y_pred = pr_model.predict(dataset_orig_test).labels
    print("-->debias_y_pred", debias_y_pred)
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(debias_y_pred)))

    print("----------" + "test on test data" + "----------")
    multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                      model=pr_model,
                                                      thresh_arr=thresh_arr,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups
                                                      )
    describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)
    multi_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(debias_y_pred))]

    print("----------" + "multivariate group metric test" + "----------")
    multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                         model=pr_model,
                                                         thresh_arr=thresh_arr,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups
                                                         )
    print(multi_group_trans_metrics)

    print("----------" + "causal metric test" + "----------")
    multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                    model=pr_model,
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
        # sens_attr = dataset_orig_train.protected_attribute_names[sens_ind]
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

        eta_list = list(range(0, 26))
        metrics = []
        for eta in eta_list:
            eta = float(eta)
            debiased_model = PrejudiceRemover(sensitive_attr=sens_attr, eta=eta)
            model = debiased_model.fit(dataset_orig_train)
            metric = metric_test_new_inputs(dataset=dataset_orig_test,
                                            model=model,
                                            thresh_arr=thresh_arr,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups
                                            )
            metrics.append(metric['stat_par_diff'])

        all_metrics = [abs(m[0]) for m in metrics]
        eta = eta_list[all_metrics.index(min(all_metrics))]

        debiased_model = PrejudiceRemover(sensitive_attr=sens_attr, eta=eta)  # eta=25.0
        pr_model = debiased_model.fit(dataset_orig_train)

        debias_y_pred = pr_model.predict(dataset_orig_test).labels
        print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(debias_y_pred)))

        names["trans_" + str(sens_attr) + '_metrics'] = []
        print("----------" + "test on test data" + "----------")
        uni_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                        model=pr_model,
                                                        thresh_arr=thresh_arr,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups
                                                        )
        describe_metrics_new_inputs(uni_orig_trans_metrics, thresh_arr)
        uni_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(debias_y_pred))]
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_orig_trans_metrics)

        print("----------" + "multivariate group metric test" + "----------")
        uni_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                           model=pr_model,
                                                           thresh_arr=thresh_arr,
                                                           unprivileged_groups=unprivileged_groups,
                                                           privileged_groups=privileged_groups
                                                           )
        print(uni_group_trans_metrics)
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_group_trans_metrics)

        print("----------" + "causal metric test" + "----------")
        uni_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                      model=pr_model,
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

    print("-->all results:")
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

    # processing_name = str(os.path.basename(__file__)).split("_demo")[0]
    processing_name = "PR"

    dataset_orig = eval(function)()
    dataset_orig_panel19_train, dataset_orig_panel19_test = dataset_orig.split([0.7], shuffle=True, seed=seed)

    min_max_scaler = MaxAbsScaler()
    dataset_orig_panel19_train.features = min_max_scaler.fit_transform(dataset_orig_panel19_train.features)
    dataset_orig_panel19_test.features = min_max_scaler.transform(dataset_orig_panel19_test.features)

    for protected_attribute in protected_attributes:
        print("-->sens_attr", protected_attribute)
        privileged_groups = all_privileged_groups[protected_attribute]
        print("-->privileged_groups", privileged_groups)
        unprivileged_groups = all_unprivileged_groups[protected_attribute]
        print("-->privileged_groups", unprivileged_groups)

        metric_orig_panel19_train = BinaryLabelDatasetMetric(
            dataset_orig_panel19_train,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
        explainer_orig_panel19_train = MetricTextExplainer(metric_orig_panel19_train)

        print(explainer_orig_panel19_train.disparate_impact())

        #### 3.2.1. Training model on original data
        dataset = dataset_orig_panel19_train

        # model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
        # fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
        # lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

        model = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,
                              hidden_layer_sizes=(64, 32, 16, 8, 4),
                              random_state=1, verbose=True)  # identity， relu
        lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel())

        # dimension = len(dataset.features[0])
        # model = initial_dnn(dimension)
        # sample_weight = dataset.instance_weights,
        # model.fit(x=dataset.features, y=dataset.labels.ravel(), sample_weight=dataset.instance_weights, batch_size=32,
        #           epochs=500)
        # lr_orig_panel19 = model

        #### 3.2.3. Testing LR model on original data
        # thresh_arr = np.linspace(0.01, 0.5, 50)
        thresh_arr = np.array([0.5])
        orig_metrics = metric_test1(dataset=dataset_orig_panel19_test,
                                    model=lr_orig_panel19,
                                    thresh_arr=thresh_arr,
                                    privileged_groups=privileged_groups,
                                    unprivileged_groups=unprivileged_groups)

        print("-->Testing MLP model on original data")
        describe_metrics(orig_metrics, thresh_arr)

        ### 5.1. Learning a Prejudice Remover (PR) model on original data
        print("-->Learning Prejudice Remover (PR) model")

        model = PrejudiceRemover(sensitive_attr=protected_attribute, eta=25.0)  # eta=25.0
        # pr_orig_scaler = StandardScaler()
        # dataset = dataset_orig_panel19_train.copy()
        # dataset.features = pr_orig_scaler.fit_transform(dataset.features)
        # pr_model = model.fit(dataset)
        pr_model = model.fit(dataset_orig_panel19_train)

        dataset_pr_train = pr_model.predict(dataset_orig_panel19_train)
        dataset_pr_test = pr_model.predict(dataset_orig_panel19_test)

        #### 5.1.3. Testing PR model

        # dataset = dataset_orig_panel19_test.copy()
        # dataset.features = pr_orig_scaler.transform(dataset.features)

        pd_metrics = get_metrics(dataset_orig_panel19_test, dataset_pr_test, privileged_groups=privileged_groups,
                                   unprivileged_groups=unprivileged_groups)

        # pd_metrics = metric_test(dataset=dataset_orig_panel19_test,
        #                          model=pr_model,
        #                          thresh_arr=thresh_arr,
        #                          privileged_groups=privileged_groups,
        #                          unprivileged_groups=unprivileged_groups
        #                          )

        # pd_metrics = metric_test1(dataset=dataset_orig_panel19_test,
        #                                model=pd_model,
        #                                thresh_arr=thresh_arr,
        #                                privileged_groups=privileged_groups,
        #                                unprivileged_groups=unprivileged_groups)

        print("-->Testing PR model on original data")
        describe_metrics(pd_metrics, thresh_arr)

        Plot_class = Plot(dataset_name=dataset_name, sens_attr=protected_attribute, processing_name=processing_name)
        # Plot.plot_acc_metric(lr_orig_metrics, lr_transf_metrics, "stat_par_diff")
        multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
        Plot_class.plot_acc_multi_metric(orig_metrics, pd_metrics, multi_metric_names)

dataset_name = "Compas"
# metric_test(dataset_name)
PR_metric_test(dataset_name)