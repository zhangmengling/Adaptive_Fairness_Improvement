# import os.path
# import random
# import matplotlib.pyplot as plt
#
# # Fairness metrics
# # from aif360.metrics import BinaryLabelDatasetMetric
# # from aif360.metrics import ClassificationMetric
#
# # Explainers
# # from aif360.explainers import MetricTextExplainer
#
# # Scalers
# from sklearn.preprocessing import StandardScaler
#
# # Classifiers
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
#
# # Bias mitigation techniques
# from aif360.algorithms.preprocessing import Reweighing
# from aif360.algorithms.inprocessing import PrejudiceRemover
#
# from sklearn.neural_network import MLPClassifier
# from plot_result import Plot
#
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization
# # import torch
# from sklearn.metrics import accuracy_score
# from collections import defaultdict
#
# from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, describe, \
#     metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
# names = locals()
# processing_name = str(os.path.basename(__file__)).split("_demo")[0]
#
#
#
# """
# global variables
# """
# seed = 1
# np.random.seed(seed)
# tf.random.set_seed(seed)
# nb_classes = 2
# BATCH_SIZE = 128
# EPOCHS = 50
# # MODEL_DIR = "compas_original_optim.h5"
# # MODEL_TRANS_DIR = "compas_optim(sex).h5"
#
#
# # import sys
# # sys.path.append("../")
#
# import numpy as np
#
# from aif360.metrics import BinaryLabelDatasetMetric
#
# from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
# from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
#             import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german
# from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
#             import get_distortion_adult, get_distortion_compas, get_distortion_german
# from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
#
# from IPython.display import Markdown, display
#
#
# dataset_name = "Adult"
#
# if dataset_name == "Adult income":
#     function = "load_preproc_data_adult"
#     protected_attributes = ['sex', 'race']
#     all_privileged_groups = {'sex': [{'sex': 1}], 'race': [{'race': 1}]}
#     all_unprivileged_groups = {'sex': [{'sex': 0}], 'race': [{'race': 0}]}
# elif dataset_name == "German credit":
#     function = "load_preproc_data_german"
#     protected_attributes = ['sex', 'age']
#     all_privileged_groups = {'sex': [{'sex': 1}], 'age': [{'age': 1}]}
#     all_unprivileged_groups = {'sex': [{'sex': 0}], 'age': [{'age': 0}]}
# else:
#     function = "load_preproc_data_compas"
#     protected_attributes = ['sex', 'race']
#     all_privileged_groups = {'sex': [{'sex': 1}], 'race': [{'race': 1}]}
#     all_unprivileged_groups = {'sex': [{'sex': 0}], 'race': [{'race': 0}]}
#
# processing_name = str(os.path.basename(__file__)).split("_demo")[0]
#
# for sens_attr in protected_attributes:
#     print("-->sens_attr", sens_attr)
#     dataset_orig = eval(function)()
#     privileged_groups = all_privileged_groups[sens_attr]
#     print("-->privileged_groups", privileged_groups)
#     unprivileged_groups = all_unprivileged_groups[sens_attr]
#     print("-->privileged_groups", unprivileged_groups)
#
#     dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed=seed)
#
#     print("-->dataset_orig_train", dataset_orig_train)
#     print(dataset_orig_train.metadata["protected_attribute_maps"])
#
#     # (dataset_orig_train,
#     #  dataset_orig_test) = AdultDataset().split([0.7], shuffle=True)
#
#     # privileged_groups = [{'sex': 1}] # White, Male
#     # unprivileged_groups = [{'sex': 0}] # Not white, Female
#
#     # metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
#     #                                              unprivileged_groups=unprivileged_groups,
#     #                                              privileged_groups=privileged_groups)
#     # print("-->Original training dataset ")
#     # print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
#     # print("Difference in statistical parity between unprivileged and privileged groups = %f" % metric_orig_train.statistical_parity_difference())
#
#     metric_orig_panel19_train = BinaryLabelDatasetMetric(
#         dataset_orig_train,
#         unprivileged_groups=unprivileged_groups,
#         privileged_groups=privileged_groups)
#     # explainer_orig_panel19_train = MetricTextExplainer(metric_orig_panel19_train)
#     # print(explainer_orig_panel19_train.disparate_impact())
#
#     dataset = dataset_orig_train
#     labels = dataset.labels.ravel()
#
#     # dimension = len(dataset.features[0])
#     # print("-->dimension:", dimension)
#     # model = initial_dnn(dimension)
#     model = MLPClassifier(solver='adam', activation='identity', max_iter=100, alpha=1e-5,
#                           hidden_layer_sizes=(64, 32, 16, 8, 4),
#                           random_state=1, verbose=True)
#     model.fit(dataset.features, dataset.labels.ravel())
#     # sample_weight=dataset.instance_weights,
#     # model.fit(x=dataset.features,y=labels, sample_weight=dataset.instance_weights, batch_size=BATCH_SIZE, epochs=EPOCHS)
#     # model.save(MODEL_DIR)
#
#     # model.load_weights(MODEL_DIR)
#
#     # model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
#     # fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
#     # lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)
#
#     lr_orig_panel19 = model
#
#     # print("-->dataset labels")
#     # print(list(dataset.labels.ravel()))
#     # print(list(lr_orig_panel19.predict(dataset.features)))
#
#     print("-->prediction accuracy on dataset", accuracy_score(list(dataset.labels.ravel()),
#                                                               list(lr_orig_panel19.predict(dataset.features))))
#     y_test = dataset_orig_train.labels.ravel()
#     y_pred = lr_orig_panel19.predict(dataset_orig_train.features)
#     print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))
#
#     # thresh_arr = np.linspace(0.01, 0.5, 50)
#     thresh_arr = np.array([0.5])
#     print("----------" + "test on test data" + "----------")
#     orig_metrics = metric_test1(dataset=dataset_orig_test,
#                                model=lr_orig_panel19,
#                                thresh_arr=thresh_arr,
#                                unprivileged_groups=unprivileged_groups,
#                                privileged_groups=privileged_groups)
#     lr_orig_best_ind = np.argmax(orig_metrics['bal_acc'])
#
#     disp_imp = np.array(orig_metrics['disp_imp'])
#     disp_imp_err = 1 - np.minimum(disp_imp, 1 / disp_imp)
#
#
#     describe_metrics(orig_metrics, thresh_arr)
#     print("-->orig_metrics:", orig_metrics['disp_imp'], orig_metrics['avg_odds_diff'], orig_metrics['stat_par_diff'],
#           orig_metrics['eq_opp_diff'])
#
#     print("---------- optimized preprocessing ----------")
#
#     optim_options = {
#         "distortion_fun": get_distortion_compas,  # get_distortion_adult, get_distortion_german, get_distortion_compas
#         "epsilon": 0.05,
#         "clist": [0.99, 1.99, 2.99],
#         "dlist": [.1, 0.05, 0]
#     }
#
#     OP = OptimPreproc(OptTools, optim_options)
#     print("-->OP fitting...")
#     OP = OP.fit(dataset_orig_train)
#     print("-->OP transforming...")
#     dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)
#
#     print("-->dataset_transf_train.feature_names", dataset_transf_train.feature_names)
#     print("-->dataset_org_train.feature_names", dataset_orig_train.feature_names)
#
#     dataset_transf_train = dataset_orig_train.align_datasets(dataset_transf_train)
#
#     metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
#                                                    unprivileged_groups=unprivileged_groups,
#                                                    privileged_groups=privileged_groups)
#
#     display(Markdown("#### Transformed training dataset"))
#     print(
#         "Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())
#     print(
#         "Difference in statistical parity between unprivileged and privileged groups = %f" % metric_transf_train.statistical_parity_difference())
#
#     print("---------- Learning a MLP on data transformed by optimized preprocessing ----------")
#
#     dataset = dataset_transf_train
#
#     from sklearn.neural_network import MLPClassifier
#
#     # clf = MLPClassifier(solver='sgd', activation='identity', max_iter=10, alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 8, 4),
#     #                     random_state=1, verbose=True)
#
#     dimension = len(dataset.features[0])
#     # model = initial_dnn(dimension)
#     model = MLPClassifier(solver='adam', activation='identity', max_iter=100, alpha=1e-5,
#                           hidden_layer_sizes=(64, 32, 16, 8, 4),
#                           random_state=1, verbose=True)
#     model.fit(dataset.features, dataset.labels.ravel())
#     # model.fit(x=dataset.features,y=labels, sample_weight=dataset.instance_weights,
#     #           batch_size=BATCH_SIZE, epochs=EPOCHS)
#     # model.save(MODEL_TRANS_DIR)
#
#     # model.load_weights(MODEL_TRANS_DIR)
#
#     # model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
#     # fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
#     # lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)
#
#     lr_transf_panel19 = model
#     '''
#     model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
#     fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
#     # lr_transf_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)
#     lr_transf_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)
#     '''
#     # print("-->dataset labels")
#     # print(list(dataset.labels.ravel()))
#     # print(list(lr_transf_panel19.predict(dataset.features)))
#
#     print("-->prediction accuracy on dataset", accuracy_score(list(dataset.labels.ravel()),
#                                                               list(lr_transf_panel19.predict(dataset.features))))
#     y_test = dataset_orig_train.labels.ravel()
#     # y_pred = lr_orig_panel19.predict_classes(dataset_orig_train.features)
#     y_pred = lr_orig_panel19.predict(dataset_orig_train.features)
#     print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))
#     print("-->prediction accuracy on original dataset", accuracy_score(y_test,
#                                                                        list(lr_transf_panel19.predict(
#                                                                            dataset.features))))
#     # y_test = dataset_orig_test.labels.ravel()
#     # y_pred = lr_transf_panel19.predict_classes(dataset_orig_test.features)
#     # print("-->prediction accuracy on test data",accuracy_score(list(y_test), list(y_pred)))
#
#     print("---------- test on test data after optimized preprocessing ----------")
#
#     # thresh_arr = np.linspace(0.01, 0.5, 50)
#     thresh_arr = np.array([0.5])
#     transf_metrics = metric_test1(dataset=dataset_orig_test,
#                                model=lr_transf_panel19,
#                                thresh_arr=thresh_arr,
#                                unprivileged_groups=unprivileged_groups,
#                                privileged_groups=privileged_groups)
#     lr_transf_best_ind = np.argmax(transf_metrics['bal_acc'])
#
#     disp_imp = np.array(transf_metrics['disp_imp'])
#     disp_imp_err = 1 - np.minimum(disp_imp, 1 / disp_imp)
#
#     describe_metrics(transf_metrics, thresh_arr)
#
#     Plot_class = Plot(dataset_name=dataset_name, sens_attr=sens_attr, processing_name=processing_name)
#     # Plot.plot_acc_metric(lr_orig_metrics, lr_transf_metrics, "stat_par_diff")
#     multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
#     Plot_class.plot_acc_multi_metric(orig_metrics, transf_metrics, multi_metric_names)
#




import os.path
import random
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
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

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
from sklearn.neural_network import MLPClassifier
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_compas, get_distortion_german
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german, load_preproc_data_bank
from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, initial_dnn2, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
names = locals()
# processing_name = str(os.path.basename(__file__)).split("_demo")[0]
processing_name = "OP"
"""
global variables
"""
seed = 1
np.random.seed(seed)
# tf.random.set_seed(seed)
tf.random.set_random_seed(seed)
nb_classes = 2
# BATCH_SIZE = 128
# EPOCHS = 1000   # 500
MAX_NUM = 1000  # 251 for causal discrimination test / 2000 for group discrimination test

def OP_metric_test(dataset_name):
    print("-->OP_metric_test")
    if dataset_name == "Adult income":
        function = "load_preproc_data_adult"
        protected_attributes = ['race', 'sex']
        privileged_groups = [{'race': 1.0}, {'sex': 1.0}]
        unprivileged_groups = [{'race': 0.0}, {'sex': 0.0}]
    elif dataset_name == "German credit":
        function = "load_preproc_data_german"
        protected_attributes = ['sex', 'age']
        privileged_groups = [{'sex': 1.0}, {'age': 1.0}]
        unprivileged_groups = [{'sex': 0.0}, {'age': 0.0}]
    elif dataset_name == "Bank":
        function = "load_preproc_data_bank"
        protected_attributes = ["age"]
        privileged_groups = [{'age': 1.0}]
        unprivileged_groups = [{'age': 0.0}]
    else:
        function = "load_preproc_data_compas"
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
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

    # min_max_scaler = MaxAbsScaler()
    # dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    # dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

    print("-->dataset_orig_train", dataset_orig_train)

    def new_inputs_to_dataset(new_inputs):
        # classified_dataset, no_matter_dataset = AdultDataset().split(np.array([MAX_NUM*2]), shuffle=False)[0]
        classified_dataset = dataset_orig_train.copy()
        classified_dataset.features = np.array(new_inputs)
        length = len(new_inputs)
        classified_dataset.instance_names = [1] * length
        classified_dataset.instance_weights = np.array([1] * length)
        try:
            classified_dataset.protected_attributes = np.array(
                [[input[classified_dataset.protected_attribute_indexs[0]],
                  input[classified_dataset.protectedF_attribute_indexs[1]]]
                 for input in new_inputs])
        except:
            classified_dataset.protected_attributes = np.array(
                [[input[classified_dataset.protected_attribute_indexs[0]]]
                 for input in new_inputs])

        return classified_dataset

    # input_privileged = []
    # input_unprivileged = []
    # for i in range(0, len(privileged_groups)):
    #     privileged_group = [privileged_groups[i]]
    #     unprivileged_group = [unprivileged_groups[i]]
    #     # for group in privileged_groups:
    #     #     group = [group]
    #     new_inputs_priviledge = dataset_orig_train.generate_inputs(max_num=MAX_NUM, if_random=False,
    #                                                                privileged_groups=privileged_group,
    #                                                                if_priviledge=True)
    #     input_privileged += new_inputs_priviledge
    #     new_inputs_unpriviledge = dataset_orig_train.generate_inputs(max_num=MAX_NUM, if_random=False,
    #                                                                  privileged_groups=unprivileged_group,
    #                                                                  if_priviledge=True)
    #     input_unprivileged += new_inputs_unpriviledge
    #
    # new_inputs = input_privileged + input_unprivileged
    # print(new_inputs)
    # random.shuffle(new_inputs)
    # classified_dataset = new_inputs_to_dataset(new_inputs)
    # print("-->classified_dataset", classified_dataset)

    # classified_dataset.features = min_max_scaler.fit_transform(classified_dataset.features)

    dataset = dataset_orig_train
    labels = dataset.labels.ravel()

    dimension = len(dataset.features[0])
    model = initial_dnn2(dimension)
    model.fit(x=dataset.features, y=labels, sample_weight=dataset.instance_weights, batch_size=BATCH_SIZE,
              epochs=EPOCHS, shuffle=False, verbose=0)

    # model = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,  # solver = adam
    #                       hidden_layer_sizes=(64, 32, 16, 8, 4),
    #                       random_state=1, verbose=True)
    # model.fit(dataset.features, dataset.labels.ravel())

    MODEL_DIR = name + "/OP_model.h5"
    model.save(MODEL_DIR)

    # model = keras.models.load_model(MODEL_DIR)

    orig_model = model

    y_test = dataset_orig_test.labels.ravel()
    try:
        y_pred = orig_model.predict_classes(dataset_orig_test.features)
    except:
        y_pred = orig_model.predict(dataset_orig_test.features)
    accuracy = accuracy_score(list(y_test), list(y_pred))
    print("-->prediction accuracy on test data", accuracy)

    thresh_arr = np.array([0.5])


    print("----------" + "test on test data" + "----------")
    multi_orig_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                model=orig_model,
                                                thresh_arr=thresh_arr,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups
                                                )
    describe_metrics_new_inputs(multi_orig_metrics, thresh_arr)
    multi_orig_metrics['acc'] = [accuracy]

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
    y_test = dataset_orig_test.labels.ravel()
    try:
        y_pred = orig_model.predict_classes(dataset_orig_test.features)
    except:
        y_pred = orig_model.predict(dataset_orig_test.features)
    print(multi_causal_metrics)

    # multi_unprivileged_groups = unprivileged_groups
    # multi_privileged_groups = privileged_groups

    print("---------- optimized preprocessing ----------")
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed=seed)

    optim_options = {
        "distortion_fun": get_distortion_adult,  # get_distortion_adult, get_distortion_german, get_distortion_compas
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
    }

    print("-->OP protected_attribute", dataset_orig_train.protected_attribute_names)
    print("-->privileged_protected_attributes", dataset_orig_train.privileged_protected_attributes)
    print("-->unprivileged_protected_attributes", dataset_orig_train.unprivileged_protected_attributes)

    OP = OptimPreproc(optimizer=OptTools, optim_options=optim_options)
    print("-->OP fitting...")

    # dataset_orig_train, test = dataset_orig.split([0.1], shuffle=True, seed=seed)

    OP = OP.fit(dataset_orig_train)
    print("-->OP transforming...")
    dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)

    print("-->dataset_transf_train.feature_names", dataset_transf_train.feature_names)
    print("-->dataset_org_train.feature_names", dataset_orig_train.feature_names)

    dataset_transf_train = dataset_orig_train.align_datasets(dataset_transf_train)

    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)

    dataset = dataset_transf_train

    dimension = len(dataset.features[0])
    model = initial_dnn2(dimension)
    model.fit(x=dataset.features, y=labels, sample_weight=dataset.instance_weights, batch_size=BATCH_SIZE,
              epochs=EPOCHS, shuffle=False, verbose=0)

    # model = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,
    #                       hidden_layer_sizes=(64, 32, 16, 8, 4),
    #                       random_state=1, verbose=True)
    # model.fit(dataset.features, dataset.labels.ravel())

    trans_model = model

    y_test = dataset_orig_test.labels.ravel()
    try:
        y_pred = trans_model.predict_classes(dataset_orig_test.features)
    except:
        y_pred = trans_model.predict(dataset_orig_test.features)
    print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))

    print("----------" + "test on test data" + "----------")
    multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                      model=trans_model,
                                                      thresh_arr=thresh_arr,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups
                                                      )
    describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)

    print("----------" + "multivariate group metric test" + "----------")
    multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                         model=trans_model,
                                                         thresh_arr=thresh_arr,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups
                                                         )
    print(multi_group_trans_metrics)

    print("----------" + "causal metric test" + "----------")
    multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                    model=trans_model,
                                                    thresh_arr=thresh_arr,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups
                                                    )
    y_test = dataset_orig_test.labels.ravel()
    try:
        y_pred = trans_model.predict_classes(dataset_orig_test.features)
    except:
        y_pred = trans_model.predict(dataset_orig_test.features)
    multi_causal_trans_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
    print(multi_causal_trans_metrics)


    if len(privileged_groups) == 1:
        multi_orig_metrics = [multi_orig_metrics, multi_group_metrics]
        print("-->multi_orig_metrics", multi_orig_metrics)
        all_multi_orig_metrics = defaultdict(list)
        for to_merge in multi_orig_metrics:
            for key, value in to_merge.items():
                # print("-->value", value)
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
        return all_multi_orig_metrics, all_multi_trans_metrics, dataset_name, processing_name



    # univariate test
    all_uni_orig_metrics = []
    all_uni_trans_metrics = []

    # dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed=seed)
    sens_inds = [0, 1]
    for sens_ind in sens_inds:
        sens_attr = protected_attributes[sens_ind]
        print("-->sens_attr", sens_attr)
        names["orig_" + str(sens_attr) + '_metrics'] = []
        privileged_groups = [{sens_attr: v} for v in
                             dataset_orig_train.privileged_protected_attributes[sens_ind]]
        unprivileged_groups = [{sens_attr: v} for v in
                               dataset_orig_train.unprivileged_protected_attributes[sens_ind]]
        print("-->unprivileged_groups", unprivileged_groups)
        print("-->privileged_groups", privileged_groups)

        # dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True, seed=seed)
        """
        print("-->dataset_orig_train", dataset_orig_train)
        print(dataset_orig_train.metadata["protected_attribute_maps"])

        metric_orig_panel19_train = BinaryLabelDatasetMetric(
            dataset_orig_train,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        dataset = dataset_orig_train
        labels = dataset.labels.ravel()

        model = MLPClassifier(solver='adam', activation='identity', max_iter=100, alpha=1e-5,
                              hidden_layer_sizes=(64, 32, 16, 8, 4),
                              random_state=1, verbose=True)
        model.fit(dataset.features, dataset.labels.ravel())

        lr_orig_panel19 = model
        print("-->prediction accuracy on dataset", accuracy_score(list(dataset.labels.ravel()),
                                                                  list(lr_orig_panel19.predict(dataset.features))))
        y_test = dataset_orig_train.labels.ravel()
        y_pred = lr_orig_panel19.predict(dataset_orig_train.features)
        print("-->prediction accuracy on test data", accuracy_score(list(y_test), list(y_pred)))
        """

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
        # uni_orig_metrics['acc'] = [accuracy_score(list(dataset_orig_test.labels.ravel()),
        #                                           list(orig_model.predict(dataset_orig_test.features)))]
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_orig_metrics)

        print("----------" + "multivariate group metric test" + "----------")
        uni_group_metrics = metric_test_multivariate(dataset=classified_dataset,
                                                     model=orig_model,
                                                     thresh_arr=thresh_arr,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups
                                                     )
        print(uni_group_metrics)
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_group_metrics)

        print("----------" + "causal metric test" + "----------")
        uni_causal_metrics = metric_test_causal(dataset=classified_dataset,
                                                model=orig_model,
                                                thresh_arr=thresh_arr,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups
                                                )
        y_test = dataset_orig_test.labels.ravel()
        try:
            y_pred = orig_model.predict_classes(dataset_orig_test.features)
        except:
            y_pred = orig_model.predict(dataset_orig_test.features)
        uni_causal_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
        print(uni_causal_metrics)
        names["orig_" + str(sens_attr) + '_metrics'].append(uni_causal_metrics)

        # print("----------" + "test on test data" + "----------")
        # orig_metrics = metric_test1(dataset=dataset_orig_test,
        #                             model=lr_orig_panel19,
        #                             thresh_arr=thresh_arr,
        #                             unprivileged_groups=unprivileged_groups,
        #                             privileged_groups=privileged_groups)
        # lr_orig_best_ind = np.argmax(orig_metrics['bal_acc'])
        #
        # disp_imp = np.array(orig_metrics['disp_imp'])
        # disp_imp_err = 1 - np.minimum(disp_imp, 1 / disp_imp)
        #
        # describe_metrics(orig_metrics, thresh_arr)
        # print("-->orig_metrics:", orig_metrics['disp_imp'], orig_metrics['avg_odds_diff'], orig_metrics['stat_par_diff'],
        #       orig_metrics['eq_opp_diff'])

        print("---------- optimized preprocessing ----------")

        optim_options = {
            "distortion_fun": get_distortion_adult,
            # get_distortion_adult, get_distortion_german, get_distortion_compas
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }

        OP = OptimPreproc(optimizer=OptTools, optim_options=optim_options)
        print("-->OP fitting...")
        OP = OP.fit(dataset_orig_train)
        print("-->OP transforming...")
        dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)

        # print("-->dataset_transf_train.feature_names", dataset_transf_train.feature_names)
        # print("-->dataset_org_train.feature_names", dataset_orig_train.feature_names)

        dataset_transf_train = dataset_orig_train.align_datasets(dataset_transf_train)

        metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)

        dimension = len(dataset.features[0])
        model = initial_dnn2(dimension)
        model.fit(x=dataset.features, y=labels, sample_weight=dataset.instance_weights, batch_size=BATCH_SIZE,
                  epochs=EPOCHS, shuffle=False, verbose=0)

        # dataset = dataset_transf_train
        # model = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5,
        #                       hidden_layer_sizes=(64, 32, 16, 8, 4),
        #                       random_state=1, verbose=True)
        # model.fit(dataset.features, dataset.labels.ravel())

        trans_uni_model = model

        # print("-->prediction accuracy on dataset", accuracy_score(list(dataset.labels.ravel()),
        #                                                           list(trans_uni_model.predict(dataset.features))))
        y_test = dataset_orig_test.labels.ravel()
        try:
            y_pred = trans_uni_model.predict_classes(dataset_orig_test.features)
        except:
            y_pred = trans_uni_model.predict(dataset_orig_test.features)
        accuracy = accuracy_score(list(y_test), list(y_pred))
        print("-->prediction accuracy on test data", accuracy)
        # print("-->prediction accuracy on original dataset", accuracy_score(y_test,
        #                                                                    list(trans_uni_model.predict(
        #                                                                        dataset.features))))
        names["trans_" + str(sens_attr) + '_metrics'] = []
        print("----------" + "test on test data" + "----------")
        uni_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                        model=trans_uni_model,
                                                        thresh_arr=thresh_arr,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups
                                                        )
        describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)
        uni_orig_trans_metrics['acc'] = [accuracy]
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_orig_trans_metrics)

        print("----------" + "multivariate group metric test" + "----------")
        uni_group_trans_metrics = metric_test_multivariate(dataset=classified_dataset,
                                                           model=trans_uni_model,
                                                           thresh_arr=thresh_arr,
                                                           unprivileged_groups=unprivileged_groups,
                                                           privileged_groups=privileged_groups
                                                           )
        print(multi_group_trans_metrics)
        names["trans_" + str(sens_attr) + '_metrics'].append(uni_group_trans_metrics)

        print("----------" + "causal metric test" + "----------")
        uni_causal_trans_metrics = metric_test_causal(dataset=classified_dataset,
                                                      model=trans_uni_model,
                                                      thresh_arr=thresh_arr,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups
                                                      )
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
    print("-->multi_orig_metrics", multi_orig_metrics)
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


dataset_name = "Adult income"
print(OP_metric_test(dataset_name))










    # print("---------- test on test data after optimized preprocessing ----------")
    #
    # # thresh_arr = np.linspace(0.01, 0.5, 50)
    # thresh_arr = np.array([0.5])
    # transf_metrics = metric_test1(dataset=dataset_orig_test,
    #                               model=lr_transf_panel19,
    #                               thresh_arr=thresh_arr,
    #                               unprivileged_groups=unprivileged_groups,
    #                               privileged_groups=privileged_groups)
    # lr_transf_best_ind = np.argmax(transf_metrics['bal_acc'])
    #
    # disp_imp = np.array(transf_metrics['disp_imp'])
    # disp_imp_err = 1 - np.minimum(disp_imp, 1 / disp_imp)
    #
    # describe_metrics(transf_metrics, thresh_arr)

    # from plot_result import Plot
    # Plot_class = Plot(dataset_name=dataset_name, sens_attr=sens_attr, processing_name=processing_name)
    # # Plot.plot_acc_metric(lr_orig_metrics, lr_transf_metrics, "stat_par_diff")
    # multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
    # Plot_class.plot_acc_multi_metric(orig_metrics, transf_metrics, multi_metric_names)









