# # Load all necessary packages
# import sys
# import numpy as np
# import pandas as pd
#
# sys.path.append("../")
# from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
# from aif360.metrics import BinaryLabelDatasetMetric
# from aif360.metrics import ClassificationMetric
# from aif360.metrics.utils import compute_boolean_conditioning_vector
# from collections import defaultdict
#
# from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
#                 import load_preproc_data_adult, load_preproc_data_compas
#
# from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, initial_dnn2, describe, \
#     metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
#
# from sklearn.preprocessing import scale
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
#
# from IPython.display import Markdown, display
# import matplotlib.pyplot as plt
#
# import random
# seed = 1
# random.seed(seed)
# np.random.seed(seed)
#
# ## import dataset
# dataset_used = "german"  # "adult", "german", "compas"
# protected_attribute_used = 1  # 1, 2, 3
#
# if dataset_used == "adult":
#     dataset_orig = AdultDataset()
#     #     dataset_orig = load_preproc_data_adult()
#     if protected_attribute_used == 1:
#         privileged_groups = [{'sex': 1}]
#         unprivileged_groups = [{'sex': 0}]
#     elif protected_attribute_used == 2:
#         privileged_groups = [{'race': 1}]
#         unprivileged_groups = [{'race': 0}]
#     else:
#         privileged_groups = [{'sex': 1}, {'race': 1}]
#         unprivileged_groups = [{'sex': 0}, {'race': 0}]
# elif dataset_used == "german":
#     dataset_orig = GermanDataset()
#     if protected_attribute_used == 1:
#         privileged_groups = [{'sex': 1}]
#         unprivileged_groups = [{'sex': 0}]
#     elif protected_attribute_used == 2:
#         privileged_groups = [{'age': 1}]
#         unprivileged_groups = [{'age': 0}]
#     else:
#         privileged_groups = [{'sex': 1}, {'age': 1}]
#         unprivileged_groups = [{'sex': 0}, {'age': 0}]
# elif dataset_used == "compas":
#     #     dataset_orig = CompasDataset()
#     dataset_orig = load_preproc_data_compas()
#     if protected_attribute_used == 1:
#         privileged_groups = [{'sex': 1}]
#         unprivileged_groups = [{'sex': 0}]
#     elif protected_attribute_used == 2:
#         privileged_groups = [{'race': 1}]
#         unprivileged_groups = [{'race': 0}]
#     else:
#         privileged_groups = [{'sex': 1}, {'race': 1}]
#         unprivileged_groups = [{'sex': 0}, {'race': 0}]
#
#     # cost constraint of fnr will optimize generalized false negative rates, that of
# # fpr will optimize generalized false positive rates, and weighted will optimize
# # a weighted combination of both
# cost_constraint = "fnr"  # "fnr", "fpr", "weighted"
# # random seed for calibrated equal odds prediction
# randseed = 12345679
#
#
# dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True, seed=seed)
# dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True, seed=seed)
#
#
# #### Training data characteristics
#
# # print out some labels, names, etc.
# print(dataset_orig_train.features.shape)
# print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
# print(dataset_orig_train.protected_attribute_names)
# print(dataset_orig_train.privileged_protected_attributes, dataset_orig_train.unprivileged_protected_attributes)
# print(dataset_orig_train.feature_names)
#
# #### Metric for the original datasets (without any classifiers)
#
#
# metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
#                              unprivileged_groups=unprivileged_groups,
#                              privileged_groups=privileged_groups)
# print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
#
# metric_orig_valid = BinaryLabelDatasetMetric(dataset_orig_valid,
#                              unprivileged_groups=unprivileged_groups,
#                              privileged_groups=privileged_groups)
# print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_valid.mean_difference())
#
# metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
#                              unprivileged_groups=unprivileged_groups,
#                              privileged_groups=privileged_groups)
# print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())
#
# ### Train classifier (logistic regression on original training data)
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_curve
#
# # Placeholder for predicted and transformed datasets
# dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
# dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
# dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
#
# dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
# dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)
#
# # Logistic regression classifier and predictions for training data
# # scale_orig = StandardScaler()
# # X_train = scale_orig.fit_transform(dataset_orig_train.features)
# # y_train = dataset_orig_train.labels.ravel()
#
# X_train = dataset_orig_train.features
# y_train = dataset_orig_train.labels.ravel()
#
# BATCH_SIZE = 32
# EPOCHS = 500
# dimension = len(X_train[0])
# lmod = initial_dnn2(dimension)
# lmod.fit(x=X_train, y=y_train,
#         sample_weight=dataset_orig_train.instance_weights,
#         batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=1)
#
# # lmod = LogisticRegression()
# # lmod.fit(X_train, y_train)
#
# # fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
# fav_idx = 1
# print("-->fav_idx", fav_idx)
# # print("-->y_train_pred_prob", lmod.predict_proba(X_train))
# y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]
# # print("-->y_train_pred_prob", y_train_pred_prob)
#
# # Prediction probs for validation and testing data
# # X_valid = scale_orig.transform(dataset_orig_valid.features)
# X_valid = dataset_orig_valid.features
# y_valid_pred_prob = lmod.predict_proba(X_valid)[:, fav_idx]
#
# # X_test = scale_orig.transform(dataset_orig_test.features)
# X_test = dataset_orig_test.features
# y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]
#
# class_thresh = 0.5
# dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
# dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)
# dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)
# # print("-->dataset_orig_train_pred.score", dataset_orig_train_pred.scores)
#
# # print("-->dataset_orig_train_pred.labels", dataset_orig_train_pred.labels)
# y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
# y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
# y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
# dataset_orig_train_pred.labels = y_train_pred
# # print("-->dataset_orig_train_pred.labels", dataset_orig_train_pred.labels)
#
# y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
# y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
# y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
# dataset_orig_valid_pred.labels = y_valid_pred
#
# y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
# y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
# y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
# dataset_orig_test_pred.labels = y_test_pred
#
# #### Results before post-processing
#
# cm_pred_train = ClassificationMetric(dataset_orig_train, dataset_orig_train_pred,
#                              unprivileged_groups=unprivileged_groups,
#                              privileged_groups=privileged_groups)
# print("Difference in GFPR between unprivileged and privileged groups")
# print(cm_pred_train.difference(cm_pred_train.generalized_false_positive_rate))
# print("Difference in GFNR between unprivileged and privileged groups")
# print(cm_pred_train.difference(cm_pred_train.generalized_false_negative_rate))
#
# cm_pred_valid = ClassificationMetric(dataset_orig_valid, dataset_orig_valid_pred,
#                              unprivileged_groups=unprivileged_groups,
#                              privileged_groups=privileged_groups)
# print("Difference in GFPR between unprivileged and privileged groups")
# print(cm_pred_valid.difference(cm_pred_valid.generalized_false_positive_rate))
# print("Difference in GFNR between unprivileged and privileged groups")
# print(cm_pred_valid.difference(cm_pred_valid.generalized_false_negative_rate))
#
# cm_pred_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
#                              unprivileged_groups=unprivileged_groups,
#                              privileged_groups=privileged_groups)
# print("Difference in GFPR between unprivileged and privileged groups")
# print(cm_pred_test.difference(cm_pred_test.generalized_false_positive_rate))
# print("Difference in GFNR between unprivileged and privileged groups")
# print(cm_pred_test.difference(cm_pred_test.generalized_false_negative_rate))
# print("-->accuracy:", cm_pred_test.accuracy())
#
# thresh_arr = [0.5]
#
# print("----------" + "test on test data" + "----------")
# multi_orig_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
#                                                 model=lmod,
#                                                 thresh_arr=None,
#                                                 unprivileged_groups=unprivileged_groups,
#                                                 privileged_groups=privileged_groups,
#                                                 dataset_pred=dataset_orig_test_pred.scores
#                                                 )
# describe_metrics_new_inputs(multi_orig_metrics, thresh_arr)
#
# y_test = dataset_orig_test.labels.ravel()
# y_pred = lmod.predict_classes(dataset_orig_test.features)
# multi_orig_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
# print("-->multi_orig_metrics", multi_orig_metrics)
#
#
# print("----------" + "multivariate group metric test" + "----------")
# multi_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,
#                                                    model=lmod,
#                                                    thresh_arr=None,
#                                                    unprivileged_groups=unprivileged_groups,
#                                                    privileged_groups=privileged_groups,
#                                                    dataset_pred=dataset_orig_test_pred.scores
#                                                    )
# print(multi_group_metrics)
#
#
# print("----------" + "causal metric test" + "----------")
# multi_causal_metrics = metric_test_causal(dataset=dataset_orig_test,
#                                               model=lmod,
#                                               thresh_arr=None,
#                                               unprivileged_groups=unprivileged_groups,
#                                               privileged_groups=privileged_groups,
#                                             dataset_pred=dataset_orig_test_pred.scores
#                                               )
# print(multi_causal_metrics)
#
# ### Perform odds equalizing post processing on scores
#
# # Odds equalizing post-processing algorithm
# from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
# from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
# from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
# from tqdm import tqdm
#
# # Learn parameters to equalize odds and apply to create a new dataset
# cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
#                                      unprivileged_groups = unprivileged_groups,
#                                      cost_constraint=cost_constraint,
#                                      seed=randseed)
#
# # cpp = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups,
# #                                privileged_groups=privileged_groups, seed=randseed)
#
# # metric_name = "Statistical parity difference"
# # Upper and lower bound on the fairness metric used
# # metric_ub = 0.05
# # metric_lb = -0.05
# # cpp = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
# #                                  privileged_groups=privileged_groups,
# #                                  low_class_thresh=0.01, high_class_thresh=0.99,
# #                                   num_class_thresh=100, num_ROC_margin=50,
# #                                   metric_name=metric_name,
# #                                   metric_ub=metric_ub, metric_lb=metric_lb)
#
# # cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)
# cpp = cpp.fit(dataset_orig_train, dataset_orig_train_pred)
#
# ### Transform validation and test data using the post processing algorithm
#
# dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)
# dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)
#
# #### Results after post-processing
#
# cm_transf_valid = ClassificationMetric(dataset_orig_valid, dataset_transf_valid_pred,
#                              unprivileged_groups=unprivileged_groups,
#                              privileged_groups=privileged_groups)
# display(Markdown("#### Original-Transformed validation dataset"))
# print("Difference in GFPR between unprivileged and privileged groups")
# print(cm_transf_valid.difference(cm_transf_valid.generalized_false_positive_rate))
# print("Difference in GFNR between unprivileged and privileged groups")
# print(cm_transf_valid.difference(cm_transf_valid.generalized_false_negative_rate))
#
# cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
#                              unprivileged_groups=unprivileged_groups,
#                              privileged_groups=privileged_groups)
# display(Markdown("#### Original-Transformed testing dataset"))
# print("Difference in GFPR between unprivileged and privileged groups")
# print(cm_transf_test.difference(cm_transf_test.generalized_false_positive_rate))
# print("Difference in GFNR between unprivileged and privileged groups")
# print(cm_transf_test.difference(cm_transf_test.generalized_false_negative_rate))
# print("-->accuracy:", cm_transf_test.accuracy())
#
# print("----------" + "test on test data" + "----------")
# multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
#                                                       model=cpp,
#                                                       thresh_arr=None,
#                                                       unprivileged_groups=unprivileged_groups,
#                                                       privileged_groups=privileged_groups,
#                                                       dataset_pred=dataset_transf_test_pred.scores
#                                                       )
# describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)
# print(multi_orig_trans_metrics )
#
# # y_test = dataset_orig_test.labels.ravel()
# # debias_y_pred = cpp.predict(dataset_orig_test.features)
# # multi_orig_metrics['acc'] = [accuracy_score(list(y_test), list(debias_y_pred))]
# # multi_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(debias_y_pred))]
#
# print("----------" + "multivariate group metric test" + "----------")
# multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
#                                                          model=cpp,
#                                                          thresh_arr=None,
#                                                          unprivileged_groups=unprivileged_groups,
#                                                          privileged_groups=privileged_groups,
#                                                          dataset_pred=dataset_transf_test_pred.scores
#                                                          )
# print(multi_group_trans_metrics)
#
#
# print("----------" + "causal metric test" + "----------")
# multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
#                                                     model=cpp,
#                                                     thresh_arr=None,
#                                                     unprivileged_groups=unprivileged_groups,
#                                                     privileged_groups=privileged_groups,
#                                                     dataset_pred=dataset_transf_test_pred.scores
#                                                     )
# print(multi_causal_trans_metrics)
#
# print("-->all results")
# multi_orig_metrics = [multi_orig_metrics, multi_group_metrics, multi_causal_metrics]
# all_multi_orig_metrics = defaultdict(list)
# for to_merge in multi_orig_metrics:
#     for key, value in to_merge.items():
#         all_multi_orig_metrics[key].append(value[0])
#
# multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics, multi_causal_trans_metrics]
# all_multi_trans_metrics = defaultdict(list)
# for to_merge in multi_trans_metrics:
#     for key, value in to_merge.items():
#         all_multi_trans_metrics[key].append(value[0])
# print([dict(all_multi_orig_metrics), dict(all_multi_trans_metrics)])
#
# # Testing: Check if the rates for validation data has gone down
# # assert np.abs(cm_transf_valid.difference(cm_transf_valid.generalized_false_negative_rate)) < np.abs(cm_pred_valid.difference(cm_pred_valid.generalized_false_negative_rate))
#
# # Thresholds
# all_thresh = np.linspace(0.01, 0.99, 25)
# display(Markdown("#### Classification thresholds used for validation and parameter selection"))
#
# bef_avg_odds_diff_test = []
# bef_avg_odds_diff_valid = []
# aft_avg_odds_diff_test = []
# aft_avg_odds_diff_valid = []
# bef_bal_acc_valid = []
# bef_bal_acc_test = []
# aft_bal_acc_valid = []
# aft_bal_acc_test = []
#
# bef_acc_valid = []
# bef_acc_test = []
# aft_acc_valid = []
# aft_acc_test = []
#
# for thresh in tqdm(all_thresh):
#     dataset_orig_valid_pred_thresh = dataset_orig_valid_pred.copy(deepcopy=True)
#     dataset_orig_test_pred_thresh = dataset_orig_test_pred.copy(deepcopy=True)
#     dataset_transf_valid_pred_thresh = dataset_transf_valid_pred.copy(deepcopy=True)
#     dataset_transf_test_pred_thresh = dataset_transf_test_pred.copy(deepcopy=True)
#
#     # Labels for the datasets from scores
#     y_temp = np.zeros_like(dataset_orig_valid_pred_thresh.labels)
#     y_temp[dataset_orig_valid_pred_thresh.scores >= thresh] = dataset_orig_valid_pred_thresh.favorable_label
#     y_temp[~(dataset_orig_valid_pred_thresh.scores >= thresh)] = dataset_orig_valid_pred_thresh.unfavorable_label
#     dataset_orig_valid_pred_thresh.labels = y_temp
#
#     y_temp = np.zeros_like(dataset_orig_test_pred_thresh.labels)
#     y_temp[dataset_orig_test_pred_thresh.scores >= thresh] = dataset_orig_test_pred_thresh.favorable_label
#     y_temp[~(dataset_orig_test_pred_thresh.scores >= thresh)] = dataset_orig_test_pred_thresh.unfavorable_label
#     dataset_orig_test_pred_thresh.labels = y_temp
#
#     y_temp = np.zeros_like(dataset_transf_valid_pred_thresh.labels)
#     y_temp[dataset_transf_valid_pred_thresh.scores >= thresh] = dataset_transf_valid_pred_thresh.favorable_label
#     y_temp[~(dataset_transf_valid_pred_thresh.scores >= thresh)] = dataset_transf_valid_pred_thresh.unfavorable_label
#     dataset_transf_valid_pred_thresh.labels = y_temp
#
#     y_temp = np.zeros_like(dataset_transf_test_pred_thresh.labels)
#     y_temp[dataset_transf_test_pred_thresh.scores >= thresh] = dataset_transf_test_pred_thresh.favorable_label
#     y_temp[~(dataset_transf_test_pred_thresh.scores >= thresh)] = dataset_transf_test_pred_thresh.unfavorable_label
#     dataset_transf_test_pred_thresh.labels = y_temp
#
#     # Metrics for original validation data
#     classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
#                                                         dataset_orig_valid_pred_thresh,
#                                                         unprivileged_groups=unprivileged_groups,
#                                                         privileged_groups=privileged_groups)
#     # bef_avg_odds_diff_valid.append(classified_metric_orig_valid.equal_opportunity_difference())
#     bef_avg_odds_diff_valid.append(classified_metric_orig_valid.statistical_parity_difference())
#
#     bef_bal_acc_valid.append(0.5 * (classified_metric_orig_valid.true_positive_rate() +
#                                     classified_metric_orig_valid.true_negative_rate()))
#     bef_acc_valid.append(classified_metric_orig_valid.accuracy())
#
#     classified_metric_orig_test = ClassificationMetric(dataset_orig_test,
#                                                        dataset_orig_test_pred_thresh,
#                                                        unprivileged_groups=unprivileged_groups,
#                                                        privileged_groups=privileged_groups)
#     # bef_avg_odds_diff_test.append(classified_metric_orig_test.equal_opportunity_difference())
#     bef_avg_odds_diff_test.append(classified_metric_orig_test.statistical_parity_difference())
#     bef_bal_acc_test.append(0.5 * (classified_metric_orig_test.true_positive_rate() +
#                                    classified_metric_orig_test.true_negative_rate()))
#     bef_acc_test.append(classified_metric_orig_test.accuracy())
#
#     # Metrics for transf validing data
#     classified_metric_transf_valid = ClassificationMetric(
#         dataset_orig_valid,
#         dataset_transf_valid_pred_thresh,
#         unprivileged_groups=unprivileged_groups,
#         privileged_groups=privileged_groups)
#     # aft_avg_odds_diff_valid.append(classified_metric_transf_valid.equal_opportunity_difference())
#     aft_avg_odds_diff_valid.append(classified_metric_transf_valid.statistical_parity_difference())
#     aft_bal_acc_valid.append(0.5 * (classified_metric_transf_valid.true_positive_rate() +
#                                     classified_metric_transf_valid.true_negative_rate()))
#     aft_acc_valid.append(classified_metric_transf_valid.accuracy())
#
#     # Metrics for transf validation data
#     classified_metric_transf_test = ClassificationMetric(dataset_orig_test,
#                                                          dataset_transf_test_pred_thresh,
#                                                          unprivileged_groups=unprivileged_groups,
#                                                          privileged_groups=privileged_groups)
#     # aft_avg_odds_diff_test.append(classified_metric_transf_test.equal_opportunity_difference())
#     aft_avg_odds_diff_test.append(classified_metric_transf_test.statistical_parity_difference())
#     aft_bal_acc_test.append(0.5 * (classified_metric_transf_test.true_positive_rate() +
#                                    classified_metric_transf_test.true_negative_rate()))
#     aft_acc_test.append(classified_metric_transf_test.accuracy())
#
#     # print("----------" + "test on test data" + "----------")
#     # multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
#     #                                                   model=cpp,
#     #                                                   thresh_arr=[thresh],
#     #                                                   unprivileged_groups=unprivileged_groups,
#     #                                                   privileged_groups=privileged_groups
#     #                                                   )
#     # describe_metrics_new_inputs(multi_orig_trans_metrics, thresh)
#     # # multi_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(debias_y_pred))]
#     #
#     # print("----------" + "multivariate group metric test" + "----------")
#     # multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
#     #                                                      model=cpp,
#     #                                                      thresh_arr=[thresh],
#     #                                                      unprivileged_groups=unprivileged_groups,
#     #                                                      privileged_groups=privileged_groups
#     #                                                      )
#     # print(multi_group_trans_metrics)
#     #
#     # print("----------" + "causal metric test" + "----------")
#     # multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
#     #                                                 model=cpp,
#     #                                                 thresh_arr=[thresh],
#     #                                                 unprivileged_groups=unprivileged_groups,
#     #                                                 privileged_groups=privileged_groups
#     #                                                 )
#     # print(multi_causal_trans_metrics)
#
# bef_bal_acc_valid = np.array(bef_bal_acc_valid)
# bef_avg_odds_diff_valid = np.array(bef_avg_odds_diff_valid)
# aft_bal_acc_valid = np.array(aft_bal_acc_valid)
# aft_avg_odds_diff_valid = np.array(aft_avg_odds_diff_valid)
#
# bef_acc_valid = np.array(bef_acc_valid)
# bef_avg_odds_diff_valid = np.array(bef_avg_odds_diff_valid)
# aft_acc_valid = np.array(aft_acc_valid)
# aft_avg_odds_diff_valid = np.array(aft_avg_odds_diff_valid)
#
# fig, ax1 = plt.subplots(figsize=(13,7))
# ax1.plot(all_thresh, bef_bal_acc_valid, color='b')
# ax1.plot(all_thresh, aft_bal_acc_valid, color='b', linestyle='dashed')
# ax1.plot(all_thresh, bef_acc_valid, color='black')
# ax1.plot(all_thresh, aft_acc_valid, color='black', linestyle='dashed')
# ax1.set_title('Original and Postprocessed validation data', fontsize=16, fontweight='bold')
# ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
# ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
# ax1.xaxis.set_tick_params(labelsize=14)
# ax1.yaxis.set_tick_params(labelsize=14)
#
# ax2 = ax1.twinx()
# ax2.plot(all_thresh, np.abs(bef_avg_odds_diff_valid), color='r')
# ax2.plot(all_thresh, np.abs(aft_avg_odds_diff_valid), color='r', linestyle='dashed')
# ax2.set_ylabel('abs(Equal opportunity diff)', color='r', fontsize=16, fontweight='bold')
# ax2.yaxis.set_tick_params(labelsize=14)
# ax2.grid(True)
# fig.legend(["Balanced Acc. - Orig.", "Balanced Acc. - Postproc.",
#              "Equal opp. diff. - Orig.","Equal opp. diff. - Postproc.",],
#            fontsize=16)
# plt.savefig("test1.png")
#
#
# bef_bal_acc_test = np.array(bef_bal_acc_test)
# bef_avg_odds_diff_test = np.array(bef_avg_odds_diff_test)
# aft_bal_acc_test = np.array(aft_bal_acc_test)
# aft_avg_odds_diff_test = np.array(aft_avg_odds_diff_test)
#
# bef_acc_test = np.array(bef_acc_test)
# bef_avg_odds_diff_test = np.array(bef_avg_odds_diff_test)
# aft_acc_test = np.array(aft_acc_test)
# aft_avg_odds_diff_test = np.array(aft_avg_odds_diff_test)
#
# print("-->bef_bal_acc_test", bef_bal_acc_test)
# print("-->bef_acc_test", bef_acc_test)
# print("-->aft_bal_acc_test", aft_bal_acc_test)
# print("-->after_acc_test", aft_acc_test)
#
# fig, ax1 = plt.subplots(figsize=(13,7))
# ax1.plot(all_thresh, bef_bal_acc_test, color='b')
# ax1.plot(all_thresh, aft_bal_acc_test, color='b', linestyle='dashed')
# ax1.plot(all_thresh, bef_acc_test, color='black')
# ax1.plot(all_thresh, aft_acc_test, color='black', linestyle='dashed')
# ax1.set_title('Original and Postprocessed testing data', fontsize=16, fontweight='bold')
# ax1.set_xlabel('Classification Thresholds', fontsize=16, fontweight='bold')
# ax1.set_ylabel('Balanced Accuracy', color='b', fontsize=16, fontweight='bold')
# ax1.xaxis.set_tick_params(labelsize=14)
# ax1.yaxis.set_tick_params(labelsize=14)
#
#
# ax2 = ax1.twinx()
# ax2.plot(all_thresh, np.abs(bef_avg_odds_diff_test), color='r')
# ax2.plot(all_thresh, np.abs(aft_avg_odds_diff_test), color='r', linestyle='dashed')
# ax2.set_ylabel('abs(Equal opportunity diff)', color='r', fontsize=16, fontweight='bold')
# ax2.yaxis.set_tick_params(labelsize=14)
# ax2.grid(True)
# fig.legend(["Balanced Acc. - Orig.", "Balanced Acc. - Postproc.",
#             "Equal opp. diff. - Orig.", "Equal opp. diff. - Postproc."],
#            fontsize=16)
# plt.savefig("test2.png")


# Load all necessary packages
import sys
import numpy as np
import pandas as pd

sys.path.append("../")
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
from aif360.datasets.compas_dataset1 import CompasDataset_1
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from collections import defaultdict

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
                import load_preproc_data_adult, load_preproc_data_compas

from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, initial_dnn2, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal

from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

import random
seed = 1
random.seed(seed)
np.random.seed(seed)

## import dataset
dataset_used = "adult"  # "adult", "german", "compas", "bank"
protected_attribute_used = 1  # 1, 2, 3

if dataset_used == "adult":
    dataset_orig = AdultDataset()
    # dataset_orig = load_preproc_data_adult()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
    elif protected_attribute_used == 2:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
    else:
        privileged_groups = [{'sex': 1}, {'race': 1}]
        unprivileged_groups = [{'sex': 0}, {'race': 0}]
elif dataset_used == "german":
    dataset_orig = GermanDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
    elif protected_attribute_used == 2:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
    else:
        privileged_groups = [{'sex': 1}, {'age': 1}]
        unprivileged_groups = [{'sex': 0}, {'age': 0}]
elif dataset_used == "compas":
    dataset_orig = CompasDataset_1()
    # dataset_orig = load_preproc_data_compas()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
    elif protected_attribute_used == 2:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
    else:
        privileged_groups = [{'sex': 1}, {'race': 1}]
        unprivileged_groups = [{'sex': 0}, {'race': 0}]
elif dataset_used == "bank":
    dataset_orig = BankDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]

# cost constraint of fnr will optimize generalized false negative rates, that of
# fpr will optimize generalized false positive rates, and weighted will optimize
# a weighted combination of both
cost_constraint = "fnr"  # "fnr", "fpr", "weighted"
# random seed for calibrated equal odds prediction
randseed = 12345679


dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True, seed=seed)
dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True, seed=seed)

acc_train, acc_test = dataset_orig.split([0.7], shuffle=True, seed=seed)


#### Training data characteristics

# print out some labels, names, etc.
print(dataset_orig_train.features.shape)
print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
print(dataset_orig_train.protected_attribute_names)
print(dataset_orig_train.privileged_protected_attributes, dataset_orig_train.unprivileged_protected_attributes)
print(dataset_orig_train.feature_names)

#### Metric for the original datasets (without any classifiers)


metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

# metric_orig_valid = BinaryLabelDatasetMetric(dataset_orig_valid,
#                              unprivileged_groups=unprivileged_groups,
#                              privileged_groups=privileged_groups)
# print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_valid.mean_difference())

metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())

### Train classifier (logistic regression on original training data)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

# Placeholder for predicted and transformed datasets
dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

acc_test_pred = acc_test.copy(deepcopy=True)

dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)

# Logistic regression classifier and predictions for training data
scale_orig = StandardScaler()
X_train = scale_orig.fit_transform(dataset_orig_train.features)
y_train = dataset_orig_train.labels.ravel()

# X_train = dataset_orig_train.features
# y_train = dataset_orig_train.labels.ravel()

BATCH_SIZE = 128
EPOCHS = 1000
dimension = len(X_train[0])
lmod = initial_dnn2(dimension)
lmod.fit(x=X_train, y=y_train,
        sample_weight=dataset_orig_train.instance_weights,
        batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=1)

thresh_arr = [0.5]

# predict_prob = lmod.predict(dataset_orig_test.features)
# print(predict_prob)
# predict_classes = np.argmax(predict_prob,axis=1)
# print(predict_classes)

print("----------" + "test on test data" + "----------")
multi_orig_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                model=lmod,
                                                thresh_arr=thresh_arr,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups
                                                )
describe_metrics_new_inputs(multi_orig_metrics, thresh_arr)

print("-->multi_orig_metrics", multi_orig_metrics)


# print("----------" + "multivariate group metric test" + "----------")
# multi_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,
#                                                    model=lmod,
#                                                    thresh_arr=thresh_arr,
#                                                    unprivileged_groups=unprivileged_groups,
#                                                    privileged_groups=privileged_groups
#                                                    )
# print(multi_group_metrics)
#
#
# print("----------" + "causal metric test" + "----------")
# multi_causal_metrics = metric_test_causal(dataset=dataset_orig_test,
#                                               model=lmod,
#                                               thresh_arr=thresh_arr,
#                                               unprivileged_groups=unprivileged_groups,
#                                               privileged_groups=privileged_groups
#                                               )
# print(multi_causal_metrics)


# lmod = LogisticRegression()
# lmod.fit(X_train, y_train)

# fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
fav_idx = 1
print("-->fav_idx", fav_idx)
# print("-->y_train_pred_prob", lmod.predict_proba(X_train))
# y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]
y_train_pred_prob = lmod.predict(X_train)[:, fav_idx]
# print("-->y_train_pred_prob", y_train_pred_prob)

# Prediction probs for validation and testing data
X_valid = scale_orig.transform(dataset_orig_valid.features)
# X_valid = dataset_orig_valid.features
# y_valid_pred_prob = lmod.predict_proba(X_valid)[:, fav_idx]
y_valid_pred_prob = lmod.predict(X_valid)[:, fav_idx]

X_test = scale_orig.transform(dataset_orig_test.features)
# X_test = dataset_orig_test.features
# y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]
y_test_pred_prob = lmod.predict(X_test)[:, fav_idx]

class_thresh = 0.5
dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)
dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

# print("-->dataset_orig_train_pred.score", dataset_orig_train_pred.scores)

# print("-->dataset_orig_train_pred.labels", dataset_orig_train_pred.labels)
y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
dataset_orig_train_pred.labels = y_train_pred
# print("-->dataset_orig_train_pred.labels", dataset_orig_train_pred.labels)

y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
dataset_orig_valid_pred.labels = y_valid_pred

y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
dataset_orig_test_pred.labels = y_test_pred


acc_X_test = acc_test.features
# acc_y_test_pred_prob = lmod.predict_proba(acc_X_test)[:, fav_idx]
acc_y_test_pred_prob = lmod.predict(acc_X_test)[:, fav_idx]
acc_test_pred.scores = acc_y_test_pred_prob.reshape(-1, 1)
acc_y_test_pred = np.zeros_like(acc_test_pred.labels)
acc_y_test_pred[acc_y_test_pred_prob >= class_thresh] = acc_test_pred.favorable_label
acc_y_test_pred[~(acc_y_test_pred_prob >= class_thresh)] = acc_test_pred.unfavorable_label
acc_test_pred.labels = acc_y_test_pred

#### Results before post-processing

cm_pred_train = ClassificationMetric(dataset_orig_train, dataset_orig_train_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("Difference in GFPR between unprivileged and privileged groups")
print(cm_pred_train.difference(cm_pred_train.generalized_false_positive_rate))
print("Difference in GFNR between unprivileged and privileged groups")
print(cm_pred_train.difference(cm_pred_train.generalized_false_negative_rate))

cm_pred_valid = ClassificationMetric(dataset_orig_valid, dataset_orig_valid_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("Difference in GFPR between unprivileged and privileged groups")
print(cm_pred_valid.difference(cm_pred_valid.generalized_false_positive_rate))
print("Difference in GFNR between unprivileged and privileged groups")
print(cm_pred_valid.difference(cm_pred_valid.generalized_false_negative_rate))

cm_pred_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("Difference in GFPR between unprivileged and privileged groups")
print(cm_pred_test.difference(cm_pred_test.generalized_false_positive_rate))
print("Difference in GFNR between unprivileged and privileged groups")
print(cm_pred_test.difference(cm_pred_test.generalized_false_negative_rate))
print("-->accuracy:", cm_pred_test.accuracy())

acc_cm_pred_test = ClassificationMetric(acc_test, acc_test_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("-->accuracy:", acc_cm_pred_test.accuracy())

thresh_arr = [0.5]

print("----------" + "test on test data" + "----------")
multi_orig_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                model=lmod,
                                                thresh_arr=None,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups,
                                                dataset_pred=dataset_orig_test_pred.scores
                                                )
describe_metrics_new_inputs(multi_orig_metrics, thresh_arr)

# y_test = dataset_orig_test.labels.ravel()
# y_pred = lmod.predict_classes(dataset_orig_test.features)
# # y_pred = lmod.predict(dataset_orig_test.features)
# multi_orig_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
# print("-->multi_orig_metrics", multi_orig_metrics)


# print("----------" + "multivariate group metric test" + "----------")
# multi_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,
#                                                    model=lmod,
#                                                    thresh_arr=None,
#                                                    unprivileged_groups=unprivileged_groups,
#                                                    privileged_groups=privileged_groups,
#                                                    dataset_pred=dataset_orig_test_pred.scores
#                                                    )
# print(multi_group_metrics)
#
#
# print("----------" + "causal metric test" + "----------")
# multi_causal_metrics = metric_test_causal(dataset=dataset_orig_test,
#                                               model=lmod,
#                                               thresh_arr=None,
#                                               unprivileged_groups=unprivileged_groups,
#                                               privileged_groups=privileged_groups,
#                                             dataset_pred=dataset_orig_test_pred.scores
#                                               )
# print(multi_causal_metrics)

### Perform odds equalizing post processing on scores

# Odds equalizing post-processing algorithm
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from tqdm import tqdm

# Learn parameters to equalize odds and apply to create a new dataset
# cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
#                                      unprivileged_groups = unprivileged_groups,
#                                      cost_constraint=cost_constraint,
#                                      seed=randseed)

cpp = CalibratedEqOddsPostprocessing(unprivileged_groups=unprivileged_groups,  # EqoddsPostprocessing
                                              privileged_groups=privileged_groups, seed=1234567)

# cpp = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups,
#                                privileged_groups=privileged_groups, seed=randseed)

# metric_name = "Statistical parity difference"
# Upper and lower bound on the fairness metric used
# metric_ub = 0.05
# metric_lb = -0.05
# cpp = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
#                                  privileged_groups=privileged_groups,
#                                  low_class_thresh=0.01, high_class_thresh=0.99,
#                                   num_class_thresh=100, num_ROC_margin=50,
#                                   metric_name=metric_name,
#                                   metric_ub=metric_ub, metric_lb=metric_lb)

cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)
# cpp = cpp.fit(dataset_orig_train, dataset_orig_train_pred)

### Transform validation and test data using the post processing algorithm

dataset_transf_valid_pred = cpp.predict(dataset_orig_valid_pred)
dataset_transf_test_pred = cpp.predict(dataset_orig_test_pred)

acc_transf_test_pred = cpp.predict(acc_test_pred)

#### Results after post-processing

cm_transf_valid = ClassificationMetric(dataset_orig_valid, dataset_transf_valid_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
display(Markdown("#### Original-Transformed validation dataset"))
print("Difference in GFPR between unprivileged and privileged groups")
print(cm_transf_valid.difference(cm_transf_valid.generalized_false_positive_rate))
print("Difference in GFNR between unprivileged and privileged groups")
print(cm_transf_valid.difference(cm_transf_valid.generalized_false_negative_rate))

cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
display(Markdown("#### Original-Transformed testing dataset"))
print("Difference in GFPR between unprivileged and privileged groups")
print(cm_transf_test.difference(cm_transf_test.generalized_false_positive_rate))
print("Difference in GFNR between unprivileged and privileged groups")
print(cm_transf_test.difference(cm_transf_test.generalized_false_negative_rate))
print("-->accuracy:", cm_transf_test.accuracy())

acc_cm_transf_test = ClassificationMetric(acc_test, acc_transf_test_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("-->accuracy:", acc_cm_transf_test.accuracy())

print("----------" + "test on test data" + "----------")
multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                      model=cpp,
                                                      thresh_arr=None,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups,
                                                      dataset_pred=dataset_transf_test_pred.scores
                                                      )
describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)
print(multi_orig_trans_metrics )

# y_test = dataset_orig_test.labels.ravel()
# y_pred = cpp.predict(dataset_orig_test).labels
# # y_pred = lmod.predict(dataset_orig_test.features)
# multi_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
print("-->multi_orig_trans_metrics", multi_orig_trans_metrics)


# print("----------" + "multivariate group metric test" + "----------")
# multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
#                                                          model=cpp,
#                                                          thresh_arr=None,
#                                                          unprivileged_groups=unprivileged_groups,
#                                                          privileged_groups=privileged_groups,
#                                                          dataset_pred=dataset_transf_test_pred.scores
#                                                          )
# print(multi_group_trans_metrics)
#
#
# print("----------" + "causal metric test" + "----------")
# multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
#                                                     model=cpp,
#                                                     thresh_arr=None,
#                                                     unprivileged_groups=unprivileged_groups,
#                                                     privileged_groups=privileged_groups,
#                                                     dataset_pred=dataset_transf_test_pred.scores
#                                                     )
# print(multi_causal_trans_metrics)


print("----------" + "test on test data" + "----------")
multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                      model=cpp,
                                                      thresh_arr=[0.5],
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups
                                                      )
describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)
print(multi_orig_trans_metrics )

# y_test = dataset_orig_test.labels.ravel()
# y_pred = cpp.predict(dataset_orig_test).labels
# # y_pred = lmod.predict(dataset_orig_test.features)
# multi_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
print("-->multi_orig_trans_metrics", multi_orig_trans_metrics)


# print("----------" + "multivariate group metric test" + "----------")
# multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
#                                                          model=cpp,
#                                                          thresh_arr=[0.5],
#                                                          unprivileged_groups=unprivileged_groups,
#                                                          privileged_groups=privileged_groups
#                                                          )
# print(multi_group_trans_metrics)
#
#
# print("----------" + "causal metric test" + "----------")
# multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
#                                                     model=cpp,
#                                                     thresh_arr=[0.5],
#                                                     unprivileged_groups=unprivileged_groups,
#                                                     privileged_groups=privileged_groups
#                                                     )
# print(multi_causal_trans_metrics)

print("-->all results")
multi_orig_metrics = [multi_orig_metrics, multi_group_metrics, multi_causal_metrics]
all_multi_orig_metrics = defaultdict(list)
for to_merge in multi_orig_metrics:
    for key, value in to_merge.items():
        all_multi_orig_metrics[key].append(value[0])

multi_trans_metrics = [multi_orig_trans_metrics, multi_group_trans_metrics, multi_causal_trans_metrics]
all_multi_trans_metrics = defaultdict(list)
for to_merge in multi_trans_metrics:
    for key, value in to_merge.items():
        all_multi_trans_metrics[key].append(value[0])
print([dict(all_multi_orig_metrics), dict(all_multi_trans_metrics)])

# Testing: Check if the rates for validation data has gone down
# assert np.abs(cm_transf_valid.difference(cm_transf_valid.generalized_false_negative_rate)) < np.abs(cm_pred_valid.difference(cm_pred_valid.generalized_false_negative_rate))

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

bef_acc_valid = []
bef_acc_test = []
aft_acc_valid = []
aft_acc_test = []

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
    # bef_avg_odds_diff_valid.append(classified_metric_orig_valid.equal_opportunity_difference())
    bef_avg_odds_diff_valid.append(classified_metric_orig_valid.statistical_parity_difference())

    bef_bal_acc_valid.append(0.5 * (classified_metric_orig_valid.true_positive_rate() +
                                    classified_metric_orig_valid.true_negative_rate()))
    bef_acc_valid.append(classified_metric_orig_valid.accuracy())

    classified_metric_orig_test = ClassificationMetric(dataset_orig_test,
                                                       dataset_orig_test_pred_thresh,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)
    # bef_avg_odds_diff_test.append(classified_metric_orig_test.equal_opportunity_difference())
    bef_avg_odds_diff_test.append(classified_metric_orig_test.statistical_parity_difference())
    bef_bal_acc_test.append(0.5 * (classified_metric_orig_test.true_positive_rate() +
                                   classified_metric_orig_test.true_negative_rate()))
    bef_acc_test.append(classified_metric_orig_test.accuracy())

    # Metrics for transf validing data
    classified_metric_transf_valid = ClassificationMetric(
        dataset_orig_valid,
        dataset_transf_valid_pred_thresh,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    # aft_avg_odds_diff_valid.append(classified_metric_transf_valid.equal_opportunity_difference())
    aft_avg_odds_diff_valid.append(classified_metric_transf_valid.statistical_parity_difference())
    aft_bal_acc_valid.append(0.5 * (classified_metric_transf_valid.true_positive_rate() +
                                    classified_metric_transf_valid.true_negative_rate()))
    aft_acc_valid.append(classified_metric_transf_valid.accuracy())

    # Metrics for transf validation data
    classified_metric_transf_test = ClassificationMetric(dataset_orig_test,
                                                         dataset_transf_test_pred_thresh,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)
    # aft_avg_odds_diff_test.append(classified_metric_transf_test.equal_opportunity_difference())
    aft_avg_odds_diff_test.append(classified_metric_transf_test.statistical_parity_difference())
    aft_bal_acc_test.append(0.5 * (classified_metric_transf_test.true_positive_rate() +
                                   classified_metric_transf_test.true_negative_rate()))
    aft_acc_test.append(classified_metric_transf_test.accuracy())

    # print("----------" + "test on test data" + "----------")
    # multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
    #                                                   model=cpp,
    #                                                   thresh_arr=[thresh],
    #                                                   unprivileged_groups=unprivileged_groups,
    #                                                   privileged_groups=privileged_groups
    #                                                   )
    # describe_metrics_new_inputs(multi_orig_trans_metrics, thresh)
    # # multi_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(debias_y_pred))]
    #
    # print("----------" + "multivariate group metric test" + "----------")
    # multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
    #                                                      model=cpp,
    #                                                      thresh_arr=[thresh],
    #                                                      unprivileged_groups=unprivileged_groups,
    #                                                      privileged_groups=privileged_groups
    #                                                      )
    # print(multi_group_trans_metrics)
    #
    # print("----------" + "causal metric test" + "----------")
    # multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
    #                                                 model=cpp,
    #                                                 thresh_arr=[thresh],
    #                                                 unprivileged_groups=unprivileged_groups,
    #                                                 privileged_groups=privileged_groups
    #                                                 )
    # print(multi_causal_trans_metrics)

bef_bal_acc_valid = np.array(bef_bal_acc_valid)
bef_avg_odds_diff_valid = np.array(bef_avg_odds_diff_valid)
aft_bal_acc_valid = np.array(aft_bal_acc_valid)
aft_avg_odds_diff_valid = np.array(aft_avg_odds_diff_valid)

bef_acc_valid = np.array(bef_acc_valid)
bef_avg_odds_diff_valid = np.array(bef_avg_odds_diff_valid)
aft_acc_valid = np.array(aft_acc_valid)
aft_avg_odds_diff_valid = np.array(aft_avg_odds_diff_valid)

fig, ax1 = plt.subplots(figsize=(13,7))
ax1.plot(all_thresh, bef_bal_acc_valid, color='b')
ax1.plot(all_thresh, aft_bal_acc_valid, color='b', linestyle='dashed')
ax1.plot(all_thresh, bef_acc_valid, color='black')
ax1.plot(all_thresh, aft_acc_valid, color='black', linestyle='dashed')
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
             "Equal opp. diff. - Orig.","Equal opp. diff. - Postproc.",],
           fontsize=16)
plt.savefig("test1.png")


bef_bal_acc_test = np.array(bef_bal_acc_test)
bef_avg_odds_diff_test = np.array(bef_avg_odds_diff_test)
aft_bal_acc_test = np.array(aft_bal_acc_test)
aft_avg_odds_diff_test = np.array(aft_avg_odds_diff_test)

bef_acc_test = np.array(bef_acc_test)
bef_avg_odds_diff_test = np.array(bef_avg_odds_diff_test)
aft_acc_test = np.array(aft_acc_test)
aft_avg_odds_diff_test = np.array(aft_avg_odds_diff_test)

print("-->bef_bal_acc_test", bef_bal_acc_test)
print("-->bef_acc_test", bef_acc_test)
print("-->aft_bal_acc_test", aft_bal_acc_test)
print("-->after_acc_test", aft_acc_test)

fig, ax1 = plt.subplots(figsize=(13,7))
ax1.plot(all_thresh, bef_bal_acc_test, color='b')
ax1.plot(all_thresh, aft_bal_acc_test, color='b', linestyle='dashed')
ax1.plot(all_thresh, bef_acc_test, color='black')
ax1.plot(all_thresh, aft_acc_test, color='black', linestyle='dashed')
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
plt.savefig("test2.png")
