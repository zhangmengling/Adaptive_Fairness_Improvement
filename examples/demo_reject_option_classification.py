import sys
sys.path.append("../")
import numpy as np
from tqdm import tqdm
from warnings import warn

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
from aif360.datasets.compas_dataset1 import CompasDataset_1
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.postprocessing.reject_option_classification\
        import RejectOptionClassification
from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, initial_dnn2, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
from common_utils import compute_metrics

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider
import tensorflow as tf
import random
import os
# random seed for calibrated equal odds prediction
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_random_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

## import dataset
dataset_used = "compas"  # "adult", "german", "compas"
protected_attribute_used = 3  # 1, 2

if dataset_used == "adult":
    dataset_orig = AdultDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        # dataset_orig = load_preproc_data_adult(['sex'])
    elif protected_attribute_used == 2:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        # dataset_orig = load_preproc_data_adult(['race'])
    else:
        privileged_groups = [{'sex': 1}, {'race': 1}]
        unprivileged_groups = [{'sex': 0}, {'race': 0}]
elif dataset_used == "german":
    dataset_orig = GermanDataset()
    if protected_attribute_used == 1:
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        # dataset_orig = load_preproc_data_german(['sex'])
    elif protected_attribute_used == 2:
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        # dataset_orig = load_preproc_data_german(['age'])
    else:
        privileged_groups = [{'sex': 1}, {'age': 1}]
        unprivileged_groups = [{'sex': 0}, {'age': 0}]
elif dataset_used == "compas":
    dataset_orig = CompasDataset_1()
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

# Metric used (should be one of allowed_metrics)
metric_name = "Statistical parity difference"

# Upper and lower bound on the fairness metric used
metric_ub = 0.05
metric_lb = -0.05

# Verify metric name
allowed_metrics = ["Statistical parity difference",
                   "Average odds difference",
                   "Equal opportunity difference"]
if metric_name not in allowed_metrics:
    raise ValueError("Metric name should be one of allowed metrics")


# Get the dataset and split into train and test
dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True, seed=seed)
dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True, seed=seed)

acc_train, acc_test = dataset_orig.split([0.7], shuffle=True, seed=seed)


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


metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())


# Logistic regression classifier and predictions
# scale_orig = StandardScaler()
# X_train = scale_orig.fit_transform(dataset_orig_train.features)
# y_train = dataset_orig_train.labels.ravel()

X_train = dataset_orig_train.features
y_train = dataset_orig_train.labels.ravel()

BATCH_SIZE = 128
EPOCHS = 100
dimension = len(X_train[0])
lmod = initial_dnn2(dimension)
lmod.fit(x=X_train, y=y_train,
        sample_weight=dataset_orig_train.instance_weights,
        batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=1)
y_train_pred = lmod.predict_proba(X_train)

y_test = dataset_orig_test.labels.ravel()
y_pred = lmod.predict_classes(dataset_orig_test.features)
print("-->accuracy", [accuracy_score(list(y_test), list(y_pred))])

y_test = acc_test.labels.ravel()
y_pred = lmod.predict_classes(acc_test.features)
print("-->accuracy", [accuracy_score(list(y_test), list(y_pred))])
dataset_orig
# lmod = LogisticRegression()
# lmod.fit(X_train, y_train)
# y_train_pred = lmod.predict(X_train)

# positive class index
# pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
pos_ind = 1
dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
dataset_orig_train_pred.labels = y_train_pred


dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
# X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
X_valid = dataset_orig_valid_pred.features
y_valid = dataset_orig_valid_pred.labels
dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
# X_test = scale_orig.transform(dataset_orig_test_pred.features)
X_test = dataset_orig_test_pred.features
y_test = dataset_orig_test_pred.labels
dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

acc_test_pred = acc_test.copy(deepcopy=True)
# X_test = scale_orig.transform(acc_test_pred.features)
X_test = acc_test_pred.features
y_test = acc_test_pred.labels
acc_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)


num_thresh = 100
ba_arr = np.zeros(num_thresh)
class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
for idx, class_thresh in enumerate(class_thresh_arr):
    fav_inds = dataset_orig_valid_pred.scores > class_thresh
    dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
    dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

    classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                        dataset_orig_valid_pred,
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups)

    ba_arr[idx] = 0.5 * (classified_metric_orig_valid.true_positive_rate()
                         + classified_metric_orig_valid.true_negative_rate())

best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
best_class_thresh = class_thresh_arr[best_ind]

print("Best balanced accuracy (no fairness constraints) = %.4f" % np.max(ba_arr))
print("Optimal classification threshold (no fairness constraints) = %.4f" % best_class_thresh)


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

y_test = dataset_orig_test.labels.ravel()
y_pred = lmod.predict_classes(dataset_orig_test.features)
multi_orig_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
print("-->multi_orig_metrics", multi_orig_metrics)


print("----------" + "multivariate group metric test" + "----------")
multi_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                   model=lmod,
                                                   thresh_arr=None,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups,
                                                   dataset_pred=dataset_orig_test_pred.scores
                                                   )
print(multi_group_metrics)


print("----------" + "causal metric test" + "----------")
multi_causal_metrics = metric_test_causal(dataset=dataset_orig_test,
                                              model=lmod,
                                              thresh_arr=None,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups,
                                            dataset_pred=dataset_orig_test_pred.scores
                                              )
print(multi_causal_metrics)


"""
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

metric_orig_valid = BinaryLabelDatasetMetric(dataset_orig_valid,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_valid.mean_difference())

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

dataset_new_valid_pred = dataset_orig_valid.copy(deepcopy=True)
dataset_new_test_pred = dataset_orig_test.copy(deepcopy=True)

# Logistic regression classifier and predictions for training data
scale_orig = StandardScaler()
X_train = scale_orig.fit_transform(dataset_orig_train.features)
y_train = dataset_orig_train.labels.ravel()

BATCH_SIZE = 128
EPOCHS = 50
dimension = len(X_train[0])
lmod = initial_dnn2(dimension)
lmod.fit(x=X_train, y=y_train,
        sample_weight=dataset_orig_train.instance_weights,
        batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False, verbose=1)

# lmod = LogisticRegression()
# lmod.fit(X_train, y_train)

# fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
fav_idx = 1
print("-->fav_idx", fav_idx)
print("-->y_train_pred_prob", lmod.predict_proba(X_train))
y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]
print("-->y_train_pred_prob", y_train_pred_prob)

# Prediction probs for validation and testing data
X_valid = scale_orig.transform(dataset_orig_valid.features)
y_valid_pred_prob = lmod.predict_proba(X_valid)[:, fav_idx]

X_test = scale_orig.transform(dataset_orig_test.features)
y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]

class_thresh = 0.5
dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
dataset_orig_valid_pred.scores = y_valid_pred_prob.reshape(-1, 1)
dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)
print("-->dataset_orig_train_pred.score", dataset_orig_train_pred.scores)

print("-->dataset_orig_train_pred.labels", dataset_orig_train_pred.labels)
y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
dataset_orig_train_pred.labels = y_train_pred
print("-->dataset_orig_train_pred.labels", dataset_orig_train_pred.labels)

y_valid_pred = np.zeros_like(dataset_orig_valid_pred.labels)
y_valid_pred[y_valid_pred_prob >= class_thresh] = dataset_orig_valid_pred.favorable_label
y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = dataset_orig_valid_pred.unfavorable_label
dataset_orig_valid_pred.labels = y_valid_pred

y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
dataset_orig_test_pred.labels = y_test_pred

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

y_test = dataset_orig_test.labels.ravel()
y_pred = lmod.predict_classes(dataset_orig_test.features)
multi_orig_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
print("-->multi_orig_metrics", multi_orig_metrics)


print("----------" + "multivariate group metric test" + "----------")
multi_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                   model=lmod,
                                                   thresh_arr=None,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups,
                                                   dataset_pred=dataset_orig_test_pred.scores
                                                   )
print(multi_group_metrics)


print("----------" + "causal metric test" + "----------")
multi_causal_metrics = metric_test_causal(dataset=dataset_orig_test,
                                              model=lmod,
                                              thresh_arr=None,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups,
                                            dataset_pred=dataset_orig_test_pred.scores
                                              )
print(multi_causal_metrics)
"""

print("-->reject option:")

# ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
#                                  privileged_groups=privileged_groups,
#                                  low_class_thresh=0.01, high_class_thresh=0.99,
#                                   num_class_thresh=100, num_ROC_margin=50,
#                                   metric_name=metric_name,
#                                   metric_ub=metric_ub, metric_lb=metric_lb)
# ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)

# Odds equalizing post-processing algorithm
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from tqdm import tqdm

ROC = CalibratedEqOddsPostprocessing(unprivileged_groups=unprivileged_groups,  # EqoddsPostprocessing
                                              privileged_groups=privileged_groups, seed=1234567)
ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)

# Metrics for the test set
fav_inds = dataset_orig_valid_pred.scores > best_class_thresh
dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

print("Validation set: Raw predictions - No fairness constraints, only maximizing balanced accuracy")
metric_valid_bef = compute_metrics(dataset_orig_valid, dataset_orig_valid_pred,
                unprivileged_groups, privileged_groups)

# print("Optimal classification threshold (with fairness constraints) = %.4f" % ROC.classification_threshold)
# print("Optimal ROC margin = %.4f" % ROC.ROC_margin)

# class_threshold = ROC.classification_threshold
# print("-->class_threshold", class_threshold)

# Transform the validation set
dataset_transf_valid_pred = ROC.predict(dataset_orig_valid_pred)

print("Validation set: Transformed predictions - With fairness constraints")
metric_valid_aft = compute_metrics(dataset_orig_valid, dataset_transf_valid_pred,
                unprivileged_groups, privileged_groups)

# Testing: Check if the metric optimized has not become worse
# assert np.abs(metric_valid_aft[metric_name]) <= np.abs(metric_valid_bef[metric_name])

# Metrics for the test set
fav_inds = dataset_orig_test_pred.scores > best_class_thresh
dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

# display(Markdown("#### Test set"))
# display(Markdown("##### Raw predictions - No fairness constraints, only maximizing balanced accuracy"))
print("Test set: Raw predictions - No fairness constraints, only maximizing balanced accuracy")
metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
                unprivileged_groups, privileged_groups)

classified_metric_pred = ClassificationMetric(dataset_orig_test,
                                              dataset_orig_test_pred,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)
print("-->balanced accuracy:", 0.5*(classified_metric_pred.true_positive_rate() + classified_metric_pred.true_negative_rate()))
print("-->Statistical parity difference:", classified_metric_pred.statistical_parity_difference())
print("-->Disparate impact:", classified_metric_pred.disparate_impact())
print("-->Average odds difference:", classified_metric_pred.average_odds_difference())
print("Equal opportunity difference:", classified_metric_pred.equal_opportunity_difference())
print("Theil index:", classified_metric_pred.theil_index())


orig_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("-->accuracy:", orig_test.accuracy())


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

y_test = dataset_orig_test.labels.ravel()
y_pred = lmod.predict_classes(dataset_orig_test.features)
multi_orig_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
print("-->orig_metrics", multi_orig_metrics)

print("----------" + "multivariate group metric test" + "----------")
multi_group_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                   model=lmod,
                                                   thresh_arr=None,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups,
                                                   dataset_pred=dataset_orig_test_pred.scores
                                                   )
print(multi_group_metrics)

print("----------" + "causal metric test" + "----------")
multi_causal_metrics = metric_test_causal(dataset=dataset_orig_test,
                                              model=lmod,
                                              thresh_arr=None,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups,
                                              dataset_pred=dataset_orig_test_pred.scores
                                              )
print(multi_causal_metrics)


# Metrics for the transformed test set
dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)

acc_transf_test_pred = ROC.predict(acc_test_pred)
print("-->orig acc_test_pred.scores", acc_test_pred.scores)
print(acc_test_pred.labels)
print("-->acc_transf_test_pred", acc_transf_test_pred.scores)
print(acc_transf_test_pred.labels)

# display(Markdown("#### Test set"))
# display(Markdown("##### Transformed predictions - With fairness constraints"))
print("Test set: Transformed predictions - With fairness constraints")
metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
                unprivileged_groups, privileged_groups)

classified_metric_pred = ClassificationMetric(dataset_orig_test,
                                              dataset_transf_test_pred,
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)
print("-->balanced accuracy:", 0.5*(classified_metric_pred.true_positive_rate() + classified_metric_pred.true_negative_rate()))
print("-->Statistical parity difference:", classified_metric_pred.statistical_parity_difference())
print("-->Disparate impact:", classified_metric_pred.disparate_impact())
print("-->Average odds difference:", classified_metric_pred.average_odds_difference())
print("Equal opportunity difference:", classified_metric_pred.equal_opportunity_difference())
print("Theil index:", classified_metric_pred.theil_index())

transf_test = ClassificationMetric(dataset_orig_test, dataset_transf_test_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("-->accuracy:", transf_test.accuracy())

transf_test = ClassificationMetric(acc_test, acc_transf_test_pred,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("-->accuracy:", transf_test.accuracy())


print("----------" + "test on test data" + "----------")
multi_orig_trans_metrics = metric_test_new_inputs(dataset=dataset_orig_test,
                                                      model=ROC,
                                                      thresh_arr=None,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups,
                                                      dataset_pred=dataset_transf_test_pred
                                                      )
describe_metrics_new_inputs(multi_orig_trans_metrics, thresh_arr)
multi_orig_trans_metrics['acc'] = transf_test.accuracy()
print("-->multi_orig_trans_metrics", multi_orig_trans_metrics)

y_test = dataset_orig_test.labels.ravel()
y_pred = dataset_transf_test_pred.labels
multi_orig_trans_metrics['acc'] = [accuracy_score(list(y_test), list(y_pred))]
print("-->multi_orig_metrics", multi_orig_trans_metrics)

print("----------" + "multivariate group metric test" + "----------")
multi_group_trans_metrics = metric_test_multivariate(dataset=dataset_orig_test,
                                                         model=ROC,
                                                         thresh_arr=None,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups,
                                                         dataset_pred=dataset_transf_test_pred
                                                         )
print(multi_group_trans_metrics)


print("----------" + "causal metric test" + "----------")

multi_causal_trans_metrics = metric_test_causal(dataset=dataset_orig_test,
                                                    model=ROC,
                                                    thresh_arr=None,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups,
                                                    dataset_pred=dataset_transf_test_pred
                                                    )
print(multi_causal_trans_metrics)


