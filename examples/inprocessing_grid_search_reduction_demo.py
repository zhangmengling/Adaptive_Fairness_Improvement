import sys
sys.path.append("../")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.datasets import AdultDataset, CompasDataset, GermanDataset
from aif360.datasets.compas_dataset1 import CompasDataset_1

from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
from aif360.algorithms.inprocessing.grid_search_reduction import GridSearchReduction

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

import numpy as np

#%% md

#### Load dataset and set options

#%%

# Get the dataset and split into train and test
# dataset_orig = load_preproc_data_compas()
dataset_orig = AdultDataset()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

np.random.seed(0)
dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

#%%

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

#%% md

#### Metric for original training data

#%%

# Metric for the original dataset
metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
# display(Markdown("#### Original training dataset"))
print("--> Original training dataset ")
print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())

#%%

min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
# display(Markdown("#### Scaled dataset - Verify that the scaling does not affect the group label statistics"))
print("-->Verify that the scaling does not affect the group label statistics")
print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_train.mean_difference())
metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test,
                             unprivileged_groups=unprivileged_groups,
                             privileged_groups=privileged_groups)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_test.mean_difference())


#%% md

### Standard Logistic Regression
### Standard MLP Classifier

def metric_test_without_thresh(dataset, dataset_pred):
    metric = ClassificationMetric(dataset,
                                  dataset_pred,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
    metric_arrs = defaultdict(list)
    metric_arrs['tpr'].append(metric.true_positive_rate())
    metric_arrs['tnr'].append(metric.true_negative_rate())
    metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                   + metric.true_negative_rate()) / 2)
    metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
    metric_arrs['disp_imp'].append(metric.disparate_impact())
    metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
    metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
    metric_arrs['theil_ind'].append(metric.theil_index())
    return metric_arrs


def metric_test1(dataset, model, thresh_arr):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        all_classes = np.array([0, 1])
        pos_ind = np.where(all_classes == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        # changed coding
        pos_ind = 1
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
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    return metric_arrs


# Make a function to print out accuracy and fairness metrics. This will be used throughout the tutorial.
def describe_metrics(metrics, thresh_arr):
    best_ind = np.argmax(metrics['bal_acc'])
    print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
#     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))


nb_classes = 2
def initial_dnn(dim):
    model = Sequential()
    # kernel_initializer = 'random_uniform',bias_initializer = 'zeros',
    model.add(InputLayer(input_shape=(dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    loss = tf.keras.losses.sparse_categorical_crossentropy
    metrics = tf.keras.metrics.categorical_accuracy
    model.compile(loss=loss, metrics=[metrics], optimizer='adam')
    return model

# nb_classes = 2
# def initial_dnn(dim):
#     model = Sequential()
#     # model.add(Input(shape=x_train.shape))
#     ## need to change the input shape of each datsete
#     ## adult_income: 98; german_credit: 58
#     model.add(InputLayer(input_shape=(dim,)))
#     model.add(Dense(64))
#     model.add(LeakyReLU(alpha=0.05))
#     # model.add(BatchNormalization())
#     model.add(Dense(32))
#     model.add(LeakyReLU(alpha=0.05))
#     # model.add(BatchNormalization())
#     model.add(Dense(16))
#     model.add(LeakyReLU(alpha=0.05))
#     # model.add(BatchNormalization())
#     model.add(Dense(8))
#     model.add(LeakyReLU(alpha=0.05))
#     # model.add(BatchNormalization())
#     model.add(Dense(4))
#     model.add(LeakyReLU(alpha=0.05))
#     # model.add(BatchNormalization())
#     model.add(Flatten())
#     model.add(Dense(nb_classes, activation='softmax'))
#     loss = tf.keras.losses.sparse_categorical_crossentropy
#     # model.add(tf.keras.regularizers.l2(0.01))
#     metrics = tf.keras.metrics.categorical_accuracy
#     model.compile(loss=loss, metrics=[metrics], optimizer='adam')
#     return model

X_train = dataset_orig_train.features
y_train = dataset_orig_train.labels.ravel()

# model = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=1))
# fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
# lr_orig_panel19 = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)
#
# lmod = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 8, 4),
#                     random_state=1, verbose=True) #identity， relu
# lmod.fit(X_train, y_train)
#
# dimension = len(X_train[0])
# lmod = initial_dnn(dimension)
# lmod.fit(x=X_train,y=y_train, batch_size=128, epochs=100)

lmod = LogisticRegression(solver='lbfgs')
lmod.fit(X_train, y_train, sample_weight=dataset_orig_train.instance_weights)

X_test = dataset_orig_test.features
y_test = dataset_orig_test.labels.ravel()

y_pred = lmod.predict(X_test)

# display(Markdown("#### Accuracy"))
print("-->accuracy:")
lr_acc = accuracy_score(y_test, y_pred)
print(lr_acc)

#%%

dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
dataset_orig_test_pred.labels = y_pred



thresh_arr = np.linspace(0.01, 0.5, 50)
val_metrics = metric_test1(dataset=dataset_orig_test,
                   model=lmod,
                   thresh_arr=thresh_arr)
lr_orig_best_ind = np.argmax(val_metrics['bal_acc'])

disp_imp = np.array(val_metrics['disp_imp'])
disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)

print("-->Validating MLP model on original data")
describe_metrics(val_metrics, thresh_arr)



# positive class index
pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

metric_test = ClassificationMetric(dataset_orig_test,
                                    dataset_orig_test_pred,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)
print("-->average_odds_difference:")
lr_aod = metric_test.average_odds_difference()
print(lr_aod)

#%% md

### Exponentiated Gradient Reduction

#%% md

# Choose a base model for the randomized classifer

#%%

estimator = LogisticRegression(solver='lbfgs')

# estimator = MLPClassifier(solver='adam', activation='identity', max_iter=500, alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 8, 4),
#                     random_state=1, verbose=True) #identity， relu

# dimension = len(X_train[0])
# estimator = initial_dnn(dimension)

#%% md

# Train the randomized classifier and observe test accuracy. Other options for `constraints` include "DemographicParity,"
# "TruePositiveRateDifference", and "ErrorRateRatio."

#%%

np.random.seed(0)  # need for reproducibility
# exp_grad_red = ExponentiatedGradientReduction(estimator=estimator,
#                                               constraints="EqualizedOdds",
#                                               drop_prot_attr=False)
prot_attr_cols = [colname for colname in X_train if "sex" in colname]
grid_search_red = GridSearchReduction(prot_attr=prot_attr_cols,
                                      estimator=estimator,
                                      constraints="EqualizedOdds",
                                      grid_size=20,
                                      drop_prot_attr=False)
grid_search_red.fit(dataset_orig_train)
grid_search_red_pred = grid_search_red.predict(dataset_orig_test)

metric_test = ClassificationMetric(dataset_orig_test,
                                   grid_search_red_pred,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)

display(Markdown("#### Accuracy"))
egr_acc = metric_test.accuracy()
print(egr_acc)

#Check if accuracy is comparable
# assert abs(lr_acc-egr_acc)<0.03

display(Markdown("#### Average odds difference"))
egr_aod = metric_test.average_odds_difference()
print(egr_aod)

#Check if average odds difference has improved
# assert abs(egr_aod)<abs(lr_aod)

# accuracy of model after gradient reduction
X_test = dataset_orig_test.features
y_test = dataset_orig_test.labels.ravel()
y_pred = grid_search_red.predict(dataset_orig_test).labels
lr_acc = accuracy_score(y_test, y_pred)
print("-->accuracy after gradient reduction:", lr_acc)


thresh_arr = np.linspace(0.01, 0.5, 50)
val_metrics = metric_test_without_thresh(dataset=dataset_orig_test,
                   dataset_pred=grid_search_red_pred)

# disp_imp = np.array(val_metrics['disp_imp'])
# disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
# print("-->disp_imp_err", disp_imp_err)

print("-->Validating GridSearchReduction model on original data")
describe_metrics(val_metrics, thresh_arr)



