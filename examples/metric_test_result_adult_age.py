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
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
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

from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
from plot_result import Plot

dataset_name = "Adult"

# basic remove sensitive features (test on test data)
basic = []

# basic remove sensitive features (test on generated samples)
basic = []

RW = [{'disp_imp': [0.09313286081440837], 'stat_par_diff': [-0.23079111542172448], 'average_odds_difference': [-0.21013769861989515], 'generalized_entropy_index': [0.09777294826077788], 'group': [0.23067479106284072], 'causal': [0.1217564870259481], 'acc': [0.8247517657897431]},
      {'disp_imp': [0.37262863551506664], 'stat_par_diff': [-0.15375144599188728], 'average_odds_difference': [-0.015322663916387619], 'generalized_entropy_index': [0.10155428993171076], 'group': [0.15361017898618096], 'causal':[0.11976048], 'acc':[0.8146176681338929]}]

# disparate impact
DI = [{'disp_imp': [0.26677659464235737], 'stat_par_diff': [-0.18116824078460625], 'average_odds_difference': [-0.06005261746031563], 'generalized_entropy_index': [0.09548867669138748], 'group': [0.18117409746417165], 'acc': [0.8234029484029484]},
      {'disp_imp': [0.29504810948339505], 'stat_par_diff': [-0.23292140833996916], 'average_odds_difference': [-0.08626620165048957], 'generalized_entropy_index': [0.09173456473750663], 'group': [0.23292787457139935], 'acc': [0.806949806949807]}]

# META (tau=0.7)
META = [{'disp_imp': [0.09313286081440837], 'stat_par_diff': [-0.23079111542172448], 'average_odds_difference': [-0.21013769861989515], 'generalized_entropy_index': [0.09777294826077788], 'acc': [0.8247517657897431], 'group': [0.23067479106284072], 'causal': [0.1217564870259481]},
        {'disp_imp': [0.5080885129631736], 'stat_par_diff': [-0.34184283773066415], 'average_odds_difference': [-0.13704230334696446], 'generalized_entropy_index': [0.07098162465600949], 'acc': [0.626164397584195], 'group': [0.34167550610188896], 'causal': [0.0718562874251497]}]

# AD
AD = [{'disp_imp': [0.09313286081440837], 'stat_par_diff': [-0.23079111542172448], 'average_odds_difference': [-0.21013769861989515], 'generalized_entropy_index': [0.09777294826077788], 'acc': [0.8247517657897431], 'group': [0.23067479106284072], 'causal': [0.1217564870259481]},
      {'disp_imp': [0.2654135386548534], 'stat_par_diff': [-0.10997504558978788], 'average_odds_difference': [-0.023219919013630935], 'generalized_entropy_index': [0.10821141346567648], 'acc': [0.8197358992732112], 'group': [0.10983794267679325], 'causal': [0.021956087824351298]}]

# PR
PR = []

# GR
GR = []

# CEO
CEO = []

# EO
EO = []

# RO
RO = []

metrics = [RW, DI, META, AD, PR, GR, CEO, EO, RO]

all_uni_orig_metrics = []
all_uni_trans_metrics = []
all_multi_orig_metrics = []
all_multi_trans_metrics = []

for metric in metrics:
    all_uni_orig_metrics.append([metric[0]])
    all_uni_trans_metrics.append([metric[1]])

for i in range(0, len(metrics)-3):
    all_uni_orig_metrics[i][0] = {'disp_imp': [1.6820532151883787], 'stat_par_diff': [0.046642460601733104], 'average_odds_difference': [0.04531758641131387], 'generalized_entropy_index': [0.05334962982024295], 'group': [0.046707916648018405], 'causal': [0.013972055888223553], 'acc': [0.9004718372161604]}

print("-->all_uni_orig_metrics", all_uni_orig_metrics)
print("-->all_uni_trans_metrics", all_uni_trans_metrics)

# processing_names = [RW_processing_name, OP_processing_name, DI_processing_name]
processing_names = ["RW", "DI", "META", "AD", "PR", "GR", "CEO", "EO", "RO"]
# processing_names = ["RW", "OP", "AD", "META"]
# dataset_name = "Adult income"
# sens_attrs = ["race", "sex"]
# dataset_name = "German credit"
# sens_attrs = ["sex", "age"]
# dataset_name = "Compas"
# sens_attrs = ["sex", "race"]

dataset_name = "Bank"
sens_attrs = ["age"]


for i in range(0, len(processing_names)):
    process_name = processing_names[i]
    print("-->process_name", process_name)
    uni_orig_metrics = all_uni_orig_metrics[i]
    uni_trans_metrics = all_uni_trans_metrics[i]
    print("group metric")
    try:
        percent = format((uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0]) / float(uni_orig_metrics[0]['group'][0]),'.0%')
    except:
        percent = uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0]
    print(str(round(uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0], 3)) + "(" + str(percent) + ")")
    # try:
    #     percent = format((uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0])/float(uni_orig_metrics[1]['group'][0]), '.0%')
    # except:
    #     percent = uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0]
    # print(str(round(uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0], 3)) + "(" + str(percent) + ")")
    try:
        try:
            percent = format((uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0]) / float(
                uni_orig_metrics[0]['causal'][0]), '.0%')
        except:
            percent = uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0]
        print(str(round(uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0], 3)) + "(" + str(
            percent) + ")")
        # try:
        #     percent = format((uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0]) / float(
        #         uni_orig_metrics[1]['causal'][0]), '.0%')
        # except:
        #     percent = uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0]
        # print(str(round(uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0], 3)) + "(" + str(
        #     percent) + ")")
    except:
        print("no causal metric")

Plot = Plot(dataset_name=dataset_name, sens_attr=sens_attrs, processing_name=processing_names)
multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
## 1 image
# Plot.plot_abs_acc_all_metric(all_uni_orig_metrics[0], all_uni_trans_metrics[0], all_uni_orig_metrics[1],
#                              all_uni_trans_metrics[1], all_multi_orig_metrics, all_multi_trans_metrics)

# 2 images: one for group metric. one for causal metric
# Plot.plot_abs_acc_multi_metric(all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics,all_multi_trans_metrics)
# 3 images: one for 'race', one for 'sex', one for 'race,sex'
Plot.plot_one_abs_acc_individual_metric(all_uni_orig_metrics, all_uni_trans_metrics)
