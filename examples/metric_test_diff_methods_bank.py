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

# pre-processing
# from metric_test import RW_metric_test
# from preprocessing_optim_demo import OP_metric_test
# from preprocessing_disparate_impace_demo import DI_metric_test
# in-processing
# from inprocessing_adversarial_debias_demo import AD_metric_test
# from inprocessing_meta_demo import META_metrics_test

# all_uni_orig_metrics = []
# all_uni_trans_metrics = []
# all_multi_orig_metrics = []
# all_multi_trans_metrics = []
# dataset_name = []
# sens_attrs = []
# processing_name = []

dataset_name = "Bank"

# basic remove sensitive features (test on test data)
basic = []

# basic remove sensitive features (test on generated samples)
basic = []


# reweighing
RW = [{'disp_imp': [1.605064376599957], 'stat_par_diff': [0.060710851300738944], 'group': [0.1141439205955335]},
    {'disp_imp': [1.623242097053304], 'stat_par_diff': [0.0647104970138678], 'group': [0.07186913381603648]}]

# disparate impact
DI = [{'disp_imp': [2.023880309049611], 'stat_par_diff': [0.07541887351657374], 'group': [0.07787114845938375], 'acc': [0.8989269481280165]},
      {'disp_imp': [2.3850058973367823], 'stat_par_diff': [0.05641775854461922], 'group': [0.06633499170812604], 'acc': [0.8925542383206035]}]

# META
META = [{'disp_imp': [1.7637044603336738], 'stat_par_diff': [0.045409454398218446], 'acc': [0.8953755329616268], 'group': [0.05839258780435252], 'causal': [0.0]},
        {'disp_imp': [1.7330693952658431], 'stat_par_diff': [0.2661504200830043], 'acc': [0.6922488247512846], 'group': [0.2636738906088752], 'causal': [0.009950248756218905]}]

# AD
AD = [{'disp_imp': [1.84184326553833], 'stat_par_diff': [0.0599149711509262], 'acc': [0.8989832732043292], 'group': [0.09237139671922281], 'causal': [0.009950248756218905]},
      {'disp_imp': [1.655617154983796], 'stat_par_diff': [0.05042640955562304], 'acc': [0.8972340658139281], 'group': [0.08900865588763679], 'causal': [0.004975124378109453]}]
AD_without_scaler = [{'disp_imp': [1.9126251929497073], 'stat_par_diff': [0.0643359651786618], 'acc': [0.8975620421996283], 'group': [0.09861932938856016], 'causal': [0.014925373134328358]},
                     {'disp_imp': [1.6052332774077651], 'stat_par_diff': [0.04518802510375544], 'acc': [0.894172952880726], 'group': [0.08900865588763679], 'causal': [0.014925373134328358]}]
# PR
PR = [{'disp_imp': [1.7751750657047607], 'stat_par_diff': [0.05560658973580322], 'acc': [0.898545971356729], 'group': [0.09237139671922281], 'causal': [0.01990049751243781]},
      {'disp_imp': [1.0047863131810313], 'stat_par_diff': [0.00017840874582447652], 'acc': [0.8849896140811195], 'group': [0.03636363636363636], 'causal': [0.05472636815920398]}]
PR_without_scaler = [{'disp_imp': [1.9368468283198412], 'stat_par_diff': [0.06340596214191718], 'acc': [0.8975620421996283], 'group': [0.09237139671922281], 'causal': [0.01990049751243781]},
                     {'disp_imp': [1.9822903489842993], 'stat_par_diff': [0.03340672132806964], 'acc': [0.8868481469334208], 'group': [0.06930693069306931], 'causal': [0.024875621890547265]}]

# GR
GR = [{'disp_imp': [1.7392799816555835], 'stat_par_diff': [0.06526976414616863], 'acc': [0.9002951787471302], 'group': [0.05696600415880847], 'causal': [0.009950248756218905]},
      {'disp_imp': [0.0], 'stat_par_diff': [-0.03626126126126126], 'acc': [0.8845523122335192], 'group': [0.0], 'causal': [0.05472636815920398]}]
GR_without_scaler = [{'disp_imp': [1.4947607625299837], 'stat_par_diff': [0.0446287579714546], 'acc': [0.8961408111949273], 'group': [0.048705491667179135], 'causal': [0.0]},
                     {'disp_imp': [1.3505960189357211], 'stat_par_diff': [0.023333586395384145], 'acc': [0.8975620421996283], 'group': [0.04912532949916127], 'causal': [0.014925373134328358]}]

# CEO
CEO = [{'disp_imp': [1.720263463773731], 'stat_par_diff': [0.0752707763943719], 'acc': [0.9001858532852302], 'group': [0.06534472239170226], 'causal': [0.01990049751243781]},
       {'disp_imp': [1000], 'stat_par_diff': [0.2397003745318352], 'acc': [0.8807259210670165], 'group': [0.23684210526315788], 'causal': [0.014925373134328358]}]
CEO_withoug_scaler = [{'disp_imp': [1.7412788987587506], 'stat_par_diff': [0.04783252353477073], 'acc': [0.8943916038045261], 'group': [0.07065527065527066], 'causal': [0.0]},
                      {'disp_imp': [133.03370786516854], 'stat_par_diff': [0.2378985727300334], 'acc': [0.8824751284574177], 'group': [0.23684210526315788], 'causal': [0.01990049751243781]}]

# EO
EO = [{'disp_imp': [1.5871700504521191], 'stat_par_diff': [0.06096517866180787], 'acc': [0.9004045042090303], 'group': [0.05178494623655913], 'causal': [0.004975124378109453]},
       {'disp_imp': [1000], 'stat_par_diff': [0.2397003745318352], 'acc': [0.8807259210670165], 'group': [0.23684210526315788], 'causal': [0.014925373134328358]}]
EO_without_scaler = [{'disp_imp': [1.34208028111141], 'stat_par_diff': [0.0305483854641158], 'acc': [0.8983273204329288], 'group': [0.020133053221288513], 'causal': [0.0]},
                      {'disp_imp': [60.81540930979134], 'stat_par_diff': [0.23575893309039378], 'acc': [0.8845523122335192], 'group': [0.23684210526315788], 'causal': [0.024875621890547265]}]

# RO
RO = [{'disp_imp': [1.5382022471910113], 'stat_par_diff': [0.04848668893612715], 'acc': [0.8994205750519296], 'group': [0.04022674158420764], 'causal': [0.009950248756218905]},
      {'disp_imp': [1.9509984654836816], 'stat_par_diff': [0.11684001417147485], 'acc': [1.0], 'group': [0.06195421288647626], 'causal': [0.8173913043478261]}]

metrics = [RW, DI, META, AD, PR, GR, CEO, EO, RO]

all_uni_orig_metrics = []
all_uni_trans_metrics = []
all_multi_orig_metrics = []
all_multi_trans_metrics = []

for metric in metrics:
    all_uni_orig_metrics.append([metric[0], metric[2]])
    all_uni_trans_metrics.append([metric[1], metric[3]])
    all_multi_orig_metrics.append(metric[4])
    all_multi_trans_metrics.append(metric[5])

print("-->all_uni_orig_metrics", all_uni_orig_metrics)
print("-->all_uni_trans_metrics", all_uni_trans_metrics)
print("-->all_multi_orig_metrics", all_multi_orig_metrics)
print("-->all_multi_trans_metrics", all_multi_trans_metrics)

# processing_names = [RW_processing_name, OP_processing_name, DI_processing_name]
processing_names = ["RW", "DI", "META", "AD", "PR", "GR", "CEO", "EO", "RO"]
# processing_names = ["RW", "OP", "AD", "META"]
# dataset_name = "Adult income"
# sens_attrs = ["race", "sex"]
# dataset_name = "German credit"
# sens_attrs = ["sex", "age"]
dataset_name = "Compas"
sens_attrs = ["sex", "race"]


for i in range(0, len(processing_names)):
    process_name = processing_names[i]
    print("-->process_name", process_name)
    uni_orig_metrics = all_uni_orig_metrics[i]
    uni_trans_metrics = all_uni_trans_metrics[i]
    multi_orig_metrics = all_multi_orig_metrics[i]
    multi_trans_metrics = all_multi_trans_metrics[i]
    print("group metric")
    try:
        percent = format((multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0]) / float(multi_orig_metrics['group'][0]), '.0%')
    except:
        percent = multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0]
    print(str(round(multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0], 3)) + "(" + str(percent)+ ")")
    try:
        percent = format((uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0]) / float(uni_orig_metrics[0]['group'][0]),'.0%')
    except:
        percent = uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0]
    print(str(round(uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0], 3)) + "(" + str(percent) + ")")
    try:
        percent = format((uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0])/float(uni_orig_metrics[1]['group'][0]), '.0%')
    except:
        percent = uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0]
    print(str(round(uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0], 3)) + "(" + str(percent) + ")")
    print("causal metric")
    try:
        try:
            percent = format((multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0]) / float(
                multi_orig_metrics['causal'][0]), '.0%')
        except:
            percent = multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0]
        print(
            str(round(multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0], 3)) + "(" + str(percent) + ")")
        try:
            percent = format((uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0]) / float(
                uni_orig_metrics[0]['causal'][0]), '.0%')
        except:
            percent = uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0]
        print(str(round(uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0], 3)) + "(" + str(
            percent) + ")")
        try:
            percent = format((uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0]) / float(
                uni_orig_metrics[1]['causal'][0]), '.0%')
        except:
            percent = uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0]
        print(str(round(uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0], 3)) + "(" + str(
            percent) + ")")
    except:
        print("no causal metric")

Plot = Plot(dataset_name=dataset_name, sens_attr=sens_attrs, processing_name=processing_names)
multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
## 1 image
# Plot.plot_abs_acc_all_metric(all_uni_orig_metrics[0], all_uni_trans_metrics[0], all_uni_orig_metrics[1],
#                              all_uni_trans_metrics[1], all_multi_orig_metrics, all_multi_trans_metrics)

# 2 images: one for group metric. one for causal metric
Plot.plot_abs_acc_multi_metric(all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics,all_multi_trans_metrics)
# 3 images: one for 'race', one for 'sex', one for 'race,sex'
# Plot.plot_abs_acc_individual_metric(all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics,all_multi_trans_metrics)
