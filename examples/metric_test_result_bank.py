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

dataset_name = "Bank"

# basic remove sensitive features (test on test data)
basic = []

# basic remove sensitive features (test on generated samples)
basic = []

RW = [{'disp_imp': [1.6820532151883787], 'stat_par_diff': [0.046642460601733104], 'average_odds_difference': [0.04531758641131387], 'generalized_entropy_index': [0.05334962982024295],'group': [0.046707916648018405], 'causal': [0.013972055888223553], 'acc': [0.9004718372161604]},
      {'disp_imp': [1.0651741454574986], 'stat_par_diff': [0.00594079219914756], 'average_odds_difference': [-0.06342541561122993], 'generalized_entropy_index': [0.053646241843423086], 'acc': [0.8974491300501327], 'group': [0.005993244352059515], 'causal':[0.02594810379241517]}]

# disparate impact
DI = [{'disp_imp': [1.6314037793797094], 'stat_par_diff': [0.046896804591847946], 'average_odds_difference': [0.02062504965408254], 'generalized_entropy_index': [0.05419974174809959], 'group': [0.046927303038167334], 'acc': [0.8981262046955477]},
      {'disp_imp': [1.4831653519691979], 'stat_par_diff': [0.04508691719994086], 'average_odds_difference': [0.008788804360964575], 'generalized_entropy_index': [0.05287903135253897], 'group': [0.04512144996087619], 'acc': [0.8977470218346131]}]
# DI = [{'disp_imp': [2.0450418638841845], 'stat_par_diff': [0.09546540123439178], 'average_odds_difference': [0.08181964629750674], 'generalized_entropy_index': [0.05454831665455027], 'accuracy': [0.893828798938288], 'group': [0.09551325004190965]},
#       {'disp_imp': [2.6766219508863265], 'stat_par_diff': [0.05277067409991974], 'average_odds_difference': [0.0596879206732539], 'generalized_entropy_index': [0.05897000129009458], 'accuracy': [0.8937340032230543], 'group': [0.052792598881073904]}]


# META (tau=0.0)
META = [{'disp_imp': [1.6820532151883787], 'stat_par_diff': [0.046642460601733104], 'average_odds_difference': [0.04531758641131387], 'generalized_entropy_index': [0.05334962982024295], 'acc': [0.9004718372161604], 'group': [0.046707916648018405], 'causal': [0.013972055888223553]},
        {'disp_imp': [1.236892890272992], 'stat_par_diff': [0.00888323552710308], 'average_odds_difference': [-0.017636620286553592], 'generalized_entropy_index': [0.05758762138071119], 'acc': [0.8960483633146564], 'group': [0.008908798338397375], 'causal': [0.003992015968063872]}]

# META (tau=0.1)
# META =[{'disp_imp': [1.6820532151883787], 'stat_par_diff': [0.046642460601733104], 'average_odds_difference': [0.04531758641131387], 'generalized_entropy_index': [0.05334962982024295], 'acc': [0.9004718372161604], 'group': [0.046707916648018405], 'causal': [0.013972055888223553]},
#        {'disp_imp': [2.07816402910313], 'stat_par_diff': [0.11325821181254106], 'average_odds_difference': [0.08499426939446834], 'generalized_entropy_index': [0.056073551410124836], 'acc': [0.8875700383367738], 'group': [0.1133845083403368], 'causal': [0.09780439121756487]}]
META = [{'disp_imp': [1.6820532151883787], 'stat_par_diff': [0.046642460601733104], 'average_odds_difference': [0.04531758641131387], 'generalized_entropy_index': [0.05334962982024295], 'accuracy': [0.9004718372161604], 'acc': [0.9004718372161604], 'group': [0.046707916648018405], 'causal': [0.020347979946918313]},
        {'disp_imp': [1.236892890272992], 'stat_par_diff': [0.00888323552710308], 'average_odds_difference': [-0.017636620286553592], 'generalized_entropy_index': [0.05758762138071119], 'accuracy': [0.8960483633146564], 'acc': [0.8960483633146564], 'group': [0.008908798338397375], 'causal': [0.005160719551754645]}]

# AD
AD = [{'disp_imp': [1.6820532151883787], 'stat_par_diff': [0.046642460601733104], 'average_odds_difference': [0.04531758641131387], 'generalized_entropy_index': [0.05334962982024295], 'acc': [0.9004718372161604], 'group': [0.046707916648018405], 'causal': [0.013972055888223553]},
      {'disp_imp': [1.1758219158965681], 'stat_par_diff': [0.012391567187152452], 'average_odds_difference': [-0.037395438345984376], 'generalized_entropy_index': [0.05262609957128131], 'acc': [0.9020200530816869], 'group': [0.012436948114700974], 'causal': [0.023952095808383235]}]

# PR
PR = [{'disp_imp': [1.6820532151883787], 'stat_par_diff': [0.046642460601733104], 'average_odds_difference': [0.04531758641131387], 'generalized_entropy_index': [0.05334962982024295], 'acc': [0.9004718372161604], 'group': [0.046707916648018405], 'causal': [0.013972055888223553]},
      {'disp_imp': [1.6364331457601824], 'stat_par_diff': [0.03222918332509503], 'average_odds_difference': [0.023411809611258237], 'generalized_entropy_index': [0.0585252469285177], 'acc': [0.893246829843704], 'group': [0.03227622486005507], 'causal': [0.041916167664670656]}]

# GR
GR = [{'disp_imp': [1.6820532151883787], 'stat_par_diff': [0.046642460601733104], 'average_odds_difference': [0.04531758641131387], 'generalized_entropy_index': [0.05334962982024295], 'acc': [0.9004718372161604], 'group': [0.046707916648018405], 'causal': [0.013972055888223553]},
      {'disp_imp': [0.7126374469839515], 'stat_par_diff': [-0.01695744537342548], 'average_odds_difference': [-0.0686587531816915], 'generalized_entropy_index': [0.057803743526211676], 'acc': [0.8943526983190799], 'group': [0.01693636214450174], 'causal': [0.03592814371257485]}]

# CEO
CEO = [{'disp_imp': [2.121609593937661], 'stat_par_diff': [0.07320660083438982], 'average_odds_difference': [0.07584579384019857], 'generalized_entropy_index': [0.05439047843187117], 'acc': [0.8988167643481145], 'group':[0.07332722208922805], 'causal':[0.031936127744510975]},
       {'disp_imp': [1.8368297826708608], 'stat_par_diff': [0.054619240239594274], 'average_odds_difference': [0.040536267951785054], 'generalized_entropy_index': [0.05477372654710735], 'acc':[0.897670303745208], 'group':[0.054722570926437356], 'causal':[0.05588822355289421]}]

# EO
EO = [{'disp_imp': [2.121609593937661], 'stat_par_diff': [0.07320660083438982], 'average_odds_difference': [0.07584579384019857], 'generalized_entropy_index': [0.05439047843187117], 'acc': [0.8988167643481145], 'group':[0.07332722208922805], 'causal':[0.031936127744510975]},
      {'disp_imp': [2.121609593937661], 'stat_par_diff': [0.07320660083438982], 'average_odds_difference': [0.07584579384019857], 'generalized_entropy_index': [0.05439047843187117], 'acc':[0.8820406959598939], 'group':[0.07332722208922805], 'causal':[0.05788423153692615]}]

# RO
RO = [{'disp_imp': [2.121609593937661], 'stat_par_diff': [0.07320660083438982], 'average_odds_difference': [0.07584579384019857], 'generalized_entropy_index': [0.05439047843187117], 'acc': [0.8988167643481145], 'group':[0.07332722208922805], 'causal': [0.031936127744510975]},
      {'disp_imp': [2.121609593937661], 'stat_par_diff': [0.07320660083438982], 'average_odds_difference': [0.07584579384019857], 'generalized_entropy_index': [0.05439047843187117], 'acc': [0.8987061815769103], 'group':[0.07072967086863355], 'causal':[0.05189620758483034]}]

metrics = [RW, DI, META, AD, PR, GR, CEO, EO, RO]

all_uni_orig_metrics = []
all_uni_trans_metrics = []
all_multi_orig_metrics = []
all_multi_trans_metrics = []

for metric in metrics:
    all_uni_orig_metrics.append([metric[0]])
    all_uni_trans_metrics.append([metric[1]])

for i in range(0, len(metrics)):
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
    print(str(round(uni_trans_metrics[0]['acc'][0] - uni_orig_metrics[0]['acc'][0], 3)))

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
        percent = ''
        print(str(round(uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0], 3)) + "" + str(
            percent) + "")
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
