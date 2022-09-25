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


# 1 reweighing -- preprocessing
RW = [{'disp_imp': [2.2670579603815115], 'stat_par_diff': [0.17837223714108658], 'acc': [0.8033333333333333], 'group': [0.1859686233033668], 'causal': [0.02564102564102564]},
 {'disp_imp': [1.0742594910304548], 'stat_par_diff': [0.01838463127453005], 'acc': [0.74], 'group': [0.020036716496197204], 'causal': [0.0]},
 {'disp_imp': [2.5754716981132075], 'stat_par_diff': [0.2423802612481858], 'acc': [0.8033333333333333], 'group': [0.2672341319882303], 'causal': [0.02564102564102564]},
 {'disp_imp': [0.0], 'stat_par_diff': [-0.004048582995951417], 'acc': [0.71], 'group': [0.02564102564102564], 'causal': [0.0]},
 {'disp_imp': [1.8881118881118881], 'stat_par_diff': [0.14472934472934473], 'group': [0.40259169778168247], 'causal': [0.02564102564102564], 'acc': [0.8033333333333333]},
 {'disp_imp': [nan], 'stat_par_diff': [0.0], 'group': [0.0], 'causal': [0.0], 'acc': [0.7133333333333334]}]

# 2 optim -- preprocessing
OP = [{'disp_imp': [inf], 'stat_par_diff': [0.14893617021276595], 'group': [0.15053763440860216], 'causal': [0.06451612903225806], 'acc': [0.74]},
      {'disp_imp': [2.3600654664484453], 'stat_par_diff': [0.0858293740962611], 'acc': [0.73], 'group': [0.08602150537634409], 'causal': [0.0]},
      {'disp_imp': [inf], 'stat_par_diff': [0.2641509433962264], 'group': [0.2692307692307692], 'causal': [0.10738255033557047], 'acc': [0.74]},
      {'disp_imp': [inf], 'stat_par_diff': [0.018867924528301886], 'acc': [0.71], 'group': [0.0], 'causal': [0.02564102564102564]},
      {'disp_imp': [inf], 'stat_par_diff': [0.11965811965811966], 'acc': [0.74], 'group': [0.4827586206896552], 'causal': [0.3433333333333333]},
      {'disp_imp': [2.307692307692308], 'stat_par_diff': [0.009686609686609688], 'group': [0.09090909090909091], 'causal': [0.02564102564102564], 'acc': [0.7066666666666667]}]

# 3 disparate impact remover -- preprocessing (MinMaxScaler)
DI_metrics =[{'disp_imp': [1.2623891497130932], 'stat_par_diff': [0.07698194061830421], 'group': [0.07809716404256345], 'acc': [0.7085714285714285]},
             {'disp_imp': [1.194078947368421], 'stat_par_diff': [0.06095041322314049], 'group': [0.05739323029515142], 'acc': [0.7085714285714285]},
             {'disp_imp': [1.1428756532630762], 'stat_par_diff': [0.0428880735372289], 'group': [0.04487649152187567], 'acc': [0.7114285714285714]},
             {'disp_imp': [1.17086000897117], 'stat_par_diff': [0.054323164486393294], 'group': [0.05827402135231319], 'acc': [0.7114285714285714]},
             {'disp_imp': [1.0853386256512032], 'stat_par_diff': [0.023758273381294992], 'group': [0.04587716321683066], 'acc': [0.71]},
             {'disp_imp': [1.1117479642659498], 'stat_par_diff': [0.032541007194244576], 'group': [0.0], 'acc': [0.7071428571428572]}]

# 4 meta --inprocessing (maxabsscaler)
META =[{'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.11481772882244953], 'causal': [0.05]},
       {'disp_imp': [1.4106138420151626], 'stat_par_diff': [0.17341458376368518], 'acc': [0.69], 'group': [0.1777602937319696], 'causal': [0.11333333333333333]},
       {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.28861788617886175], 'causal': [0.07666666666666666]},
       {'disp_imp': [1.583623374244367], 'stat_par_diff': [0.24337330990757006], 'acc': [0.6933333333333334], 'group': [0.23921200750469046], 'causal': [0.03]},
       {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902], 'acc': [0.7766666666666666], 'group': [0.32493368700265257], 'causal': [0.12333333333333334]},
       {'disp_imp': [1.307692307692308], 'stat_par_diff': [0.13675213675213682], 'acc': [0.69], 'group': [0.3826173826173826], 'causal': [0.08333333333333333]}]

# 5 adversarial debias --inprocessing  (maxabsscaler)
AD =[{'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.11481772882244953], 'causal': [0.05]},
     {'disp_imp': [0.7304964539007092], 'stat_par_diff': [-0.00392480892377608], 'acc': [0.72], 'group': [0.003881458169420404], 'causal': [0.013333333333333334]},
     {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.28861788617886175], 'causal': [0.07666666666666666]},
     {'disp_imp': [nan], 'stat_par_diff': [0.0], 'acc': [0.7133333333333334], 'group': [0.0], 'causal': [0.0033333333333333335]},
     {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902], 'acc': [0.7766666666666666], 'group': [0.32493368700265257], 'causal': [0.12333333333333334]},
     {'disp_imp': [0.8653846153846153], 'stat_par_diff': [-0.003988603988603991], 'acc': [0.72], 'group': [0.045454545454545456], 'causal': [0.02666666666666667]}]

# 6 prejudice remover --inprocssing (MaxAbsScaler) 
PR = [{'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7733333333333333, 0.7733333333333333], 'group': [0.11481772882244953], 'causal': [0.05]},
{'disp_imp': [1.3064648117839606], 'stat_par_diff': [0.1547200991530675], 'acc': [0.6366666666666667], 'group': [0.16422764227642273], 'causal': [0.23333333333333334]},
{'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7733333333333333, 0.7733333333333333], 'group': [0.28861788617886175], 'causal': [0.07666666666666666]},
{'disp_imp': [2.402256370355962], 'stat_par_diff': [0.5506836758078069], 'acc': [0.68], 'group': [0.5479987492182614], 'causal': [0.4]},
{'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902], 'acc': [0.7766666666666666], 'group': [0.32493368700265257], 'causal': [0.12333333333333334]},
{'disp_imp': [1.3432835820895521], 'stat_par_diff': [0.17037037037037034], 'acc': [0.6466666666666666], 'group': [0.39352027283061763], 'causal': [0.23666666666666666]}]


# 7 gradient reduction -- DemographicParity (MaxAbsScaler)
GR =[{'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.11481772882244953], 'causal': [0.05]}, 
     {'disp_imp': [1.0719241443108234], 'stat_par_diff': [0.032121462507746334], 'acc': [0.7], 'group': [0.03996852871754525], 'causal': [0.056666666666666664]}, 
     {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.28861788617886175], 'causal': [0.07666666666666666]}, 
     {'disp_imp': [1.9460916442048517], 'stat_par_diff': [0.3485600794438927], 'acc': [0.6366666666666667], 'group': [0.3416197623514697], 'causal': [0.07333333333333333]},
     {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902], 'acc': [0.7766666666666666], 'group': [0.32493368700265257], 'causal': [0.12333333333333334]},
     {'disp_imp': [1.342657342657343], 'stat_par_diff': [0.13960113960113968], 'acc': [0.68], 'group': [0.42657342657342656], 'causal': [0.06333333333333334]}]


# 8 calibrated euqodds CEO -- postprocessing  (MaxAbsScaler)
CEO =[{'disp_imp': [1.3456513624486748], 'stat_par_diff': [0.0956413964057013], 'acc': [0.76, 0.76], 'group': [0.09829530553370047], 'causal': [0.0]}, {'disp_imp': [2.2599734042553195], 'stat_par_diff': [0.19572402396199134], 'acc': [0.93], 'group': [0.20612076095947066], 'causal': [0.2153846153846154]}, {'disp_imp': [2.1452530697813716], 'stat_par_diff': [0.2921090825758154], 'acc': [0.76, 0.76], 'group': [0.2823639774859287], 'causal': [0.05813953488372093]}, {'disp_imp': [5.065627563576702], 'stat_par_diff': [0.3785807043006646], 'acc': [0.8733333333333333], 'group': [0.4288211788211788], 'causal': [0.14583333333333334]}, {'disp_imp': [1.4574898785425099], 'stat_par_diff': [0.12877492877492874], 'acc': [0.76], 'group': [0.3099658961727927], 'causal': [0.10738255033557047]}, {'disp_imp': [1.5889029003783102], 'stat_par_diff': [0.13304843304843306], 'acc': [0.97], 'group': [0.36491095111784766], 'causal': [0.2534246575342466]}]

# 9 eqodds EO -- postprocessing  (MaxAbsScaler)
EO =

# 10 reject_option -- postprocessing (MaxAbsScaler)
RO =

metrics = [RW, OP, DI, META, AD, PR, GR, CEO, EO, RO]

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
processing_names = ["RW", "OP", "DI", "META", "AD", "PR", "GR", "CEO", "EO", "RO"]
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
Plot.plot_abs_acc_individual_metric(all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics,all_multi_trans_metrics)
