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
RW = [{'disp_imp': [0.7512919512919514], 'stat_par_diff': [-0.18251961639058412], 'acc': [0.714902807775378], 'group': [0.16199999999999998], 'causal': [0.11442786069651742]},
      {'disp_imp': [0.9433068362480128], 'stat_par_diff': [-0.03886224934612026], 'acc': [0.6787257019438445], 'group': [0.057999999999999996], 'causal': [0.204]},
      {'disp_imp': [0.7657525357740516], 'stat_par_diff': [-0.16350936859336007], 'acc': [0.714902807775378], 'group': [0.122], 'causal': [0.15422885572139303]},
      {'disp_imp': [0.9256829401253823], 'stat_par_diff': [-0.0519974148297655], 'acc': [0.7105831533477321], 'group': [0.05600000000000005], 'causal': [0.11442786069651742]},
      {'disp_imp': [0.8139875916525663], 'stat_par_diff': [-0.13149920255183412], 'group': [0.23599999999999993], 'causal': [0.204], 'acc': [0.714902807775378]},
      {'disp_imp': [0.9303810093756234], 'stat_par_diff': [-0.04638490164805953], 'group': [0.12], 'causal': [0.2783171521035599], 'acc': [0.7078833693304536]}]

# 2 optim -- preprocessing
OP = [{'disp_imp': [nan], 'stat_par_diff': [0.0], 'group': [0.032], 'causal': [0.014925373134328358], 'acc': [0.7133333333333334]},
      {'disp_imp': [2.3600654664484453], 'stat_par_diff': [0.0858293740962611], 'acc': [0.73], 'group': [0.032], 'causal': [0.01990049751243781]},
      {'disp_imp': [nan], 'stat_par_diff': [0.0], 'group': [0.036], 'causal': [0.024875621890547265], 'acc': [0.7133333333333334]},
      {'disp_imp': [inf], 'stat_par_diff': [0.018867924528301886], 'acc': [0.71], 'group': [0.072], 'causal': [0.029850746268656716]},
      {'disp_imp': [nan], 'stat_par_diff': [0.0], 'acc': [0.7133333333333334], 'group': [0.066], 'causal': [0.04]},
      {'disp_imp': [1.5384615384615383], 'stat_par_diff': [0.017948717948717947], 'group': [0.11], 'causal': [0.06], 'acc': [0.7166666666666667]}]


# 3 disparate impact remover -- preprocessing (MinMaxScaler)
DI_metrics = [{'disp_imp': [0.8281294720316501], 'stat_par_diff': [-0.12970552920151934], 'group': [0.17330375090074715], 'acc': [0.7212962962962963]},
{'disp_imp': [0.8204363646829337], 'stat_par_diff': [-0.12813195894978702], 'group': [0.14348063284233498], 'acc': [0.7212962962962963]},
{'disp_imp': [0.7577545900868846], 'stat_par_diff': [-0.18220997432342545], 'group': [0.1721174674194808], 'acc': [0.7256944444444444]},
{'disp_imp': [0.7223832423774883], 'stat_par_diff': [-0.19453572394523366], 'group': [0.1836341873706004], 'acc': [0.7256944444444444]},
{'disp_imp': [0.8491988243102303], 'stat_par_diff': [-0.11417803302225416], 'group': [0.2639003072130155], 'acc': [0.7206018518518519]},
{'disp_imp': [0.8377759126853437], 'stat_par_diff': [-0.1174469285201517], 'group': [0.08403361344537816], 'acc': [0.725462962962963]}]

# 4 meta --inprocessing (maxabsscaler)
META = [{'disp_imp': [0.7589346628181579], 'stat_par_diff': [-0.20023975588491727], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.2057257659467604], 'causal': [0.07984031936127745]},
{'disp_imp': [5.256521739130435], 'stat_par_diff': [0.2631720430107527], 'acc': [0.3050755939524838], 'group': [0.3014285714285714], 'causal': [0.20359281437125748]},
{'disp_imp': [0.8154854153982398], 'stat_par_diff': [-0.14127849382048963], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.11579931566801943], 'causal': [0.021956087824351298]},
{'disp_imp': [4.562906756165183], 'stat_par_diff': [0.3233661247344638], 'acc': [0.31317494600431967], 'group': [0.32883501926239667], 'causal': [0.27944111776447106]},
{'disp_imp': [0.8432766615146832], 'stat_par_diff': [-0.12129186602870812], 'acc': [0.7294816414686826], 'group': [0.30939716312056736], 'causal': [0.066]},
{'disp_imp': [2.272300469483568], 'stat_par_diff': [0.1080542264752791], 'acc': [0.3185745140388769], 'group': [0.24174496644295296], 'causal': [0.132]}]

# 5 adversarial debias --inprocessing  (maxabsscaler)
AD = [{'disp_imp': [0.7589346628181579], 'stat_par_diff': [-0.20023975588491727], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.2057257659467604], 'causal': [0.07984031936127745]},
{'disp_imp': [4.033590733590734], 'stat_par_diff': [0.3425021795989538], 'acc': [0.7154427645788337], 'group': [0.388808252553004], 'causal': [0.2375249500998004]},
{'disp_imp': [0.8154854153982398], 'stat_par_diff': [-0.14127849382048963], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.11579931566801943], 'causal': [0.021956087824351298]},
{'disp_imp': [1.5315693448849985], 'stat_par_diff': [0.14122551902060193], 'acc': [0.7208423326133909], 'group': [0.09468001988777608], 'causal': [0.05788423153692615]},
{'disp_imp': [0.8432766615146832], 'stat_par_diff': [-0.12129186602870812], 'acc': [0.7294816414686826], 'group': [0.30939716312056736], 'causal': [0.066]},
{'disp_imp': [1.9441860465116279], 'stat_par_diff': [0.1456937799043062], 'acc': [0.7181425485961123], 'group': [0.3022677564176355], 'causal': [0.096]}]

# 6 prejudice remover --inprocssing (MaxAbsScaler) 
PR = [{'disp_imp': [0.7589346628181579], 'stat_par_diff': [-0.20023975588491727], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.2057257659467604], 'causal': [0.07984031936127745]},
{'disp_imp': [5.9371478476448045], 'stat_par_diff': [0.31767171137660943], 'acc': [0.3002159827213823], 'group': [0.34550449550449547], 'causal': [0.26147704590818366]},
{'disp_imp': [0.8154854153982398], 'stat_par_diff': [-0.14127849382048963], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.11579931566801943], 'causal': [0.021956087824351298]},
{'disp_imp': [1.708083467094703], 'stat_par_diff': [0.14605682077035953], 'acc': [0.3169546436285097], 'group': [0.1485718431808838], 'causal': [0.08383233532934131]},
{'disp_imp': [0.8432766615146832], 'stat_par_diff': [-0.12129186602870812], 'acc': [0.7294816414686826], 'group': [0.30939716312056736], 'causal': [0.066]},
{'disp_imp': [1.54676710608914], 'stat_par_diff': [0.11576289207868154], 'acc': [0.30561555075593955], 'group': [0.3707103825136612], 'causal': [0.23]}]

# 7 gradient reduction -- DemographicParity (MaxAbsScaler)
GR = [{'disp_imp': [0.7589346628181579], 'stat_par_diff': [-0.20023975588491727], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.2057257659467604], 'causal': [0.07984031936127745]},
{'disp_imp': [1.8748338502436865], 'stat_par_diff': [0.14345393780877655], 'acc': [0.6830453563714903], 'group': [0.12322288303596718], 'causal': [0.005988023952095809]},
{'disp_imp': [0.8154854153982398], 'stat_par_diff': [-0.14127849382048963], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.11579931566801943], 'causal': [0.021956087824351298]},
{'disp_imp': [1.2645264847512039], 'stat_par_diff': [0.06547685266118908], 'acc': [0.6862850971922246], 'group': [0.03883973544741792], 'causal': [0.043912175648702596]},
{'disp_imp': [0.8432766615146832], 'stat_par_diff': [-0.12129186602870812], 'acc': [0.7294816414686826], 'group': [0.30939716312056736], 'causal': [0.066]},
{'disp_imp': [1.5940740740740742], 'stat_par_diff': [0.1172514619883041], 'acc': [0.6727861771058316], 'group': [0.3567313019390581], 'causal': [0.234]}]

# 8 calibrated euqodds CEO -- postprocessing  (MaxAbsScaler)
CEO = [{'disp_imp': [0.770705089947992], 'stat_par_diff': [-0.1953937808776519], 'acc': [0.7289416846652268, 0.7289416846652268], 'group': [0.17722567287784677], 'causal': [0.059880239520958084]},
{'disp_imp': [1.508108108108108], 'stat_par_diff': [0.16117407730310956], 'acc': [0.9973002159827213], 'group': [0.17625290247678016], 'causal': [0.5528942115768463]},
{'disp_imp': [0.8197130818619582], 'stat_par_diff': [-0.14280151931726082], 'acc': [0.7289416846652268, 0.7289416846652268], 'group': [0.1329751677353866], 'causal': [0.02594810379241517]},
{'disp_imp': [1.2334824630306342], 'stat_par_diff': [0.09054187922805124], 'acc': [1.0], 'group': [0.08242753623188404], 'causal': [0.5469061876247505]},
{'disp_imp': [0.8489687292082502], 'stat_par_diff': [-0.12068048910154172], 'acc': [0.7289416846652268], 'group': [0.2984026902059689], 'causal': [0.046]},
{'disp_imp': [1.23005698005698], 'stat_par_diff': [0.08585858585858586], 'acc': [1.0], 'group': [0.21969062377841064], 'causal': [0.546]}]

# 9 eqodds EO -- postprocessing  (MaxAbsScaler)
EO = [{'disp_imp': [0.7725038019123672], 'stat_par_diff': [-0.19334127290557535], 'acc': [0.7289416846652268, 0.7289416846652268], 'group': [0.17594612280510416], 'causal': [0.059880239520958084]},
{'disp_imp': [1.4379321250190844], 'stat_par_diff': [0.14558601475165273], 'acc': [0.8828293736501079], 'group': [0.15111871301775148], 'causal': [0.5469061876247505]},
{'disp_imp': [0.8197130818619582], 'stat_par_diff': [-0.14280151931726082], 'acc': [0.7289416846652268, 0.7289416846652268], 'group': [0.1329751677353866], 'causal': [0.02594810379241517]},
{'disp_imp': [1.2334824630306342], 'stat_par_diff': [0.09054187922805124], 'acc': [0.9260259179265659], 'group': [0.08242753623188404], 'causal': [0.5469061876247505]},
{'disp_imp': [0.8489687292082502], 'stat_par_diff': [-0.12068048910154172], 'acc': [0.7289416846652268], 'group': [0.2984026902059689], 'causal': [0.046]},
{'disp_imp': [1.23005698005698], 'stat_par_diff': [0.08585858585858586], 'acc': [0.9190064794816415], 'group': [0.21969062377841064], 'causal': [0.546]}]

# 10 reject_option -- postprocessing (MaxAbsScaler)
RO = [{'disp_imp': [0.770705089947992], 'stat_par_diff': [-0.1953937808776519], 'acc': [0.7289416846652268, 0.7289416846652268], 'group': [0.17722567287784677], 'causal': [0.059880239520958084]},
{'disp_imp': [1.4468029004614371], 'stat_par_diff': [0.1477332170880558], 'acc': [1.0], 'group': [0.15312036350148372], 'causal': [0.5469061876247505]},
{'disp_imp': [0.8197130818619582], 'stat_par_diff': [-0.14280151931726082], 'acc': [0.7289416846652268, 0.7289416846652268], 'group': [0.1329751677353866], 'causal': [0.02594810379241517]},
{'disp_imp': [1.2334824630306342], 'stat_par_diff': [0.09054187922805124], 'acc': [1.0], 'group': [0.08242753623188404], 'causal': [0.5469061876247505]},
{'disp_imp': [0.8489687292082502], 'stat_par_diff': [-0.12068048910154172], 'acc': [0.7289416846652268], 'group': [0.2984026902059689], 'causal': [0.046]},
{'disp_imp': [1.23005698005698], 'stat_par_diff': [0.08585858585858586], 'acc': [1.0], 'group': [0.21969062377841064], 'causal': [0.546]}]

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
