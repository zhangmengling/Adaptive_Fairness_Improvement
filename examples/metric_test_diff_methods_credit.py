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


"""
(RW_uni_orig_metrics,
 RW_uni_trans_metrics,
 RW_multi_orig_metrics,
 RW_multi_trans_metrics,
 RW_dataset_name,
 RW_sens_attrs,
 RW_processing_name) = RW_metric_test(dataset_name)

(OP_uni_orig_metrics,
 OP_uni_trans_metrics,
 OP_multi_orig_metrics,
 OP_multi_trans_metrics,
 OP_dataset_name,
 OP_sens_attrs,
 OP_processing_name) = OP_metric_test(dataset_name)

(DI_uni_orig_metrics,
 DI_uni_trans_metrics,
 DI_multi_orig_metrics,
 DI_multi_trans_metrics,
 DI_dataset_name,
 DI_sens_attrs,
 DI_processing_name) = DI_metric_test(dataset_name)

(AD_uni_orig_metrics,
 AD_uni_trans_metrics,
 AD_multi_orig_metrics,
 AD_multi_trans_metrics,
 AD_dataset_name,
 AD_sens_attrs,
 AD_processing_name) = AD_metric_test(dataset_name)

 (META_uni_orig_metrics,
 META_uni_trans_metrics,
 META_multi_orig_metrics,
 META_multi_trans_metrics,
 META_dataset_name,
 META_sens_attrs,
 META_processing_name) = META_metric_test(dataset_name)

all_uni_orig_metrics = []
all_uni_orig_metrics.append(RW_uni_orig_metrics)
all_uni_orig_metrics.append(OP_uni_orig_metrics)
all_uni_orig_metrics.append(DI_uni_orig_metrics)
all_uni_orig_metrics.append(AD_uni_orig_metrics)
all_uni_trans_metrics = []
all_uni_trans_metrics.append(RW_uni_trans_metrics)
all_uni_trans_metrics.append(OP_uni_trans_metrics)
all_uni_trans_metrics.append(DI_uni_trans_metrics)
all_uni_trans_metrics.append(AD_uni_trans_metrics)
all_multi_orig_metrics = []
all_multi_orig_metrics.append(RW_multi_orig_metrics)
all_multi_orig_metrics.append(OP_multi_orig_metrics)
all_multi_orig_metrics.append(DI_multi_orig_metrics)
all_multi_orig_metrics.append(AD_multi_orig_metrics)
all_multi_trans_metrics = []
all_multi_trans_metrics.append(RW_multi_trans_metrics)
all_multi_trans_metrics.append(OP_multi_trans_metrics)
all_multi_trans_metrics.append(DI_multi_trans_metrics)
all_multi_trans_metrics.append(AD_multi_trans_metrics)
"""

# reweighing -- preprocessing
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.2670579603815115], 'stat_par_diff': [0.17837223714108658], 'acc': [0.8033333333333333], 'group': [0.07599999999999998], 'causal': [0.03992015968063872]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.0742594910304548], 'stat_par_diff': [0.01838463127453005], 'acc': [0.74], 'group': [0.027999999999999997], 'causal': [0.003992015968063872]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.5754716981132075], 'stat_par_diff': [0.2423802612481858], 'acc': [0.8033333333333333], 'group': [0.16799999999999998], 'causal': [0.029940119760479042]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.0], 'stat_par_diff': [-0.004048582995951417], 'acc': [0.71], 'group': [0.0], 'causal': [0.0]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.8881118881118881], 'stat_par_diff': [0.14472934472934473], 'group': [0.18], 'causal': [0.088], 'acc': [0.8033333333333333]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [nan], 'stat_par_diff': [0.0], 'group': [0.0], 'causal': [0.0], 'acc': [0.7133333333333334]})

# optim -- preprocessing
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [inf], 'stat_par_diff': [0.14893617021276595], 'group': [0.116], 'causal': [0.08582834331337326], 'acc': [0.74]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [2.3600654664484453], 'stat_par_diff': [0.0858293740962611], 'acc': [0.73], 'group': [0.032], 'causal': [0.01996007984031936]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [inf], 'stat_par_diff': [0.2641509433962264], 'group': [0.128], 'causal': [0.08782435129740519], 'acc': [0.74]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [inf], 'stat_par_diff': [0.018867924528301886], 'acc': [0.71], 'group': [0.072], 'causal': [0.05189620758483034]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [inf], 'stat_par_diff': [0.11965811965811966], 'acc': [0.74], 'group': [0.2], 'causal': [0.172]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [2.307692307692308], 'stat_par_diff': [0.009686609686609688], 'group': [0.138], 'causal': [0.1], 'acc': [0.7066666666666667]})

# optim2 -- preprocessing
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [inf], 'stat_par_diff': [0.14893617021276595], 'group': [0.116], 'causal': [0.08582834331337326], 'acc': [0.74]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [2.556737588652482], 'stat_par_diff': [0.09068374302829993], 'acc': [0.7333333333333333], 'group': [0.082], 'causal': [0.03792415169660679]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [inf], 'stat_par_diff': [0.2641509433962264], 'group': [0.128], 'causal': [0.08782435129740519], 'acc': [0.74]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [inf], 'stat_par_diff': [0.2830188679245283], 'acc': [0.7366666666666667], 'group': [0.128], 'causal': [0.10179640718562874]})
# -->multi_orig_metrics [defaultdict(<class 'list'>, {'disp_imp': [inf], 'stat_par_diff': [0.11965811965811966], 'acc': [0.74]}), defaultdict(<class 'list'>, {'group': [0.2]}), defaultdict(<class 'list'>, {'causal': [0.172]})]
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [inf], 'stat_par_diff': [0.11965811965811966], 'acc': [0.74], 'group': [0.2], 'causal': [0.172]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [2.307692307692308], 'stat_par_diff': [0.009686609686609688], 'group': [0.138], 'causal': [0.1], 'acc': [0.7066666666666667]})

# disparate impact remover -- preprocessing (MinMaxScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.2623891497130932], 'stat_par_diff': [0.07698194061830421], 'group': [0.04200000000000004], 'acc': [0.7085714285714285]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.194078947368421], 'stat_par_diff': [0.06095041322314049], 'group': [0.033999999999999975], 'acc': [0.7085714285714285]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.1428756532630762], 'stat_par_diff': [0.0428880735372289], 'group': [0.10799999999999998], 'acc': [0.7114285714285714]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.17086000897117], 'stat_par_diff': [0.054323164486393294], 'group': [0.10999999999999999], 'acc': [0.7114285714285714]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.1117479642659498], 'stat_par_diff': [0.032541007194244576], 'group': [0.11200000000000002], 'acc': [0.7128571428571429]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.052113973845554], 'stat_par_diff': [0.01692661870503598], 'group': [0.11599999999999999], 'acc': [0.7028571428571428]})

# meta --inprocessing (maxabsscaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.17999999999999994], 'causal': [0.10179640718562874]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.4106138420151626], 'stat_par_diff': [0.17341458376368518], 'acc': [0.69], 'group': [0.1459999999999999], 'causal': [0.1536926147704591]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.19599999999999995], 'causal': [0.09780439121756487]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.583623374244367], 'stat_par_diff': [0.24337330990757006], 'acc': [0.6933333333333334], 'group': [0.050000000000000044], 'causal': [0.06187624750499002]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902], 'acc': [0.7766666666666666], 'group': [0.28600000000000003], 'causal': [0.182]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.307692307692308], 'stat_par_diff': [0.13675213675213682], 'acc': [0.69], 'group': [0.20199999999999996], 'causal': [0.098]})

# meta --inprocessing (maxabsscaler) metaclassifier-->metaclassifier
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.4779811974270165], 'stat_par_diff': [0.09977277422020245], 'acc': [0.76, 0.76], 'group': [0.126], 'causal': [0.015968063872255488]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.4106138420151626], 'stat_par_diff': [0.17341458376368518], 'acc': [0.69], 'group': [0.1459999999999999], 'causal': [0.1536926147704591]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.1875240662302655], 'stat_par_diff': [0.23558169734932397], 'acc': [0.76, 0.76], 'group': [0.17999999999999994], 'causal': [0.021956087824351298]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.583623374244367], 'stat_par_diff': [0.24337330990757006], 'acc': [0.6933333333333334], 'group': [0.050000000000000044], 'causal': [0.06187624750499002]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.5119363395225465], 'stat_par_diff': [0.10997150997150998], 'acc': [0.76], 'group': [0.22399999999999998], 'causal': [0.056]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.307692307692308], 'stat_par_diff': [0.13675213675213682], 'acc': [0.69], 'group': [0.20199999999999996], 'causal': [0.098]})

# adversarial debias --inprocessing  (maxabsscaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.18008971011581976], 'causal': [0.0962962962962963]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.7304964539007092], 'stat_par_diff': [-0.00392480892377608], 'acc': [0.72], 'group': [0.00841248303934871], 'causal': [0.0]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.1872295227310633], 'causal': [0.14285714285714285]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [nan], 'stat_par_diff': [0.0], 'acc': [0.7133333333333334], 'group': [0.02], 'causal': [0.0196078431372549]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902], 'acc': [0.7766666666666666], 'group': [0.31122967479674796], 'causal': [0.20161290322580644]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.8653846153846153], 'stat_par_diff': [-0.003988603988603991], 'acc': [0.72], 'group': [0.12583333333333335], 'causal': [0.03636363636363636]})

# adversarial debias --inprocessing (maxabsscaler) adclassifier-->adclassifier
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.524514338575393], 'stat_par_diff': [0.11712456104110719], 'acc': [0.7866666666666666, 0.7866666666666666], 'group': [0.15399999999999997], 'causal': [0.05788423153692615]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.2174940898345152], 'stat_par_diff': [0.009502168973352612], 'acc': [0.7066666666666667], 'group': [0.036], 'causal': [0.013972055888223553]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.3301886792452833], 'stat_par_diff': [0.2800397219463754], 'acc': [0.7866666666666666, 0.7866666666666666], 'group': [0.25400000000000006], 'causal': [0.0998003992015968]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.0421940928270044], 'stat_par_diff': [0.04048582995951422], 'acc': [0.32], 'group': [0.026000000000000023], 'causal': [0.007984031936127744]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.5750915750915753], 'stat_par_diff': [0.1341880341880342], 'acc': [0.7866666666666666], 'group': [0.31999999999999995], 'causal': [0.182]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.7307692307692306], 'stat_par_diff': [0.010826210826210825], 'acc': [0.72], 'group': [0.08199999999999999], 'causal': [0.054]})

# prejudice remover --inprocssing (StandardScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.9479905437352245], 'stat_par_diff': [0.16566825036149554], 'acc': [0.7666666666666667, 0.7666666666666667], 'group': [0.05400000000000002], 'causal': [0.07984031936127745]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.910529187124932], 'stat_par_diff': [0.17238173931005993], 'acc': [0.7433333333333333], 'group': [0.10000000000000003], 'causal': [0.17365269461077845]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.3819706498951785], 'stat_par_diff': [0.25177602933312965], 'acc': [0.7666666666666667, 0.7666666666666667], 'group': [0.13], 'causal': [0.07385229540918163]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [3.6012006861063464], 'stat_par_diff': [0.46337178214040176], 'acc': [0.76], 'group': [0.03200000000000003], 'causal': [0.249500998003992]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.741654571843251], 'stat_par_diff': [0.14558404558404558], 'acc': [0.7666666666666667], 'group': [0.15000000000000002], 'causal': [0.17]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.5843857634902412], 'stat_par_diff': [0.145014245014245], 'acc': [0.7466666666666667], 'group': [0.244], 'causal': [0.334]})
# prejudice remover --inprocssing (MaxAbsScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.17799999999999994], 'causal': [0.0998003992015968]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.3191489361702127], 'stat_par_diff': [0.15957446808510634], 'acc': [0.6366666666666667], 'group': [0.08399999999999996], 'causal': [0.2055888223552894]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.19399999999999995], 'causal': [0.09580838323353294]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [2.402256370355962], 'stat_par_diff': [0.5506836758078069], 'acc': [0.6833333333333333], 'group': [0.36], 'causal': [0.3532934131736527]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902], 'acc': [0.7766666666666666], 'group': [0.28600000000000003], 'causal': [0.182]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.3432835820895521], 'stat_par_diff': [0.17037037037037034], 'acc': [0.6466666666666666], 'group': [0.20400000000000007], 'causal': [0.266]})


# exponentiated gradient reduction
# gradient reduction -- DemographicParity (MaxAbsScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.17999999999999994], 'causal': [0.10179640718562874]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.0719241443108234], 'stat_par_diff': [0.032121462507746334], 'acc': [0.7066666666666667], 'group': [0.05799999999999994], 'causal': [0.09780439121756487]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.19599999999999995], 'causal': [0.09780439121756487]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.9460916442048517], 'stat_par_diff': [0.3485600794438927], 'acc': [0.6666666666666666], 'group': [0.21599999999999997], 'causal': [0.13572854291417166]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902], 'acc': [0.7766666666666666], 'group': [0.28600000000000003], 'causal': [0.182]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.342657342657343], 'stat_par_diff': [0.13960113960113968], 'acc': [0.68], 'group': [0.196], 'causal': [0.09]})

# gradient reduction -- EqualizedOdds  (MaxAbsScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.17999999999999994], 'causal': [0.10179640718562874]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.095744680851064], 'stat_par_diff': [0.041830200371824056], 'acc': [0.6933333333333334], 'group': [0.015999999999999903], 'causal': [0.05189620758483034]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.19599999999999995], 'causal': [0.09780439121756487]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.9460916442048517], 'stat_par_diff': [0.3485600794438927], 'acc': [0.6766666666666666], 'group': [0.21599999999999997], 'causal': [0.13572854291417166]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902], 'acc': [0.7766666666666666], 'group': [0.28600000000000003], 'causal': [0.182]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.047570850202429], 'stat_par_diff': [0.02678062678062676], 'acc': [0.6166666666666667], 'group': [0.19599999999999995], 'causal': [0.19]})

# calibrated euqodds CEO -- postprocessing  (MaxAbsScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.3456513624486748], 'stat_par_diff': [0.0956413964057013], 'acc': [0.76, 0.76], 'group': [0.09829530553370047], 'causal': [0.0196078431372549]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.0750702529104779], 'stat_par_diff': [0.019314191282792825], 'acc': [0.9766666666666667], 'group': [0.025911355887752396], 'causal': [0.26013513513513514]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.1452530697813716], 'stat_par_diff': [0.2921090825758154], 'acc': [0.76, 0.76], 'group': [0.2823639774859287], 'causal': [0.05813953488372093]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.4515929477265697], 'stat_par_diff': [0.11152700328469939], 'acc': [0.98], 'group': [0.12148217636022512], 'causal': [0.26174496644295303]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.4574898785425099], 'stat_par_diff': [0.12877492877492874], 'acc': [0.76], 'group': [0.05982905982905981], 'causal': [0.10738255033557047]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.2587412587412588], 'stat_par_diff': [0.06324786324786327], 'acc': [0.98], 'group': [0.014041514041514047], 'causal': [0.26174496644295303]})

# eqodds EO -- postprocessing  (MaxAbsScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.325023084025854], 'stat_par_diff': [0.09037227214377402], 'acc': [0.7566666666666667, 0.7566666666666667], 'group': [0.09292866082603252], 'causal': [0.0196078431372549]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.3435948361469712], 'stat_par_diff': [0.08883183568677794], 'acc': [0.9633333333333334], 'group': [0.09616186900292034], 'causal': [0.2866666666666667]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.1452530697813716], 'stat_par_diff': [0.2921090825758154], 'acc': [0.7566666666666667, 0.7566666666666667], 'group': [0.2823639774859287], 'causal': [0.05813953488372093]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.909990720692855], 'stat_par_diff': [0.22473455045451074], 'acc': [0.9033333333333333], 'group': [0.23686679174484054], 'causal': [0.2866666666666667]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.4574898785425099], 'stat_par_diff': [0.12877492877492874], 'acc': [0.76], 'group': [0.05982905982905981], 'causal': [0.10738255033557047]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.3846153846153848], 'stat_par_diff': [0.09971509971509973], 'acc': [0.9533333333333334], 'group': [0.03357753357753357], 'causal': [0.2866666666666667]})

# reject_option -- postprocessing (MaxAbsScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.3456513624486748], 'stat_par_diff': [0.0956413964057013], 'acc': [0.76, 0.76], 'group': [0.09829530553370047], 'causal': [0.0196078431372549]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.3645122440786832], 'stat_par_diff': [0.0937822763891758], 'acc': [1.0], 'group': [0.10118017309205352], 'causal': [0.2866666666666667]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [2.1452530697813716], 'stat_par_diff': [0.2921090825758154], 'acc': [0.76, 0.76], 'group': [0.2823639774859287], 'causal': [0.05813953488372093]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.909990720692855], 'stat_par_diff': [0.22473455045451074], 'acc': [1.0], 'group': [0.23686679174484054], 'causal': [0.2866666666666667]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [1.4574898785425099], 'stat_par_diff': [0.12877492877492874], 'acc': [0.76], 'group': [0.05982905982905981], 'causal': [0.10738255033557047]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.3846153846153848], 'stat_par_diff': [0.09971509971509973], 'acc': [1.0], 'group': [0.03357753357753357], 'causal': [0.2866666666666667]})

# basic remove sensitive features (test on generated samples)
basic = [{'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666], 'group': [0.17999999999999994]}, 
         {'disp_imp': [1.195357833655706], 'stat_par_diff': [0.07302210287130756], 'acc': [0.71], 'group': [0.09400000000000003]},
         {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666], 'group': [0.19599999999999995]}, 
         {'disp_imp': [1.2890405459654757], 'stat_par_diff': [0.10999923611641588], 'acc': [0.6866666666666666], 'group': [0.10399999999999998]}, 
         {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902], 'acc': [0.7766666666666666], 'group': [0.28600000000000003]}, 
         {'disp_imp': [1.1096563011456628], 'stat_par_diff': [0.09544159544159547], 'acc': [0.4066666666666667], 'group': [0.08999999999999997]}]

# basic remove sensitive features (test on test data)
basic = [{'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666], 'group': [0.18008971011581976]},
         {'disp_imp': [1.195357833655706], 'stat_par_diff': [0.07302210287130756], 'acc': [0.71], 'group': [0.07600314712824546]},
         {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666], 'group': [0.1872295227310633]},
         {'disp_imp': [1.2890405459654757], 'stat_par_diff': [0.10999923611641588], 'acc': [0.6866666666666666], 'group': [0.09865540963101938]},
         {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902], 'acc': [0.7766666666666666], 'group': [0.31122967479674796]},
         {'disp_imp': [1.1096563011456628], 'stat_par_diff': [0.09544159544159547], 'acc': [0.4066666666666667], 'group': [0.17582417582417587]}]

# i: method index, j: sensitive attribute index e.g. (race, sex) for Adult Income
all_uni_orig_metrics = [[{'disp_imp': [2.2670579603815115], 'stat_par_diff': [0.17837223714108658], 'acc': [0.8033333333333333], 'group': [0.07599999999999998], 'causal': [0.03992015968063872]},
                         {'disp_imp': [2.5754716981132075], 'stat_par_diff': [0.2423802612481858],'acc': [0.8033333333333333], 'group': [0.16799999999999998],'causal': [0.029940119760479042]}],
                        [{'disp_imp': ["inf"], 'stat_par_diff': [0.14893617021276595], 'group': [0.116], 'causal': [0.08582834331337326], 'acc': [0.74]},
                         {'disp_imp': ["inf"], 'stat_par_diff': [0.2641509433962264], 'group': [0.128],'causal': [0.08782435129740519], 'acc': [0.74]}],
                        [{'disp_imp': [1.2623891497130932], 'stat_par_diff': [0.07698194061830421], 'group': [0.04200000000000004], 'acc': [0.7085714285714285]},
                         {'disp_imp': [1.1428756532630762], 'stat_par_diff': [0.0428880735372289],'group': [0.10799999999999998], 'acc': [0.7114285714285714]}],
                        [{'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.17999999999999994], 'causal': [0.10179640718562874]},
                         {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.19599999999999995], 'causal': [0.09780439121756487]}],
                        [{'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.18008971011581976], 'causal': [0.0962962962962963]},
                         {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.1872295227310633], 'causal': [0.14285714285714285]}],
                        [{'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.17799999999999994], 'causal': [0.0998003992015968]},
                         {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.19399999999999995], 'causal': [0.09580838323353294]}],
                        [{'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.17999999999999994], 'causal': [0.10179640718562874]},
                         {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666, 0.7766666666666666], 'group': [0.19599999999999995], 'causal': [0.09780439121756487]}],
                        [{'disp_imp': [1.3456513624486748], 'stat_par_diff': [0.0956413964057013], 'acc': [0.76, 0.76], 'group': [0.09829530553370047], 'causal': [0.0196078431372549]},
                         {'disp_imp': [2.1452530697813716], 'stat_par_diff': [0.2921090825758154], 'acc': [0.76, 0.76], 'group': [0.2823639774859287], 'causal': [0.05813953488372093]}],
                        [{'disp_imp': [1.325023084025854], 'stat_par_diff': [0.09037227214377402], 'acc': [0.7566666666666667, 0.7566666666666667], 'group': [0.09292866082603252], 'causal': [0.0196078431372549]},
                         {'disp_imp': [2.1452530697813716], 'stat_par_diff': [0.2921090825758154], 'acc': [0.7566666666666667, 0.7566666666666667], 'group': [0.2823639774859287], 'causal': [0.05813953488372093]}],
                        [{'disp_imp': [1.3456513624486748], 'stat_par_diff': [0.0956413964057013], 'acc': [0.76, 0.76], 'group': [0.09829530553370047], 'causal': [0.0196078431372549]},
                         {'disp_imp': [2.1452530697813716], 'stat_par_diff': [0.2921090825758154], 'acc': [0.76, 0.76],'group': [0.2823639774859287], 'causal': [0.05813953488372093]}]]

all_uni_trans_metrics = [[{'disp_imp': [1.0742594910304548], 'stat_par_diff': [0.01838463127453005], 'acc': [0.74], 'group': [0.027999999999999997], 'causal': [0.003992015968063872]},
                          {'disp_imp': [0.0], 'stat_par_diff': [-0.004048582995951417], 'acc': [0.71], 'group': [0.0],'causal': [0.0]}],
                         [{'disp_imp': [2.556737588652482], 'stat_par_diff': [0.09068374302829993], 'acc': [0.7333333333333333], 'group': [0.082], 'causal': [0.03792415169660679]},
                          {'disp_imp': ["inf"], 'stat_par_diff': [0.2830188679245283], 'acc': [0.7366666666666667],'group': [0.128], 'causal': [0.10179640718562874]}],
                         [{'disp_imp': [1.194078947368421], 'stat_par_diff': [0.06095041322314049], 'group': [0.033999999999999975], 'acc': [0.7085714285714285]},
                          {'disp_imp': [1.17086000897117], 'stat_par_diff': [0.054323164486393294],'group': [0.10999999999999999], 'acc': [0.7114285714285714]}],
                         [{'disp_imp': [1.4106138420151626], 'stat_par_diff': [0.17341458376368518], 'acc': [0.69], 'group': [0.1459999999999999], 'causal': [0.1536926147704591]},
                          {'disp_imp': [1.583623374244367], 'stat_par_diff': [0.24337330990757006],'acc': [0.6933333333333334], 'group': [0.050000000000000044],'causal': [0.06187624750499002]}],
                         [{'disp_imp': [0.7304964539007092], 'stat_par_diff': [-0.00392480892377608], 'acc': [0.72], 'group': [0.00841248303934871], 'causal': [0.0]},
                          {'disp_imp': ["nan"], 'stat_par_diff': [0.0], 'acc': [0.7133333333333334], 'group': [0.02],'causal': [0.0196078431372549]}],
                         [{'disp_imp': [1.3191489361702127], 'stat_par_diff': [0.15957446808510634], 'acc': [0.6366666666666667], 'group': [0.08399999999999996], 'causal': [0.2055888223552894]},
                          {'disp_imp': [2.402256370355962], 'stat_par_diff': [0.5506836758078069],'acc': [0.6833333333333333], 'group': [0.36], 'causal': [0.3532934131736527]}],
                         [{'disp_imp': [1.0719241443108234], 'stat_par_diff': [0.032121462507746334], 'acc': [0.7066666666666667], 'group': [0.05799999999999994], 'causal': [0.09780439121756487]},
                          {'disp_imp': [1.9460916442048517], 'stat_par_diff': [0.3485600794438927],'acc': [0.6666666666666666], 'group': [0.21599999999999997],'causal': [0.13572854291417166]}],
                         [{'disp_imp': [1.0750702529104779], 'stat_par_diff': [0.019314191282792825], 'acc': [0.9766666666666667], 'group': [0.025911355887752396], 'causal': [0.26013513513513514]},
                          {'disp_imp': [1.4515929477265697], 'stat_par_diff': [0.11152700328469939], 'acc': [0.98],'group': [0.12148217636022512], 'causal': [0.26174496644295303]}],
                         [{'disp_imp': [1.3435948361469712], 'stat_par_diff': [0.08883183568677794], 'acc': [0.9633333333333334], 'group': [0.09616186900292034], 'causal': [0.2866666666666667]},
                          {'disp_imp': [1.909990720692855], 'stat_par_diff': [0.22473455045451074],'acc': [0.9033333333333333], 'group': [0.23686679174484054], 'causal': [0.2866666666666667]}],
                         [{'disp_imp': [1.3645122440786832], 'stat_par_diff': [0.0937822763891758], 'acc': [1.0], 'group': [0.10118017309205352], 'causal': [0.2866666666666667]},
                          {'disp_imp': [1.909990720692855], 'stat_par_diff': [0.22473455045451074], 'acc': [1.0],'group': [0.23686679174484054], 'causal': [0.2866666666666667]}]]

all_multi_orig_metrics = [{'disp_imp': [1.8881118881118881], 'stat_par_diff': [0.14472934472934473], 'group': [0.18], 'causal': [0.088], 'acc': [0.8033333333333333]},
                          {'disp_imp': ["inf"], 'stat_par_diff': [0.11965811965811966], 'acc': [0.74], 'group': [0.2],'causal': [0.172]},
                          {'disp_imp': [1.1117479642659498], 'stat_par_diff': [0.032541007194244576],'group': [0.11200000000000002], 'acc': [0.7128571428571429]},
                          {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902],'acc': [0.7766666666666666], 'group': [0.28600000000000003], 'causal': [0.182]},
                          {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902],'acc': [0.7766666666666666], 'group': [0.31122967479674796],'causal': [0.20161290322580644]},
                          {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902],'acc': [0.7766666666666666], 'group': [0.28600000000000003], 'causal': [0.182]},
                          {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902],'acc': [0.7766666666666666], 'group': [0.28600000000000003], 'causal': [0.182]},
                          {'disp_imp': [1.4574898785425099], 'stat_par_diff': [0.12877492877492874], 'acc': [0.76],'group': [0.05982905982905981], 'causal': [0.10738255033557047]},
                          {'disp_imp': [1.4574898785425099], 'stat_par_diff': [0.12877492877492874], 'acc': [0.76],'group': [0.05982905982905981], 'causal': [0.10738255033557047]},
                          {'disp_imp': [1.4574898785425099], 'stat_par_diff': [0.12877492877492874], 'acc': [0.76],'group': [0.05982905982905981], 'causal': [0.10738255033557047]}]

all_multi_trans_metrics = [{'disp_imp': ["nan"], 'stat_par_diff': [0.0], 'group': [0.0], 'causal': [0.0], 'acc': [0.7133333333333334]},
                           {'disp_imp': [2.307692307692308], 'stat_par_diff': [0.009686609686609688], 'group': [0.138],'causal': [0.1], 'acc': [0.7066666666666667]},
                           {'disp_imp': [1.052113973845554], 'stat_par_diff': [0.01692661870503598],'group': [0.11599999999999999], 'acc': [0.7028571428571428]},
                           {'disp_imp': [1.307692307692308], 'stat_par_diff': [0.13675213675213682], 'acc': [0.69],'group': [0.20199999999999996], 'causal': [0.098]},
                           {'disp_imp': [0.8653846153846153], 'stat_par_diff': [-0.003988603988603991], 'acc': [0.72], 'group': [0.12583333333333335], 'causal': [0.03636363636363636]},
                           {'disp_imp': [1.3432835820895521], 'stat_par_diff': [0.17037037037037034],'acc': [0.6466666666666666], 'group': [0.20400000000000007], 'causal': [0.266]},
                           {'disp_imp': [1.342657342657343], 'stat_par_diff': [0.13960113960113968], 'acc': [0.68],'group': [0.196], 'causal': [0.09]},
                           {'disp_imp': [1.2587412587412588], 'stat_par_diff': [0.06324786324786327], 'acc': [0.98],'group': [0.014041514041514047], 'causal': [0.26174496644295303]},
                           {'disp_imp': [1.3846153846153848], 'stat_par_diff': [0.09971509971509973],'acc': [0.9533333333333334], 'group': [0.03357753357753357],'causal': [0.2866666666666667]},
                           {'disp_imp': [1.3846153846153848], 'stat_par_diff': [0.09971509971509973], 'acc': [1.0], 'group': [0.03357753357753357], 'causal': [0.2866666666666667]}]

print("-->all_uni_orig_metrics", all_uni_orig_metrics)
print("-->all_uni_trans_metrics", all_uni_trans_metrics)
print("-->all_multi_orig_metrics", all_multi_orig_metrics)
print("-->all_multi_trans_metrics", all_multi_trans_metrics)

# processing_names = [RW_processing_name, OP_processing_name, DI_processing_name]
processing_names = ["RW", "OP", "DI", "META", "AD", "PR", "GR", "CEO", "EO", "RO"]
# processing_names = ["RW", "OP", "AD", "META"]
# dataset_name = "Adult income"
# sens_attrs = ["race", "sex"]
dataset_name = "German credit"
sens_attrs = ["sex", "age"]


for i in range(0, len(processing_names)):
    process_name = processing_names[i]
    print("-->process_name", process_name)
    uni_orig_metrics = all_uni_orig_metrics[i]
    uni_trans_metrics = all_uni_trans_metrics[i]
    multi_orig_metrics = all_multi_orig_metrics[i]
    multi_trans_metrics = all_multi_trans_metrics[i]
    print("group metric")
    percent = format((multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0]) / float(multi_orig_metrics['group'][0]), '.0%')
    print(str(round(multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0], 3)) + "(" + str(percent)+ ")")
    percent = format(
        (uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0]) / float(uni_orig_metrics[0]['group'][0]),
        '.0%')
    print(str(round(uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0], 3)) + "(" + str(percent) + ")")
    percent = format((uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0])/float(uni_orig_metrics[1]['group'][0]), '.0%')
    print(str(round(uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0], 3)) + "(" + str(percent) + ")")
    try:
        print("causal metric")
        percent = format((multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0]) / float(multi_orig_metrics['causal'][0]),'.0%')
        print(str(round(multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0], 3)) + "(" + str(percent) + ")")
        percent = format((uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0]) / float(
            uni_orig_metrics[0]['causal'][0]), '.0%')
        print(str(round(uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0], 3)) + "(" + str(
            percent) + ")")
        percent = format((uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0]) / float(uni_orig_metrics[1]['causal'][0]), '.0%')
        print(str(round(uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0], 3)) + "(" + str( percent) + ")")
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
