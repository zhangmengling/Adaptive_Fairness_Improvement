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

from aif360.metrics.metric_test import metric_test1, get_metrics, describe_metrics, initial_dnn, describe, \
    metric_test_new_inputs, describe_metrics_new_inputs, metric_test_multivariate, metric_test_causal
from plot_result import Plot

#pre-processing
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

dataset_name = "Adult income"

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

# adversarial debias --inprocessing (maxabsscaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5350309096261406], 'stat_par_diff': [-0.10105242954480025], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.11000000000000001], 'causal': [0.017964071856287425]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.9489329192183816], 'stat_par_diff': [-0.012042383303299459], 'acc': [0.8458023144394486], 'group': [0.04200000000000001], 'causal': [0.10978043912175649]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.26219575992624666], 'stat_par_diff': [-0.19669418759895374], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.16399999999999998], 'causal': [0.0718562874251497]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4704536173609729], 'stat_par_diff': [-0.1140472645334842], 'acc': [0.8486769366846023], 'group': [0.013999999999999999], 'causal': [0.05588822355289421]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4235316765164495], 'stat_par_diff': [-0.12330508928211137], 'acc': [0.8495614358369573], 'group': [0.266], 'causal': [0.088]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5544277624086987], 'stat_par_diff': [-0.09211231004802446], 'acc': [0.8508144763027935], 'group': [0.09200000000000001], 'causal': [0.11]})

# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5350309096261406], 'stat_par_diff': [-0.10105242954480025], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.11000000000000001], 'causal': [0.017964071856287425]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.9489329192183816], 'stat_par_diff': [-0.012042383303299459], 'acc': [0.8458023144394486], 'group': [0.04200000000000001], 'causal': [0.10978043912175649]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.26219575992624666], 'stat_par_diff': [-0.19669418759895374], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.16399999999999998], 'causal': [0.0718562874251497]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4704536173609729], 'stat_par_diff': [-0.1140472645334842], 'acc': [0.8486769366846023], 'group': [0.013999999999999999], 'causal': [0.05588822355289421]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4235316765164495], 'stat_par_diff': [-0.12330508928211137], 'acc': [0.8495614358369573], 'group': [0.266], 'causal': [0.088]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5544277624086987], 'stat_par_diff': [-0.09211231004802446], 'acc': [0.8508144763027935], 'group': [0.09200000000000001], 'causal': [0.078]})


# adversarial debias --inprocessing (maxabsscaler) adclassifier-->adcalssifier
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5822590216856773], 'stat_par_diff': [-0.09491827765515429], 'acc': [0.8523623498194147, 0.8523623498194147], 'group': [0.016], 'causal': [0.017964071856287425]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.835131281728252], 'stat_par_diff': [-0.032443646503097584], 'acc': [0.8495614358369573], 'group': [0.011999999999999997], 'causal': [0.06786427145708583]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.27663267375341527], 'stat_par_diff': [-0.20183328121975552], 'acc': [0.8523623498194147, 0.8523623498194147], 'group': [0.016], 'causal': [0.01996007984031936]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.7358454961949664], 'stat_par_diff': [-0.055825131648823995], 'acc': [0.8406427360507113], 'group': [0.05600000000000001], 'causal': [0.06986027944111776]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.44989343619347605], 'stat_par_diff': [-0.1237772273088657], 'acc': [0.8523623498194147], 'group': [0.016], 'causal': [0.024]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5559719888745466], 'stat_par_diff': [-0.1082346700088358], 'acc': [0.8458023144394486], 'group': [0.188], 'causal': [0.106]})


# meta --inprocessing (maxabsscaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5350309096261406], 'stat_par_diff': [-0.10105242954480025], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.11000000000000001], 'causal': [0.017964071856287425]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.7000536179618609], 'stat_par_diff': [-0.1704990954295199], 'acc': [0.6762733102380777], 'group': [0.14999999999999997], 'causal': [0.021956087824351298]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.26219575992624666], 'stat_par_diff': [-0.19669418759895374], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.16399999999999998], 'causal': [0.0718562874251497]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.9356373313973183], 'stat_par_diff': [-0.04239868872311803], 'acc': [0.5749244490307364], 'group': [0.08599999999999997], 'causal': [0.2954091816367265]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4235316765164495], 'stat_par_diff': [-0.12330508928211137], 'acc': [0.8495614358369573], 'group': [0.266], 'causal': [0.088]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.7899930362027393], 'stat_par_diff': [-0.12026633733887088], 'acc': [0.6355126409670524], 'group': [0.29], 'causal': [0.492]})

# meta --inprocessing (maxabsscaler) metaclassifier-->metaclassifier
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5279271621237025], 'stat_par_diff': [-0.08364357968217409], 'acc': [0.8381366551190389, 0.8381366551190389], 'group': [0.038000000000000006], 'causal': [0.001996007984031936]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.7000536179618609], 'stat_par_diff': [-0.1704990954295199], 'acc': [0.6762733102380777], 'group': [0.14999999999999997], 'causal': [0.021956087824351298]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.28602276128088316], 'stat_par_diff': [-0.15345647027292625], 'acc': [0.8381366551190389, 0.8381366551190389], 'group': [0.044], 'causal': [0.003992015968063872]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.9356373313973183], 'stat_par_diff': [-0.04239868872311803], 'acc': [0.5749244490307364], 'group': [0.08599999999999997], 'causal': [0.2954091816367265]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.436181502181819], 'stat_par_diff': [-0.09790088782724896], 'acc': [0.8381366551190389], 'group': [0.066], 'causal': [0.008]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.7899930362027393], 'stat_par_diff': [-0.12026633733887088], 'acc': [0.6355126409670524], 'group': [0.29], 'causal': [0.492]})


# prejudice remover --inprocssing (StandardScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5350309096261406], 'stat_par_diff': [-0.10105242954480025], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.11000000000000001], 'causal': [0.017964071856287425]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.9489329192183816], 'stat_par_diff': [-0.012042383303299459], 'acc': [0.8458023144394486], 'group': [0.04200000000000001], 'causal': [0.10978043912175649]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.26219575992624666], 'stat_par_diff': [-0.19669418759895374], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.16399999999999998], 'causal': [0.0718562874251497]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4704536173609729], 'stat_par_diff': [-0.1140472645334842], 'acc': [0.8486769366846023], 'group': [0.013999999999999999], 'causal': [0.05588822355289421]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4235316765164495], 'stat_par_diff': [-0.12330508928211137], 'acc': [0.8495614358369573], 'group': [0.266], 'causal': [0.088]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5544277624086987], 'stat_par_diff': [-0.09211231004802446], 'acc': [0.8508144763027935], 'group': [0.09200000000000001], 'causal': [0.078]})

# prejudge remover --inprocessing (MaxAbsScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5350309096261406], 'stat_par_diff': [-0.10105242954480025], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.11000000000000001], 'causal': [0.017964071856287425]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.9489329192183816], 'stat_par_diff': [-0.012042383303299459], 'acc': [0.8458023144394486], 'group': [0.04200000000000001], 'causal': [0.10978043912175649]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.26219575992624666], 'stat_par_diff': [-0.19669418759895374], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.16399999999999998], 'causal': [0.0718562874251497]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4704536173609729], 'stat_par_diff': [-0.1140472645334842], 'acc': [0.8486769366846023], 'group': [0.013999999999999999], 'causal': [0.05588822355289421]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4235316765164495], 'stat_par_diff': [-0.12330508928211137], 'acc': [0.8495614358369573], 'group': [0.266], 'causal': [0.088]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5544277624086987], 'stat_par_diff': [-0.09211231004802446], 'acc': [0.8508144763027935], 'group': [0.09200000000000001], 'causal': [0.078]})

# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5350309096261406], 'stat_par_diff': [-0.10105242954480025], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.10800000000000001], 'causal': [0.017964071856287425]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.46164935123128426], 'stat_par_diff': [-0.11209467600737874], 'acc': [0.8403479029999263], 'group': [0.074], 'causal': [0.03792415169660679]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.26219575992624666], 'stat_par_diff': [-0.19669418759895374], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.16399999999999998], 'causal': [0.0718562874251497]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.20666829821844288], 'stat_par_diff': [-0.0323385347647196], 'acc': [0.7798334193263065], 'group': [0.0], 'causal': [0.023952095808383235]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4235316765164495], 'stat_par_diff': [-0.12330508928211137], 'acc': [0.8495614358369573], 'group': [0.266], 'causal': [0.088]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4001938017367525], 'stat_par_diff': [-0.08685447037011337], 'acc': [0.830176162747844], 'group': [0.084], 'causal': [0.03]})


# CEO --post processing (MaxAbsScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5350805470830751], 'stat_par_diff': [-0.1014813008043391], 'acc': [0.8497088523623498, 0.8497088523623498], 'group': [0.1281746031746032], 'causal': [0.0]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.7708754588862966], 'stat_par_diff': [-0.04869290454204564], 'acc': [0.9585022481020122], 'group': [0.03728222996515679], 'causal': [0.1826086956521739]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.26282934844520384], 'stat_par_diff': [-0.1973287324488915], 'acc': [0.8497088523623498, 0.8497088523623498], 'group': [0.22343580112681624], 'causal': [0.10738255033557047]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.6024393446876292], 'stat_par_diff': [-0.07422576594551403], 'acc': [0.9156040392127958], 'group': [0.10301142263759087], 'causal': [0.1366120218579235]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.42422868731032287], 'stat_par_diff': [-0.12370033638263211], 'acc': [0.8497088523623498], 'group': [0.275974025974026], 'causal': [0.13978494623655913]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.6421267635025326], 'stat_par_diff': [-0.07604066169019694], 'acc': [0.9566595415346061], 'group': [0.20261064017889646], 'causal': [0.22388059701492538]})

# EO --postprocessing (MaxAbsScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5350805470830751], 'stat_par_diff': [-0.1014813008043391], 'acc': [0.8496351440996536, 0.8496351440996536], 'group': [0.1281746031746032], 'causal': [0.0]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.627876156298822], 'stat_par_diff': [-0.09709386740311857], 'acc': [0.9713274858111595], 'group': [0.10230928491798055], 'causal': [0.26013513513513514]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.26282934844520384], 'stat_par_diff': [-0.1973287324488915], 'acc': [0.8496351440996536, 0.8496351440996536], 'group': [0.22343580112681624], 'causal': [0.10738255033557047]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.36108418385231245], 'stat_par_diff': [-0.1990214062724895], 'acc': [0.94420284513894], 'group': [0.2546454095656954], 'causal': [0.26013513513513514]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.42422868731032287], 'stat_par_diff': [-0.12370033638263211], 'acc': [0.8497088523623498], 'group': [0.275974025974026], 'causal': [0.13978494623655913]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5271890049212574], 'stat_par_diff': [-0.12236540761314657], 'acc': [0.9588707894154935], 'group': [0.2924331829225963], 'causal': [0.26013513513513514]})

# reject option --postprocessing (MaxAbsScaler)
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5350805470830751], 'stat_par_diff': [-0.1014813008043391], 'acc': [0.8497088523623498, 0.8497088523623498], 'group': [0.1281746031746032], 'causal': [0.0]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.627876156298822], 'stat_par_diff': [-0.09709386740311857], 'acc': [1.0], 'group': [0.10230928491798055], 'causal': [0.26013513513513514]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.26282934844520384], 'stat_par_diff': [-0.1973287324488915], 'acc': [0.8497088523623498, 0.8497088523623498], 'group': [0.22343580112681624], 'causal': [0.10738255033557047]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.36108418385231245], 'stat_par_diff': [-0.1990214062724895], 'acc': [1.0], 'group': [0.2546454095656954], 'causal': [0.26013513513513514]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.42422868731032287], 'stat_par_diff': [-0.12370033638263211], 'acc': [0.8497088523623498], 'group': [0.275974025974026], 'causal': [0.13978494623655913]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5271890049212574], 'stat_par_diff': [-0.12236540761314657], 'acc': [1.0], 'group': [0.2924331829225963], 'causal': [0.26013513513513514]})



# prejudice remover
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5869088573918627], 'stat_par_diff': [-0.08402455668937868], 'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.10400000000000001], 'causal': [0.007984031936127744]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4617612942112671], 'stat_par_diff': [-0.13132061958635005], 'acc': [0.8294390801208815], 'group': [0.124], 'causal': [0.05788423153692615]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.2615807494641273], 'stat_par_diff': [-0.18575167632008655], 'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.12400000000000001], 'causal': [0.021956087824351298]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.2837490482734109], 'stat_par_diff': [-0.18563975620771977], 'acc': [0.8441070243974349], 'group': [0.158], 'causal': [0.07784431137724551]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4354775536043105], 'stat_par_diff': [-0.11363388092184563], 'acc': [0.8480872705830323], 'group': [0.21800000000000003], 'causal': [0.03]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4170977390553082], 'stat_par_diff': [-0.09841326283813712], 'acc': [0.8211100464362056], 'group': [0.172], 'causal': [0.03]})

# exponentiated gradient reduction
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5869088573918627], 'stat_par_diff': [-0.08402455668937868], 'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.10400000000000001], 'causal': [0.007984031936127744]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.728814444293606], 'stat_par_diff': [-0.05191972425706151], 'acc': [0.8306184123240216], 'group': [0.07399999999999998], 'causal': [0.021956087824351298]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.2615807494641273], 'stat_par_diff': [-0.18575167632008655], 'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.12400000000000001], 'causal': [0.021956087824351298]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4164003760220749], 'stat_par_diff': [-0.13243099914136675], 'acc': [0.830765828849414], 'group': [0.07999999999999999], 'causal': [0.05588822355289421]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4354775536043105], 'stat_par_diff': [-0.11363388092184563], 'acc': [0.8480872705830323], 'group': [0.21800000000000003], 'causal': [0.03]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5869709101579331], 'stat_par_diff': [-0.07884420426119343], 'acc': [0.8303235792732365], 'group': [0.15799999999999997], 'causal': [0.07]})


# gradient reduction -- DemographicParity
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5869088573918627], 'stat_par_diff': [-0.08402455668937868], 'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.10400000000000001], 'causal': [0.007984031936127744]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.8780046863567744], 'stat_par_diff': [-0.023409004474869294], 'acc': [0.83968452863566], 'group': [0.05200000000000002], 'causal': [0.04590818363273453]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.2615807494641273], 'stat_par_diff': [-0.18575167632008655], 'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.12400000000000001], 'causal': [0.021956087824351298]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [1.0514117778692134], 'stat_par_diff': [0.008483643781361211], 'acc': [0.8210363381735093], 'group': [0.086], 'causal': [0.17764471057884232]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4354775536043105], 'stat_par_diff': [-0.11363388092184563], 'acc': [0.8480872705830323], 'group': [0.21800000000000003], 'causal': [0.03]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.7745549481127963], 'stat_par_diff': [-0.04088667056206907], 'acc': [0.823321294317093], 'group': [0.102], 'causal': [0.104]})

# gradient reduction -- EqualizedOdds
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5869088573918627], 'stat_par_diff': [-0.08402455668937868], 'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.10400000000000001], 'causal': [0.007984031936127744]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.8780046863567744], 'stat_par_diff': [-0.023409004474869294], 'acc': [0.8396108203729639], 'group': [0.05200000000000002], 'causal': [0.04590818363273453]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.2615807494641273], 'stat_par_diff': [-0.18575167632008655], 'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.12400000000000001], 'causal': [0.021956087824351298]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5424947954051059], 'stat_par_diff': [-0.0990804186953738], 'acc': [0.830470995798629], 'group': [0.04200000000000001], 'causal': [0.07984031936127745]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4354775536043105], 'stat_par_diff': [-0.11363388092184563], 'acc': [0.8480872705830323], 'group': [0.21800000000000003], 'causal': [0.03]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5869709101579331], 'stat_par_diff': [-0.07884420426119343], 'acc': [0.8303235792732365], 'group': [0.15799999999999997], 'causal': [0.07]})

# calibrated euqodds -- postprocessing
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5869088573918627], 'stat_par_diff': [-0.08402455668937868], 'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.08399999999999999], 'causal': [0.013972055888223553]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.7137094136205712], 'stat_par_diff': [-0.06571491279514055], 'acc': [0.9730964841158694], 'group': [0.06], 'causal': [0.2215568862275449]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.2615807494641273], 'stat_par_diff': [-0.18575167632008655], 'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.196], 'causal': [0.033932135728542916]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.6265808120521608], 'stat_par_diff': [-0.06703230545505082], 'acc': [0.9107392938748434], 'group': [0.072], 'causal': [0.15169660678642716]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4354775536043105], 'stat_par_diff': [-0.11363388092184563], 'acc': [0.8480872705830323], 'group': [0.22000000000000003], 'causal': [0.036]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.6327423752891234], 'stat_par_diff': [-0.07919200495026152], 'acc': [0.959607872042456], 'group': [0.172], 'causal': [0.212]})

# eqodds -- postprocessing
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5499430640479012], 'stat_par_diff': [-0.10404084428947677], 'acc': [0.8491928945234761, 0.8491928945234761], 'group': [0.094], 'causal': [0.003992015968063872]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.627876156298822], 'stat_par_diff': [-0.09709386740311857], 'acc': [0.9799513525466205], 'group': [0.10200000000000001], 'causal': [0.26147704590818366]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.25831705477656997], 'stat_par_diff': [-0.21120504968056045], 'acc': [0.8491928945234761, 0.8491928945234761], 'group': [0.22799999999999998], 'causal': [0.021956087824351298]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.3612057808300871], 'stat_par_diff': [-0.19896184376499687], 'acc': [0.9390432667502027], 'group': [0.21000000000000002], 'causal': [0.26147704590818366]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.42270753953329554], 'stat_par_diff': [-0.13148605555890985], 'acc': [0.8492666027861723], 'group': [0.244], 'causal': [0.03]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5271890049212574], 'stat_par_diff': [-0.12236540761314657], 'acc': [0.9586496646274048], 'group': [0.246], 'causal': [0.26]})

# reject_option -- postprocessing
# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5499430640479012], 'stat_par_diff': [-0.10404084428947677], 'acc': [0.8492666027861723, 0.8492666027861723], 'group': [0.094], 'causal': [0.003992015968063872]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.627876156298822], 'stat_par_diff': [-0.09709386740311857], 'acc': [1.0], 'group': [0.10200000000000001], 'causal': [0.26147704590818366]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.25832895742502726], 'stat_par_diff': [-0.21114384340118025], 'acc': [0.8492666027861723, 0.8492666027861723], 'group': [0.22799999999999998], 'causal': [0.021956087824351298]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.36108418385231245], 'stat_par_diff': [-0.1990214062724895], 'acc': [1.0], 'group': [0.21200000000000002], 'causal': [0.26147704590818366]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.42270753953329554], 'stat_par_diff': [-0.13148605555890985], 'acc': [0.8492666027861723], 'group': [0.244], 'causal': [0.03]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5271890049212574], 'stat_par_diff': [-0.12236540761314657], 'acc': [1.0], 'group': [0.246], 'causal': [0.26]})



# -->uni_race_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.5350309096261406], 'stat_par_diff': [-0.10105242954480025], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.10800000000000001], 'causal': [0.017964071856287425]})
# -->uni_race_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.46164935123128426], 'stat_par_diff': [-0.11209467600737874], 'acc': [0.8403479029999263], 'group': [0.074], 'causal': [0.03792415169660679]})
# -->uni_sex_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.26219575992624666], 'stat_par_diff': [-0.19669418759895374], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.16399999999999998], 'causal': [0.0718562874251497]})
# -->uni_sex_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.20666829821844288], 'stat_par_diff': [-0.0323385347647196], 'acc': [0.7798334193263065], 'group': [0.0], 'causal': [0.023952095808383235]})
# -->all_multi_orig_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4235316765164495], 'stat_par_diff': [-0.12330508928211137], 'acc': [0.8495614358369573], 'group': [0.266], 'causal': [0.088]})
# -->all_multi_trans_metrics defaultdict(<class 'list'>, {'disp_imp': [0.4001938017367525], 'stat_par_diff': [-0.08685447037011337], 'acc': [0.830176162747844], 'group': [0.084], 'causal': [0.03]})


# basic remove sensitive features (test on test data)
basic = [{'disp_imp': [0.5350309096261406], 'stat_par_diff': [-0.10105242954480025], 'acc': [0.8495614358369573], 'group': [0.098]},
         {'disp_imp': [0.5191945126098773], 'stat_par_diff': [-0.11007436493465413], 'acc': [0.8388000294833051], 'group': [0.11399999999999999]},
         {'disp_imp': [0.26219575992624666], 'stat_par_diff': [-0.19669418759895374], 'acc': [0.8495614358369573], 'group': [0.21200000000000002]},
         {'disp_imp': [0.2686241992913151], 'stat_par_diff': [-0.20271266062149162], 'acc': [0.8379892385936464], 'group': [0.23399999999999999]},
         {'disp_imp': [0.4235316765164495], 'stat_par_diff': [-0.12330508928211137], 'acc': [0.8495614358369573], 'group': [0.24200000000000002]},
         {'disp_imp': [0.4228995338737346], 'stat_par_diff': [-0.12012128192748707], 'acc': [0.8368836146532026], 'group': [0.24]}]

# basic remove sensitive features (test on generated samples)
basic = [{'disp_imp': [0.5350309096261406], 'stat_par_diff': [-0.10105242954480025], 'acc': [0.8495614358369573], 'group': [0.11000000000000001]},
         {'disp_imp': [0.5191945126098773], 'stat_par_diff': [-0.11007436493465413], 'acc': [0.8388000294833051], 'group': [0.09200000000000001]},
         {'disp_imp': [0.26219575992624666], 'stat_par_diff': [-0.19669418759895374], 'acc': [0.8495614358369573], 'group': [0.16399999999999998]},
         {'disp_imp': [0.2686241992913151], 'stat_par_diff': [-0.20271266062149162], 'acc': [0.8379892385936464], 'group': [0.104]},
         {'disp_imp': [0.4235316765164495], 'stat_par_diff': [-0.12330508928211137], 'acc': [0.8495614358369573], 'group': [0.266]},
         {'disp_imp': [0.4228995338737346], 'stat_par_diff': [-0.12012128192748707], 'acc': [0.8368836146532026], 'group': [0.17400000000000002]}]


# i: method index, j: sensitive attribute index e.g. (race, sex) for Adult Income
all_uni_orig_metrics = [[{'disp_imp': [0.584176974233109], 'stat_par_diff': [-0.1026330731582483], 'acc': [0.8356305741873664], 'group': [0.086], 'causal': [0.06986027944111776]},
                         {'disp_imp': [0.3499775857715558], 'stat_par_diff': [-0.19114555570445152], 'acc': [0.8356305741873664], 'group': [0.134], 'causal': [0.059880239520958084]}],
                        [{'disp_imp': [0.362920548355989], 'stat_par_diff': [-0.10454975011865361], 'group': [0.126], 'causal': [0.0658682634730539], 'acc': [0.8016788370982052]},
                         {'disp_imp': [0.0], 'stat_par_diff': [-0.22355105795768168], 'group': [0.21], 'causal': [0.17764471057884232], 'acc': [0.8016788370982052]}],
                        [{'disp_imp': [0.5806862455744509], 'stat_par_diff': [-0.08993312125096391], 'group': [0.05199999999999999], 'acc': [0.8457431685357764]},
                         {'disp_imp': [0.31700067711336477], 'stat_par_diff': [-0.17545200852534631], 'group': [0.116], 'acc': [0.8448270415416206]}],
                        [{'disp_imp': [0.5350309096261406], 'stat_par_diff': [-0.10105242954480025], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.11000000000000001], 'causal': [0.017964071856287425]},
                         {'disp_imp': [0.26219575992624666], 'stat_par_diff': [-0.19669418759895374], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.16399999999999998], 'causal': [0.0718562874251497]}],
                        [{'disp_imp': [0.5350309096261406], 'stat_par_diff': [-0.10105242954480025], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.11000000000000001], 'causal': [0.017964071856287425]},
                         {'disp_imp': [0.26219575992624666], 'stat_par_diff': [-0.19669418759895374], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.16399999999999998], 'causal': [0.0718562874251497]}],
                        [{'disp_imp': [0.5350309096261406], 'stat_par_diff': [-0.10105242954480025], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.10800000000000001], 'causal': [0.017964071856287425]},
                         {'disp_imp': [0.26219575992624666], 'stat_par_diff': [-0.19669418759895374], 'acc': [0.8495614358369573, 0.8495614358369573], 'group': [0.16399999999999998], 'causal': [0.0718562874251497]}],
                        [{'disp_imp': [0.5869088573918627], 'stat_par_diff': [-0.08402455668937868], 'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.10400000000000001], 'causal': [0.007984031936127744]},
                         {'disp_imp': [0.2615807494641273], 'stat_par_diff': [-0.18575167632008655],'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.12400000000000001],'causal': [0.021956087824351298]}],
                        [{'disp_imp': [0.5869088573918627], 'stat_par_diff': [-0.08402455668937868], 'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.08399999999999999], 'causal': [0.013972055888223553]},
                         {'disp_imp': [0.2615807494641273], 'stat_par_diff': [-0.18575167632008655],'acc': [0.8480872705830323, 0.8480872705830323], 'group': [0.196],'causal': [0.033932135728542916]}],
                        [{'disp_imp': [0.5499430640479012], 'stat_par_diff': [-0.10404084428947677], 'acc': [0.8491928945234761, 0.8491928945234761], 'group': [0.094], 'causal': [0.003992015968063872]},
                         {'disp_imp': [0.25831705477656997], 'stat_par_diff': [-0.21120504968056045], 'acc': [0.8491928945234761, 0.8491928945234761], 'group': [0.22799999999999998], 'causal': [0.021956087824351298]}],
                        [{'disp_imp': [0.5499430640479012], 'stat_par_diff': [-0.10404084428947677], 'acc': [0.8492666027861723, 0.8492666027861723], 'group': [0.094], 'causal': [0.003992015968063872]},
                         {'disp_imp': [0.25832895742502726], 'stat_par_diff': [-0.21114384340118025], 'acc': [0.8492666027861723, 0.8492666027861723], 'group': [0.22799999999999998], 'causal': [0.021956087824351298]}]]

all_uni_trans_metrics = [[{'disp_imp': [0.6869002147058121], 'stat_par_diff': [-0.0815049991291954], 'acc': [0.831650328001769], 'group': [0.062], 'causal': [0.09580838323353294]},
                          {'disp_imp': [0.43671438273349666], 'stat_par_diff': [-0.17091958130462617], 'acc': [0.827817498341564], 'group': [0.11000000000000001], 'causal': [0.1277445109780439]}],
                         [{'disp_imp': [0.38612949523383466], 'stat_par_diff': [-0.09468577748322285], 'acc': [0.8023612912031666], 'group': [0.08600000000000001], 'causal': [0.031936127744510975]},
                          {'disp_imp': [0.0], 'stat_par_diff': [-0.2131248083409997], 'acc': [0.802088309561182], 'group': [0.174], 'causal': [0.14770459081836326]}],
                         [{'disp_imp': [0.5776922288802987], 'stat_par_diff': [-0.10205009760107189], 'group': [0.052000000000000005], 'acc': [0.8457431685357764]},
                          {'disp_imp': [0.3164054804281286], 'stat_par_diff': [-0.1998406234154937], 'group': [0.13], 'acc': [0.8448270415416206]}],
                         [{'disp_imp': [0.7000536179618609], 'stat_par_diff': [-0.1704990954295199], 'acc': [0.6762733102380777], 'group': [0.14999999999999997], 'causal': [0.021956087824351298]},
                          {'disp_imp': [0.9356373313973183], 'stat_par_diff': [-0.04239868872311803], 'acc': [0.5749244490307364], 'group': [0.08599999999999997], 'causal': [0.2954091816367265]}],
                         [{'disp_imp': [0.9489329192183816], 'stat_par_diff': [-0.012042383303299459], 'acc': [0.8458023144394486], 'group': [0.04200000000000001], 'causal': [0.10978043912175649]},
                          {'disp_imp': [0.4704536173609729], 'stat_par_diff': [-0.1140472645334842], 'acc': [0.8486769366846023], 'group': [0.013999999999999999], 'causal': [0.05588822355289421]}],
                         [{'disp_imp': [0.46164935123128426], 'stat_par_diff': [-0.11209467600737874], 'acc': [0.8403479029999263], 'group': [0.074], 'causal': [0.03792415169660679]},
                          {'disp_imp': [0.20666829821844288], 'stat_par_diff': [-0.0323385347647196], 'acc': [0.7798334193263065], 'group': [0.0], 'causal': [0.023952095808383235]}],
                         [{'disp_imp': [0.8780046863567744], 'stat_par_diff': [-0.023409004474869294], 'acc': [0.8396108203729639], 'group': [0.05200000000000002], 'causal': [0.04590818363273453]},
                          {'disp_imp': [0.5424947954051059], 'stat_par_diff': [-0.0990804186953738],'acc': [0.830470995798629], 'group': [0.04200000000000001], 'causal': [0.07984031936127745]}],
                         [{'disp_imp': [0.7137094136205712], 'stat_par_diff': [-0.06571491279514055], 'acc': [0.9730964841158694], 'group': [0.06], 'causal': [0.2215568862275449]},
                          {'disp_imp': [0.6265808120521608], 'stat_par_diff': [-0.06703230545505082],'acc': [0.9107392938748434], 'group': [0.072], 'causal': [0.15169660678642716]}],
                         [{'disp_imp': [0.627876156298822], 'stat_par_diff': [-0.09709386740311857], 'acc': [0.9799513525466205], 'group': [0.10200000000000001], 'causal': [0.26147704590818366]},
                          {'disp_imp': [0.3612057808300871], 'stat_par_diff': [-0.19896184376499687],'acc': [0.9390432667502027], 'group': [0.21000000000000002],'causal': [0.26147704590818366]}],
                         [{'disp_imp': [0.627876156298822], 'stat_par_diff': [-0.09709386740311857], 'acc': [1.0], 'group': [0.10200000000000001], 'causal': [0.26147704590818366]},
                          {'disp_imp': [0.36108418385231245], 'stat_par_diff': [-0.1990214062724895], 'acc': [1.0], 'group': [0.21200000000000002], 'causal': [0.26147704590818366]}]]

all_multi_orig_metrics = [{'disp_imp': [0.5101726154022098], 'stat_par_diff': [-0.11955292188481477], 'group': [0.22999999999999998], 'causal': [0.126], 'acc': [0.8356305741873664]},
                          {'disp_imp': [0.12954426763297364], 'stat_par_diff': [-0.13900596470877594], 'acc': [0.8016788370982052], 'group': [0.252], 'causal': [0.204]},
                          {'disp_imp': [0.49074072810204933], 'stat_par_diff': [-0.11029352471431203], 'group': [0.166], 'acc': [0.8462486179118622]},
                          {'disp_imp': [0.4235316765164495], 'stat_par_diff': [-0.12330508928211137],'acc': [0.8495614358369573], 'group': [0.266], 'causal': [0.088]},
                          {'disp_imp': [0.4235316765164495], 'stat_par_diff': [-0.12330508928211137], 'acc': [0.8495614358369573], 'group': [0.266], 'causal': [0.088]},
                          {'disp_imp': [0.4235316765164495], 'stat_par_diff': [-0.12330508928211137], 'acc': [0.8495614358369573], 'group': [0.266], 'causal': [0.088]},
                          {'disp_imp': [0.4354775536043105], 'stat_par_diff': [-0.11363388092184563],'acc': [0.8480872705830323], 'group': [0.21800000000000003], 'causal': [0.03]},
                          {'disp_imp': [0.4354775536043105], 'stat_par_diff': [-0.11363388092184563],'acc': [0.8480872705830323], 'group': [0.22000000000000003], 'causal': [0.036]},
                          {'disp_imp': [0.42270753953329554], 'stat_par_diff': [-0.13148605555890985],'acc': [0.8492666027861723], 'group': [0.244], 'causal': [0.03]},
                          {'disp_imp': [0.42270753953329554], 'stat_par_diff': [-0.13148605555890985],'acc': [0.8492666027861723], 'group': [0.244], 'causal': [0.03]}]

all_multi_trans_metrics = [{'disp_imp': [0.5868815745338717], 'stat_par_diff': [-0.09939838268128542], 'group': [0.24], 'causal': [0.144], 'acc': [0.8264170413503353]},
                           {'disp_imp': [0.21745539399048172], 'stat_par_diff': [-0.13028124875514435], 'group': [0.208], 'causal': [0.172], 'acc': [0.8017470825087013]},
                           {'disp_imp': [0.4850352961124847], 'stat_par_diff': [-0.12300095797669965], 'group': [0.17400000000000002], 'acc': [0.8429631969673037]},
                           {'disp_imp': [0.7899930362027393], 'stat_par_diff': [-0.12026633733887088], 'acc': [0.6355126409670524], 'group': [0.29], 'causal': [0.492]},
                           {'disp_imp': [0.5544277624086987], 'stat_par_diff': [-0.09211231004802446], 'acc': [0.8508144763027935], 'group': [0.09200000000000001], 'causal': [0.11]},
                           {'disp_imp': [0.4001938017367525], 'stat_par_diff': [-0.08685447037011337], 'acc': [0.830176162747844], 'group': [0.084], 'causal': [0.03]},
                           {'disp_imp': [0.5869709101579331], 'stat_par_diff': [-0.07884420426119343],'acc': [0.8303235792732365], 'group': [0.15799999999999997], 'causal': [0.07]},
                           {'disp_imp': [0.6327423752891234], 'stat_par_diff': [-0.07919200495026152],'acc': [0.959607872042456], 'group': [0.172], 'causal': [0.212]},
                           {'disp_imp': [0.5271890049212574], 'stat_par_diff': [-0.12236540761314657], 'acc': [0.9586496646274048], 'group': [0.246], 'causal': [0.26]},
                           {'disp_imp': [0.5271890049212574], 'stat_par_diff': [-0.12236540761314657], 'acc': [1.0],'group': [0.246], 'causal': [0.26]}]

print("-->all_uni_orig_metrics", all_uni_orig_metrics)
print("-->all_uni_trans_metrics", all_uni_trans_metrics)
print("-->all_multi_orig_metrics", all_multi_orig_metrics)
print("-->all_multi_trans_metrics", all_multi_trans_metrics)

# processing_names = ["RW", "OP", "DI", "AD", "META", "PR", "GR", "CEO", "EO", "RO"]
processing_names = ["RW", "OP", "DI", "META", "AD", "PR", "GR", "CEO", "EO", "RO"]
# processing_names = ["RW", "OP", "AD", "META"]
sens_attrs = ["race", "sex"]
#
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
    percent = format((uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0])/float(uni_orig_metrics[1]['group'][0]), '.0%')
    print(str(round(uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0], 3)) + "(" + str(percent) + ")")
    percent = format((uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0]) / float(uni_orig_metrics[0]['group'][0]),'.0%')
    print(str(round(uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0], 3)) + "(" + str(percent) + ")")
    try:
        print("causal metric")
        percent = format((multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0]) / float(multi_orig_metrics['causal'][0]),'.0%')
        print(str(round(multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0], 3)) + "(" + str(percent) + ")")
        percent = format((uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0]) / float(uni_orig_metrics[1]['causal'][0]), '.0%')
        print(str(round(uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0], 3)) + "(" + str( percent) + ")")
        percent = format((uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0]) / float(uni_orig_metrics[0]['causal'][0]), '.0%')
        print(str(round(uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0], 3)) + "(" + str(percent) + ")")
    except:
        print("no causal metric")

Plot = Plot(dataset_name=dataset_name, sens_attr=sens_attrs, processing_name=processing_names)
multi_metric_names = ["stat_par_diff", "avg_odds_diff", "eq_opp_diff"]
## 1 image
# Plot.plot_abs_acc_all_metric(all_uni_orig_metrics[0], all_uni_trans_metrics[0], all_uni_orig_metrics[1],
#                              all_uni_trans_metrics[1], all_multi_orig_metrics, all_multi_trans_metrics)

# 2 images: one for group metric. one for causal metric
Plot.plot_abs_acc_multi_metric(all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics, all_multi_trans_metrics)
# 3 images: one for 'race', one for 'sex', one for 'race,sex'
# Plot.plot_abs_acc_individual_metric(all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics, all_multi_trans_metrics)


