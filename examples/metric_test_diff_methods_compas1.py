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
RW = [{'disp_imp': [0.8541774139284513], 'stat_par_diff': [-0.09447108398721304], 'acc': [0.6862850971922246], 'group': [0.11328759703917674], 'causal': [0.2215568862275449]},
{'disp_imp': [0.9910050675675677], 'stat_par_diff': [-0.004642545771577966], 'acc': [0.6511879049676026], 'group': [0.03044979784366575], 'causal': [0.3393213572854291]},
{'disp_imp': [0.7649488070563668], 'stat_par_diff': [-0.15980378134121598], 'acc': [0.6862850971922246], 'group': [0.17018034825870643], 'causal': [0.21157684630738524]},
{'disp_imp': [0.9196761423096833], 'stat_par_diff': [-0.05712802419888863], 'acc': [0.6927645788336934], 'group': [0.06429804372842352], 'causal': [0.16966067864271456]},
{'disp_imp': [0.8442063492063492], 'stat_par_diff': [-0.10435938330675176], 'group': [0.22918730248573183], 'causal': [0.336], 'acc': [0.6862850971922246]},
{'disp_imp': [0.9441223832528179], 'stat_par_diff': [-0.03689526847421587], 'group': [0.15896586473306845], 'causal': [0.458], 'acc': [0.6744060475161987]}]

# 2 optim -- preprocessing
OP = [{'disp_imp': [0.7246763808878346], 'stat_par_diff': [-0.22006500541711815], 'group': [0.2363818197293387], 'causal': [0.1437125748502994], 'acc': [0.6647727272727273]},
{'disp_imp': [0.6938754578754579], 'stat_par_diff': [-0.22635969664138678], 'acc': [0.6704545454545454], 'group': [0.240750768665963], 'causal': [0.12974051896207583]},
{'disp_imp': [0.6803250225710119], 'stat_par_diff': [-0.24481438144878198], 'group': [0.25216375320181544], 'causal': [0.059880239520958084], 'acc': [0.6647727272727273]},
{'disp_imp': [0.6158226448474127], 'stat_par_diff': [-0.27719125624933516], 'acc': [0.6717171717171717], 'group': [0.270787103511291], 'causal': [0.06786427145708583]},
{'disp_imp': [0.7609608507662643], 'stat_par_diff': [-0.18520972249789447], 'acc': [0.6616161616161617], 'group': [0.4297453703703703], 'causal': [0.228]},
{'disp_imp': [0.7582321602376928], 'stat_par_diff': [-0.16780801217435204], 'group': [0.37670922934660706], 'causal': [0.1], 'acc': [0.6704545454545454]}]

# 3 disparate impact remover -- preprocessing (MinMaxScaler)
DI = [{'disp_imp': [0.8281294720316501], 'stat_par_diff': [-0.12970552920151934], 'group': [0.17330375090074715], 'acc': [0.7212962962962963]},
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

# basic remove sensitive features (test on test data)
basic = [{'disp_imp': [1.492077863286555], 'stat_par_diff': [0.11227019210906836], 'acc': [0.7766666666666666], 'group': [0.18008971011581976]},
         {'disp_imp': [1.195357833655706], 'stat_par_diff': [0.07302210287130756], 'acc': [0.71], 'group': [0.07600314712824546]},
         {'disp_imp': [2.419811320754717], 'stat_par_diff': [0.29890764647467727], 'acc': [0.7766666666666666], 'group': [0.1872295227310633]},
         {'disp_imp': [1.2890405459654757], 'stat_par_diff': [0.10999923611641588], 'acc': [0.6866666666666666], 'group': [0.09865540963101938]},
         {'disp_imp': [1.5865384615384615], 'stat_par_diff': [0.13903133903133902], 'acc': [0.7766666666666666], 'group': [0.31122967479674796]},
         {'disp_imp': [1.1096563011456628], 'stat_par_diff': [0.09544159544159547], 'acc': [0.4066666666666667], 'group': [0.17582417582417587]}]


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
    
all_uni_orig_metrics = [[{'disp_imp': [0.8541774139284513], 'stat_par_diff': [-0.09447108398721304], 'acc': [0.6862850971922246], 'group': [0.11328759703917674], 'causal': [0.2215568862275449]}, {'disp_imp': [0.7649488070563668], 'stat_par_diff': [-0.15980378134121598], 'acc': [0.6862850971922246], 'group': [0.17018034825870643], 'causal': [0.21157684630738524]}], [{'disp_imp': [0.7246763808878346], 'stat_par_diff': [-0.22006500541711815], 'group': [0.2363818197293387], 'causal': [0.1437125748502994], 'acc': [0.6647727272727273]}, {'disp_imp': [0.6803250225710119], 'stat_par_diff': [-0.24481438144878198], 'group': [0.25216375320181544], 'causal': [0.059880239520958084], 'acc': [0.6647727272727273]}], [{'disp_imp': [0.8281294720316501], 'stat_par_diff': [-0.12970552920151934], 'group': [0.17330375090074715], 'acc': [0.7212962962962963]}, {'disp_imp': [0.7577545900868846], 'stat_par_diff': [-0.18220997432342545], 'group': [0.1721174674194808], 'acc': [0.7256944444444444]}], [{'disp_imp': [0.7589346628181579], 'stat_par_diff': [-0.20023975588491727], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.2057257659467604], 'causal': [0.07984031936127745]}, {'disp_imp': [0.8154854153982398], 'stat_par_diff': [-0.14127849382048963], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.11579931566801943], 'causal': [0.021956087824351298]}], [{'disp_imp': [0.7589346628181579], 'stat_par_diff': [-0.20023975588491727], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.2057257659467604], 'causal': [0.07984031936127745]}, {'disp_imp': [0.8154854153982398], 'stat_par_diff': [-0.14127849382048963], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.11579931566801943], 'causal': [0.021956087824351298]}], [{'disp_imp': [0.7589346628181579], 'stat_par_diff': [-0.20023975588491727], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.2057257659467604], 'causal': [0.07984031936127745]}, {'disp_imp': [0.8154854153982398], 'stat_par_diff': [-0.14127849382048963], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.11579931566801943], 'causal': [0.021956087824351298]}], [{'disp_imp': [0.7589346628181579], 'stat_par_diff': [-0.20023975588491727], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.2057257659467604], 'causal': [0.07984031936127745]}, {'disp_imp': [0.8154854153982398], 'stat_par_diff': [-0.14127849382048963], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.11579931566801943], 'causal': [0.021956087824351298]}], [{'disp_imp': [0.770705089947992], 'stat_par_diff': [-0.1953937808776519], 'acc': [0.7289416846652268, 0.7289416846652268], 'group': [0.17722567287784677], 'causal': [0.059880239520958084]}, {'disp_imp': [0.8197130818619582], 'stat_par_diff': [-0.14280151931726082], 'acc': [0.7289416846652268, 0.7289416846652268], 'group': [0.1329751677353866], 'causal': [0.02594810379241517]}], [{'disp_imp': [0.7725038019123672], 'stat_par_diff': [-0.19334127290557535], 'acc': [0.7289416846652268, 0.7289416846652268], 'group': [0.17594612280510416], 'causal': [0.059880239520958084]}, {'disp_imp': [0.8197130818619582], 'stat_par_diff': [-0.14280151931726082], 'acc': [0.7289416846652268, 0.7289416846652268], 'group': [0.1329751677353866], 'causal': [0.02594810379241517]}], [{'disp_imp': [0.770705089947992], 'stat_par_diff': [-0.1953937808776519], 'acc': [0.7289416846652268, 0.7289416846652268], 'group': [0.17722567287784677], 'causal': [0.059880239520958084]}, {'disp_imp': [0.8197130818619582], 'stat_par_diff': [-0.14280151931726082], 'acc': [0.7289416846652268, 0.7289416846652268], 'group': [0.1329751677353866], 'causal': [0.02594810379241517]}]]
all_uni_trans_metrics = [[{'disp_imp': [0.9910050675675677], 'stat_par_diff': [-0.004642545771577966], 'acc': [0.6511879049676026], 'group': [0.03044979784366575], 'causal': [0.3393213572854291]}, {'disp_imp': [0.9196761423096833], 'stat_par_diff': [-0.05712802419888863], 'acc': [0.6927645788336934], 'group': [0.06429804372842352], 'causal': [0.16966067864271456]}], [{'disp_imp': [0.6938754578754579], 'stat_par_diff': [-0.22635969664138678], 'acc': [0.6704545454545454], 'group': [0.240750768665963], 'causal': [0.12974051896207583]}, {'disp_imp': [0.6158226448474127], 'stat_par_diff': [-0.27719125624933516], 'acc': [0.6717171717171717], 'group': [0.270787103511291], 'causal': [0.06786427145708583]}], [{'disp_imp': [0.8204363646829337], 'stat_par_diff': [-0.12813195894978702], 'group': [0.14348063284233498], 'acc': [0.7212962962962963]}, {'disp_imp': [0.7223832423774883], 'stat_par_diff': [-0.19453572394523366], 'group': [0.1836341873706004], 'acc': [0.7256944444444444]}], [{'disp_imp': [5.256521739130435], 'stat_par_diff': [0.2631720430107527], 'acc': [0.3050755939524838], 'group': [0.3014285714285714], 'causal': [0.20359281437125748]}, {'disp_imp': [4.562906756165183], 'stat_par_diff': [0.3233661247344638], 'acc': [0.31317494600431967], 'group': [0.32883501926239667], 'causal': [0.27944111776447106]}], [{'disp_imp': [4.033590733590734], 'stat_par_diff': [0.3425021795989538], 'acc': [0.7154427645788337], 'group': [0.388808252553004], 'causal': [0.2375249500998004]}, {'disp_imp': [1.5315693448849985], 'stat_par_diff': [0.14122551902060193], 'acc': [0.7208423326133909], 'group': [0.09468001988777608], 'causal': [0.05788423153692615]}], [{'disp_imp': [5.9371478476448045], 'stat_par_diff': [0.31767171137660943], 'acc': [0.3002159827213823], 'group': [0.34550449550449547], 'causal': [0.26147704590818366]}, {'disp_imp': [1.708083467094703], 'stat_par_diff': [0.14605682077035953], 'acc': [0.3169546436285097], 'group': [0.1485718431808838], 'causal': [0.08383233532934131]}], [{'disp_imp': [1.8748338502436865], 'stat_par_diff': [0.14345393780877655], 'acc': [0.6830453563714903], 'group': [0.12322288303596718], 'causal': [0.005988023952095809]}, {'disp_imp': [1.2645264847512039], 'stat_par_diff': [0.06547685266118908], 'acc': [0.6862850971922246], 'group': [0.03883973544741792], 'causal': [0.043912175648702596]}], [{'disp_imp': [1.508108108108108], 'stat_par_diff': [0.16117407730310956], 'acc': [0.9973002159827213], 'group': [0.17625290247678016], 'causal': [0.5528942115768463]}, {'disp_imp': [1.2334824630306342], 'stat_par_diff': [0.09054187922805124], 'acc': [1.0], 'group': [0.08242753623188404], 'causal': [0.5469061876247505]}], [{'disp_imp': [1.4379321250190844], 'stat_par_diff': [0.14558601475165273], 'acc': [0.8828293736501079], 'group': [0.15111871301775148], 'causal': [0.5469061876247505]}, {'disp_imp': [1.2334824630306342], 'stat_par_diff': [0.09054187922805124], 'acc': [0.9260259179265659], 'group': [0.08242753623188404], 'causal': [0.5469061876247505]}], [{'disp_imp': [1.4468029004614371], 'stat_par_diff': [0.1477332170880558], 'acc': [1.0], 'group': [0.15312036350148372], 'causal': [0.5469061876247505]}, {'disp_imp': [1.2334824630306342], 'stat_par_diff': [0.09054187922805124], 'acc': [1.0], 'group': [0.08242753623188404], 'causal': [0.5469061876247505]}]]
all_multi_orig_metrics = [{'disp_imp': [0.8442063492063492], 'stat_par_diff': [-0.10435938330675176], 'group': [0.22918730248573183], 'causal': [0.336], 'acc': [0.6862850971922246]}, {'disp_imp': [0.7609608507662643], 'stat_par_diff': [-0.18520972249789447], 'acc': [0.6616161616161617], 'group': [0.4297453703703703], 'causal': [0.228]}, {'disp_imp': [0.8491988243102303], 'stat_par_diff': [-0.11417803302225416], 'group': [0.2639003072130155], 'acc': [0.7206018518518519]}, {'disp_imp': [0.8432766615146832], 'stat_par_diff': [-0.12129186602870812], 'acc': [0.7294816414686826], 'group': [0.30939716312056736], 'causal': [0.066]}, {'disp_imp': [0.8432766615146832], 'stat_par_diff': [-0.12129186602870812], 'acc': [0.7294816414686826], 'group': [0.30939716312056736], 'causal': [0.066]}, {'disp_imp': [0.8432766615146832], 'stat_par_diff': [-0.12129186602870812], 'acc': [0.7294816414686826], 'group': [0.30939716312056736], 'causal': [0.066]}, {'disp_imp': [0.8432766615146832], 'stat_par_diff': [-0.12129186602870812], 'acc': [0.7294816414686826], 'group': [0.30939716312056736], 'causal': [0.066]}, {'disp_imp': [0.8489687292082502], 'stat_par_diff': [-0.12068048910154172], 'acc': [0.7289416846652268], 'group': [0.2984026902059689], 'causal': [0.046]}, {'disp_imp': [0.8489687292082502], 'stat_par_diff': [-0.12068048910154172], 'acc': [0.7289416846652268], 'group': [0.2984026902059689], 'causal': [0.046]}, {'disp_imp': [0.8489687292082502], 'stat_par_diff': [-0.12068048910154172], 'acc': [0.7289416846652268], 'group': [0.2984026902059689], 'causal': [0.046]}]
all_multi_trans_metrics = [{'disp_imp': [0.9441223832528179], 'stat_par_diff': [-0.03689526847421587], 'group': [0.15896586473306845], 'causal': [0.458], 'acc': [0.6744060475161987]}, {'disp_imp': [0.7582321602376928], 'stat_par_diff': [-0.16780801217435204], 'group': [0.37670922934660706], 'causal': [0.1], 'acc': [0.6704545454545454]}, {'disp_imp': [0.8377759126853437], 'stat_par_diff': [-0.1174469285201517], 'group': [0.08403361344537816], 'acc': [0.725462962962963]}, {'disp_imp': [2.272300469483568], 'stat_par_diff': [0.1080542264752791], 'acc': [0.3185745140388769], 'group': [0.24174496644295296], 'causal': [0.132]}, {'disp_imp': [1.9441860465116279], 'stat_par_diff': [0.1456937799043062], 'acc': [0.7181425485961123], 'group': [0.3022677564176355], 'causal': [0.096]}, {'disp_imp': [1.54676710608914], 'stat_par_diff': [0.11576289207868154], 'acc': [0.30561555075593955], 'group': [0.3707103825136612], 'causal': [0.23]}, {'disp_imp': [1.5940740740740742], 'stat_par_diff': [0.1172514619883041], 'acc': [0.6727861771058316], 'group': [0.3567313019390581], 'causal': [0.234]}, {'disp_imp': [1.23005698005698], 'stat_par_diff': [0.08585858585858586], 'acc': [1.0], 'group': [0.21969062377841064], 'causal': [0.546]}, {'disp_imp': [1.23005698005698], 'stat_par_diff': [0.08585858585858586], 'acc': [0.9190064794816415], 'group': [0.21969062377841064], 'causal': [0.546]}, {'disp_imp': [1.23005698005698], 'stat_par_diff': [0.08585858585858586], 'acc': [1.0], 'group': [0.21969062377841064], 'causal': [0.546]}]

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
# Plot.plot_abs_acc_multi_metric(all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics,all_multi_trans_metrics)
# 3 images: one for 'race', one for 'sex', one for 'race,sex'
# Plot.plot_abs_acc_individual_metric(all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics,all_multi_trans_metrics)
