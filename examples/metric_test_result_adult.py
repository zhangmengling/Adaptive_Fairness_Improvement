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

dataset_name = "Adult income"
# 1 reweighing -- preprocessing
# RW = [{'disp_imp': [1.3660295216052227], 'stat_par_diff': [0.053352586190530016], 'acc': [0.8168696898351929], 'group': [0.053154351853237525], 'causal': [0.08455317842153752]},
#       {'disp_imp': [1.1242760359414887], 'stat_par_diff': [0.027694309340850348], 'acc': [0.8157436789845429], 'group': [0.027448183855985714], 'causal': [0.1311290817893336]},
#       {'disp_imp': [0.42661035362176686], 'stat_par_diff': [-0.13478741156905721], 'acc': [0.8168696898351929], 'group': [0.1346398252910103], 'causal': [0.09734875626983315]},
#       {'disp_imp': [0.6267423928200627], 'stat_par_diff': [-0.08864726742209181], 'acc': [0.8207595455010749], 'group': [0.08848472850435313], 'causal': [0.1440270242604156]},
#       {'disp_imp': [0.5338347706781654], 'stat_par_diff': [-0.09242433295104528], 'group': [0.16252465813432443], 'causal': [0.13778278227044732], 'acc': [0.8168696898351929]},
#       {'disp_imp': [0.6792146101874698], 'stat_par_diff': [-0.06757560875852558], 'group': [0.23556044765500095], 'causal': [0.1814924762002252], 'acc': [0.8215784624833657]}]
RW = [{'disp_imp': [1.5927914689003158], 'stat_par_diff': [0.11880754036824828], 'average_odds_difference': [0.04903032702027723], 'generalized_entropy_index': [0.08507146858713717], 'accuracy': [0.8097041662401474], 'acc': [0.8097041662401474], 'group': [0.11858537550782874], 'causal': [0.11690039922202887]},
      {'disp_imp': [1.273866668026954], 'stat_par_diff': [0.05412081316300002], 'average_odds_difference': [-0.013765573361300097], 'generalized_entropy_index': [0.08966525973602832], 'accuracy': [0.8233186610707339], 'acc': [0.8233186610707339], 'group': [0.05389252266596359], 'causal': [0.11874296243218344]},
      {'disp_imp': [0.34843605687229356], 'stat_par_diff': [-0.24944688092461145], 'average_odds_difference': [-0.14120086231149304], 'generalized_entropy_index': [0.08507146858713717], 'accuracy': [0.8097041662401474], 'acc': [0.8097041662401474], 'group': [0.24931124892349024], 'causal': [0.10328590439144232]},
      {'disp_imp': [0.5521032223846793], 'stat_par_diff': [-0.11947534534752416], 'average_odds_difference': [-0.0016555020074790205], 'generalized_entropy_index': [0.09411564020410572], 'accuracy': [0.8199406285187839], 'acc': [0.8199406285187839], 'group': [0.11931773786861347], 'causal': [0.12212099498413348]},
      {'disp_imp': [0.504779029536584], 'stat_par_diff': [-0.14114104486565757], 'average_odds_difference': [-0.09459843684655957], 'generalized_entropy_index': [0.08507146858713717], 'accuracy': [0.8097041662401474], 'group': [0.30590455941808203], 'causal': [0.1794451837444979], 'acc': [0.8097041662401474]},
      {'disp_imp': [0.5912864475690675], 'stat_par_diff': [-0.09673393867944005], 'average_odds_difference': [-0.023121987211474647], 'generalized_entropy_index': [0.09354423994379656], 'accuracy': [0.8156413143617566], 'group': [0.22734199960379572], 'causal': [0.18783908281297984], 'acc': [0.8156413143617566]}]


# 2 optim -- preprocessing
OP = []

# 3 disparate impact remover -- preprocessing (MinMaxScaler)
DI = [{'disp_imp': [1.4443537679404828], 'stat_par_diff': [0.08222821383829307], 'group': [0.08248762798442608], 'acc': [0.805019305019305]},
      {'disp_imp': [1.6510037619530644], 'stat_par_diff': [0.11458294062103364], 'group': [0.1148462778042233], 'acc': [0.805019305019305]},
      {'disp_imp': [0.38441295546558707], 'stat_par_diff': [-0.16224397882542058], 'group': [0.16224799577748472], 'acc': [0.8162951912951913]},
      {'disp_imp': [0.4359988025193151], 'stat_par_diff': [-0.1696978326543056], 'group': [0.16970037590133286], 'acc': [0.8162951912951913]},
      {'disp_imp': [0.5281796167221384], 'stat_par_diff': [-0.09180805688988793], 'group': [0.20962766406594707], 'acc': [0.8208143208143208]},
      {'disp_imp': [0.5933052324578383], 'stat_par_diff': [-0.09734595596780321], 'group': [0.24666532144410303], 'acc': [0.8015970515970516]}]
# DI = [{'disp_imp': [1.7918522841097433], 'stat_par_diff': [0.11407636883799194], 'average_odds_difference': [0.08578843870985763], 'generalized_entropy_index': [0.09541610870642307], 'accuracy': [0.8071253071253072], 'group': [0.1143476707101897], 'causal': [0.24877149877149876]},
#       {'disp_imp': [1.8759192315712574], 'stat_par_diff': [0.13014713115269133], 'average_odds_difference': [0.08693723949450816], 'generalized_entropy_index': [0.0963996944157489], 'accuracy': [0.7974727974727974], 'group': [0.13041812754937956], 'causal': [0.19164619164619165]},
#       {'disp_imp': [0.2864535169699951], 'stat_par_diff': [-0.22615365427787476], 'average_odds_difference': [-0.14336420337959502], 'generalized_entropy_index': [0.09541610870642307], 'accuracy': [0.8071253071253072], 'group': [0.22616257059763528], 'causal': [0.26013513513513514]},
#       {'disp_imp': [0.34071622140773117], 'stat_par_diff': [-0.21997824339680166], 'average_odds_difference': [-0.12491638030169705], 'generalized_entropy_index': [0.09193740459024205], 'accuracy': [0.8065988065988066], 'group': [0.2199852474620897], 'causal': [0.19335731835731837]},
#       {'disp_imp': [0.4671660003461021], 'stat_par_diff': [-0.10898207842561759], 'average_odds_difference': [-0.052876906561096323], 'generalized_entropy_index': [0.09541610870642307], 'accuracy': [0.8071253071253072], 'group': [0.26054456935104636]},
#       {'disp_imp': [0.7199172549663438], 'stat_par_diff': [-0.05806027052936488], 'average_odds_difference': [0.008891838111593657], 'generalized_entropy_index': [0.09101159251305388], 'accuracy': [0.7960249210249211], 'group': [0.3235761090058422]}]

# 4 meta --inprocessing
# META =[{'disp_imp': [1.5927914689003158], 'stat_par_diff': [0.11880754036824828], 'acc': [0.8097041662401474], 'group': [0.11858537550782874], 'causal': [0.11690039922202887]},
#        {'disp_imp': [0.8889226783092952], 'stat_par_diff': [-0.06655298531573417], 'acc': [0.6669055174531682], 'group': [0.06702918868638041], 'causal': [0.2037055993448664]},
#        {'disp_imp': [0.34843605687229356], 'stat_par_diff': [-0.24944688092461145], 'acc': [0.8097041662401474], 'group': [0.24931124892349024], 'causal': [0.10328590439144232]},
#        {'disp_imp': [0.7058722893806472], 'stat_par_diff': [-0.18357349383608595], 'acc': [0.6448971235540997], 'group': [0.18337754136403644], 'causal': [0.17719316204319788]},
#        {'disp_imp': [0.504779029536584], 'stat_par_diff': [-0.14114104486565757], 'acc': [0.8097041662401474], 'group': [0.30590455941808203], 'causal': [0.1794451837444979]},
#        {'disp_imp': [0.47739567371084424], 'stat_par_diff': [-0.28493916179334394], 'acc': [0.6739686764254273], 'group': [0.4935708884740835], 'causal': [0.01709489200532296]}]
META = [{'disp_imp': [1.5927914689003158], 'stat_par_diff': [0.11880754036824828], 'average_odds_difference': [0.04903032702027723], 'generalized_entropy_index': [0.08507146858713717], 'accuracy': [0.8097041662401474], 'acc': [0.8097041662401474, 0.8097041662401474], 'group': [0.11858537550782874], 'causal': [0.11690039922202887]},
        {'disp_imp': [1.5623671381762967], 'stat_par_diff': [0.02364542977615823], 'average_odds_difference': [0.005380080361359886], 'generalized_entropy_index': [0.14016271749043843], 'accuracy': [0.7799160610093152], 'acc': [0.7799160610093152], 'group': [0.02350393038124421], 'causal': [0.005732418876036442]},
        {'disp_imp': [0.34843605687229356], 'stat_par_diff': [-0.24944688092461145], 'average_odds_difference': [-0.14120086231149304], 'generalized_entropy_index': [0.08507146858713717], 'accuracy': [0.8097041662401474], 'acc': [0.8097041662401474, 0.8097041662401474], 'group': [0.24931124892349024], 'causal': [0.10328590439144232]},
        {'disp_imp': [0.2671756568218547], 'stat_par_diff': [-0.057088771202419925], 'average_odds_difference': [-0.041481613623973984], 'generalized_entropy_index': [0.14032205967022274], 'accuracy': [0.7799160610093152], 'acc': [0.7799160610093152], 'group': [0.05694243009799406], 'causal': [0.006756065103900092]},
        {'disp_imp': [0.504779029536584], 'stat_par_diff': [-0.14114104486565757], 'average_odds_difference': [-0.09459843684655957], 'generalized_entropy_index': [0.08507146858713717], 'accuracy': [0.8097041662401474], 'acc': [0.8097041662401474], 'group': [0.30590455941808203], 'causal': [0.1794451837444979]},
        {'disp_imp': [0.38752450019600154], 'stat_par_diff': [-0.03035814125422545], 'average_odds_difference': [-0.033340144966399324], 'generalized_entropy_index': [0.14152992467921788], 'accuracy': [0.7785853209130924], 'acc': [0.7785853209130924], 'group': [0.05729553997764909], 'causal': [0.006653700481113727]}]

# 5 adversarial debias --inprocessing  (maxabsscaler)
# AD = [{'disp_imp': [1.5414048526223376], 'stat_par_diff': [0.10547340506587935], 'acc': [0.8078616030299929], 'group': [0.10525290085155317], 'causal': [0.17982456140350878]},
#       {'disp_imp': [1.5235930632142574], 'stat_par_diff': [0.05503781323205227], 'acc': [0.8286416214556249], 'group': [0.0548634106551846], 'causal': [0.024875621890547265]},
#       {'disp_imp': [0.273674566474341], 'stat_par_diff': [-0.22765095430904259], 'acc': [0.8241375780530249], 'group': [0.2275198218863464], 'causal': [0.19421487603305784]},
#       {'disp_imp': [1.4010061762468449], 'stat_par_diff': [0.04260462709139712], 'acc': [0.8115467294503019], 'group': [0.04278706170331702], 'causal': [0.1826086956521739]},
#       {'disp_imp': [0.46424880489953013], 'stat_par_diff': [-0.15335629004734638], 'acc': [0.8134916572832429], 'group': [0.24029285632573572], 'causal': [0.32941176470588235]},
#       {'disp_imp': [0.27823249092016833], 'stat_par_diff': [-0.14846766605607442], 'acc': [0.8236257549390931], 'group': [0.1607210861992247], 'causal': [0.105]}]
# 5 adversarial debias --inprocessing
AD = [{'disp_imp': [1.7098804256560192], 'stat_par_diff': [0.1149140703059148], 'acc': [0.8182004299314157], 'group': [0.11471384615074184], 'causal': [0.11377245508982035]},
      {'disp_imp': [1.2244444724854573], 'stat_par_diff': [0.026895588503863477], 'acc': [0.8307912785341386], 'group': [0.026709256376139565], 'causal': [0.0499001996007984]},
      {'disp_imp': [0.3228328132693765], 'stat_par_diff': [-0.18720111376123938], 'acc': [0.8248541304125294], 'group': [0.18706328132135386], 'causal': [0.09780439121756487]},
      {'disp_imp': [0.6884012608041471], 'stat_par_diff': [-0.05695484684301283], 'acc': [0.8248541304125294], 'group': [0.05679127640945675], 'causal': [0.05389221556886228]},
      {'disp_imp': [0.504779029536584], 'stat_par_diff': [-0.14114104486565757], 'acc': [0.8097041662401474], 'group': [0.30590455941808203], 'causal': [0.166]},
      {'disp_imp': [0.26197447079576636], 'stat_par_diff': [-0.11705981132359591], 'acc': [0.8290510799467704], 'group': [0.13649700294625622], 'causal': [0.086]}]

# 6 prejudice remover --inprocssing (MaxAbsScaler)
# PR = [{'disp_imp': [1.3541083143105657], 'stat_par_diff': [0.07389655298637857], 'acc': [0.8134916572832429, 0.8134916572832429], 'group': [0.0736642921122107], 'causal': [0.19421487603305784]},
#       {'disp_imp': [100], 'stat_par_diff': [0.0011988970147464332], 'acc': [0.7492066741734057], 'group': [0.001199040767386091], 'causal': [0.0]},
#       {'disp_imp': [0.3754480764044213], 'stat_par_diff': [-0.21291758013740436], 'acc': [0.8134916572832429, 0.8134916572832429], 'group': [0.21277729361697573], 'causal': [0.1991869918699187]},
#       {'disp_imp': [3.0251060414994844], 'stat_par_diff': [0.0033767116047437205], 'acc': [0.7486948510594739], 'group': [0.003378049519801093], 'causal': [0.0]},
#       {'disp_imp': [0.46424880489953013], 'stat_par_diff': [-0.15335629004734638], 'acc': [0.8134916572832429], 'group': [0.24029285632573572], 'causal': [0.32941176470588235]},
#       {'disp_imp': [100], 'stat_par_diff': [0.0027440219521756176], 'acc': [0.7491043095506194], 'group': [0.004846526655896607], 'causal': [0.0]}]
# 6 prejudice remover --inprocssing
PR = [{'disp_imp': [1.5935001337231285], 'stat_par_diff': [0.11886627328068261], 'average_odds_difference': [0.05042386170454762], 'generalized_entropy_index': [0.08513912030019033], 'acc': [0.8097041662401474, 0.8097041662401474], 'group': [0.11864428569267924], 'causal': [0.11776447105788423]},
      {'disp_imp': [1.646656712188423], 'stat_par_diff': [0.04981249183524267], 'average_odds_difference': [0.00487654033644235], 'generalized_entropy_index': [0.1195318843965374], 'acc': [0.8024362780223155], 'group': [0.049653815840109986], 'causal': [0.017964071856287425]},
      {'disp_imp': [0.34857405135026276], 'stat_par_diff': [-0.2492953198454966], 'average_odds_difference': [-0.1409588487587341], 'generalized_entropy_index': [0.08513912030019033], 'acc': [0.8097041662401474, 0.8097041662401474], 'group': [0.24915966487013266], 'causal': [0.09381237524950099]},
      {'disp_imp': [0.29485744712208156], 'stat_par_diff': [-0.10633780541278097], 'average_odds_difference': [-0.052659539822952996], 'generalized_entropy_index': [0.12040188608904483], 'acc': [0.8016173610400246], 'group': [0.10619505376930247], 'causal': [0.021956087824351298]},
      {'disp_imp': [0.504779029536584], 'stat_par_diff': [-0.14114104486565757], 'average_odds_difference': [-0.09459843684655957], 'generalized_entropy_index': [0.08507146858713717], 'acc': [0.8097041662401474], 'group': [0.30590455941808203], 'causal': [0.166]},
      {'disp_imp': [0.47941175281979576], 'stat_par_diff': [-0.06257380418398985], 'average_odds_difference': [-0.04847382746761579], 'generalized_entropy_index': [0.11893909821080248], 'acc': [0.8030504657590337], 'group': [0.11066319569420846], 'causal': [0.008]}]

# 7 gradient reduction -- DemographicParity (MaxAbsScaler)
# GR = [{'disp_imp': [1.3535718825575762], 'stat_par_diff': [0.0738363146476228], 'acc': [0.8134916572832429], 'group': [0.07360386929298537], 'causal': [0.1791380898761388]},
#       {'disp_imp': [100], 'stat_par_diff': [0.001078877966914409], 'acc': [0.7482853925683284], 'group': [0.00107900731327179], 'causal': [0.0009212816050772853]},
#       {'disp_imp': [0.37545646308560215], 'stat_par_diff': [-0.2129771079201872], 'acc': [0.8134916572832429], 'group': [0.21283682622203287], 'causal': [0.18835090592691167]},
#       {'disp_imp': [0.0], 'stat_par_diff': [-0.0001515610791148833], 'acc': [0.7486948510594739], 'group': [0.00015158405335758679], 'causal': [0.002866209438018221]},
#       {'disp_imp': [0.46424880489953013], 'stat_par_diff': [-0.15335629004734638], 'acc': [0.8134916572832429], 'group': [0.24029285632573572], 'causal': [0.2844712867233084]},
#       {'disp_imp': [-100], 'stat_par_diff': [0.0], 'acc': [0.7486948510594739], 'group': [0.00017271157167530224], 'causal': [0.00010236462278636504]}]
GR = [{'disp_imp': [1.5927914689003158], 'stat_par_diff': [0.11880754036824828], 'average_odds_difference': [0.04903032702027723], 'generalized_entropy_index': [0.08507146858713717], 'acc': [0.8097041662401474], 'group': [0.11858537550782874], 'causal': [0.11776447105788423]},
      {'disp_imp': [1.2854813347060232], 'stat_par_diff': [0.02720775159076326], 'average_odds_difference': [-0.035395619322784594], 'generalized_entropy_index': [0.11908952221681239], 'acc': [0.80202681953117], 'group': [0.02703571620140327], 'causal': [0.013972055888223553]},
      {'disp_imp': [0.34843605687229356], 'stat_par_diff': [-0.24944688092461145], 'average_odds_difference': [-0.14120086231149304], 'generalized_entropy_index': [0.08507146858713717], 'acc': [0.8097041662401474], 'group': [0.24931124892349024], 'causal': [0.09381237524950099]},
      {'disp_imp': [0.9882102730134322], 'stat_par_diff': [-0.0013544427183719843], 'average_odds_difference': [0.15358088439531875], 'generalized_entropy_index': [0.1181062044875336], 'acc': [0.8052001228375474], 'group': [0.0011844596219771075], 'causal': [0.15169660678642716]},
      {'disp_imp': [0.504779029536584], 'stat_par_diff': [-0.14114104486565757], 'average_odds_difference': [-0.09459843684655957], 'generalized_entropy_index': [0.08507146858713717], 'acc': [0.8097041662401474], 'group': [0.30590455941808203], 'causal': [0.166]},
      {'disp_imp': [0.767883886660837], 'stat_par_diff': [-0.033652522008280145], 'average_odds_difference': [0.0453878916214376], 'generalized_entropy_index': [0.1197949837750279], 'acc': [0.8029481011362473], 'group': [0.03378874130297281], 'causal': [0.12]}]

# 9 calibrated EO -- postprocessing  (Standard)
# CEO = [{'disp_imp': [1.8341990566600992], 'stat_par_diff': [0.08072894096710637], 'acc': [0.8306463995086749], 'group': [0.08065656988343665], 'causal': [0.11442786069651742]},
#        {'disp_imp': [0.4190], 'stat_par_diff': [0.0698], 'acc': [0.8268079226163059], 'group': [0.06972858708157352], 'causal': [0.13930348258706468]},
#        {'disp_imp': [0.30532577185206156], 'stat_par_diff': [-0.1500010188772482], 'acc': [0.8306463995086749], 'group': [0.15002032791453865], 'causal': [0.11442786069651742]},
#        {'disp_imp': [0.5266], 'stat_par_diff': [ -0.0733], 'acc': [0.8096115461384923], 'group': [0.07334521854313639], 'causal': [0.09950248756218906]},
#        {'disp_imp': [0.49379793995178617], 'stat_par_diff': [-0.07651891605379976], 'acc': [0.8306463995086749], 'group': [0.19566341105095628], 'causal': [0.115]},
#        {'disp_imp': [0.3982], 'stat_par_diff': [-0.0494], 'acc': [0.8096115461384923], 'group': [0.12427184466019417], 'causal': [0.02]}]
# CEO (no scaler)
CEO = [{},
       {'disp_imp': [0.5664173497419002], 'stat_par_diff': [-0.09352369781119695], 'average_odds_difference': [0.022735918491612703], 'generalized_entropy_index': [0.11786355133584864], 'acc': [0.7893443881467833], 'group':[0.09351700960021343], 'causal':[0.16367265469061876]},
       {},
       {'disp_imp': [1.5764944702581012], 'stat_par_diff': [0.09484263865536505], 'average_odds_difference': [0.03817147727327591], 'generalized_entropy_index': [0.09962546580388996], 'acc': [0.8017810532780593], 'group':[0.09471201258543241], 'causal':[0.2215568862275449]},
       {},
       {'disp_imp': [0.7637685430788879], 'stat_par_diff': [-0.03982972238786192], 'average_odds_difference': [0.03256726270099458], 'generalized_entropy_index': [0.1605865356027947], 'acc': [0.7544910179640718], 'group':[0.16893203883495145], 'causal':[0.115]}
       ]

# 9 eqodds EO -- postprocessing  (Standard)
# EO = [{'disp_imp': [1.8341990566600992], 'stat_par_diff': [0.08072894096710637], 'acc': [0.8306463995086749], 'group': [0.08065656988343665], 'causal': [0.11442786069651742]},
#       {'disp_imp': [0.4548], 'stat_par_diff': [0.0807], 'acc': [0.8197451251343467], 'group': [0.08065656988343665], 'causal': [0.15942028985507245]},
#       {'disp_imp': [0.30532577185206156], 'stat_par_diff': [-0.1500010188772482], 'acc': [0.8306463995086749], 'group': [0.15002032791453865], 'causal': [0.11442786069651742]},
#       {'disp_imp': [ 0.6947], 'stat_par_diff': [ -0.1500], 'acc': [1.0], 'group': [0.809765085214187], 'causal': [0.15942028985507245]},
#       {'disp_imp': [0.49379793995178617], 'stat_par_diff': [-0.07651891605379976], 'acc': [0.8306463995086749], 'group': [0.19566341105095628], 'causal': [0.115]},
#       {'disp_imp': [ 0.5062], 'stat_par_diff': [-0.076], 'acc': [0.8023339133995291], 'group': [0.19566341105095628], 'causal': [0.15942028985507245]}]

EO = [{},
      {'disp_imp': [0.0], 'stat_par_diff': [-0.3229742173112339], 'average_odds_difference': [-0.4069501160135601], 'generalized_entropy_index': [0.10557324322086668], 'acc': [0.8014739751266697], 'group':[0.32304858392816027], 'causal':[0.20159680638722555]},
      {},
      {'disp_imp': [4.351108969964239], 'stat_par_diff': [0.20899389275045788], 'average_odds_difference': [0.26992028164099613], 'generalized_entropy_index': [0.10171460640252054], 'acc': [0.799324428066943], 'group':[0.20897537409765898], 'causal':[0.22554890219560877]},
      {},
      {'disp_imp': [0.03868645973909132], 'stat_par_diff': [-0.212383224011131], 'average_odds_difference': [-0.3267346067254254], 'generalized_entropy_index': [0.16367090830256806], 'acc': [0.7529556272071242], 'group':[0.22135922330097088], 'causal':[0.15942028985507245]}
      ]

# 10 reject_option -- postprocessing (Standard)
RO = [{'disp_imp': [1.4641376412070577], 'stat_par_diff': [0.0863629343268845], 'acc': [0.7785966528481498], 'group': [0.08729827628450909], 'causal': [0.2534246575342466]},
      {'disp_imp': [0.1868], 'stat_par_diff': [0.0345], 'acc': [0.805926608321818], 'group': [0.03436835727198681], 'causal': [0.19672131147540983]},
      {'disp_imp': [0.39436888728935027], 'stat_par_diff': [-0.21816042831831836], 'acc': [0.7518935516888434], 'group': [0.21818233902312797], 'causal': [0.275974025974026]},
      {'disp_imp': [0.0130], 'stat_par_diff': [-0.0039], 'acc': [0.7645854657113613], 'group': [0.0037746680447230396], 'causal': [0.28434504792332266]},
      {'disp_imp': [0.48378212974296203], 'stat_par_diff': [-0.1346215230278158], 'acc': [0.7785966528481498], 'group': [0.2443272429033581], 'causal': [0.25597269624573377]},
      {'disp_imp': [0.2947], 'stat_par_diff': [-0.0809], 'acc': [0.7637033625057578], 'group': [0.3754993089412372], 'causal': [0.4128686327077748]}]


# basic remove sensitive features (test on test data)
basic = []

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


for i in range(0, len(metrics)):
    all_uni_orig_metrics[i][0] = RW[0]
    all_uni_orig_metrics[i][1] = RW[2]
    all_multi_orig_metrics[i] = RW[4]

# all_uni_orig_metrics = []
#
# all_uni_trans_metrics = []
#
# all_multi_orig_metrics = []
#
# all_multi_trans_metrics = []

print("-->all_uni_orig_metrics", all_uni_orig_metrics)
print("-->all_uni_trans_metrics", all_uni_trans_metrics)
print("-->all_multi_orig_metrics", all_multi_orig_metrics)
print("-->all_multi_trans_metrics", all_multi_trans_metrics)

processing_names = ["RW", "DI", "META", "AD", "PR", "GR", "CEO", "EO", "RO"]
sens_attrs = ["race", "sex"]

for i in range(0, len(processing_names)):
    process_name = processing_names[i]
    print("-->process_name", process_name)
    uni_orig_metrics = all_uni_orig_metrics[i]
    uni_trans_metrics = all_uni_trans_metrics[i]
    multi_orig_metrics = all_multi_orig_metrics[i]
    multi_trans_metrics = all_multi_trans_metrics[i]
    print("group metric")
    percent = format((uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0])/float(uni_orig_metrics[1]['group'][0]), '.0%')
    print(str(round(uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0], 3)))
    percent = format((uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0]) / float(uni_orig_metrics[0]['group'][0]),'.0%')
    print(str(round(uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0], 3)))
    percent = format((multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0]) / float(multi_orig_metrics['group'][0]),'.0%')
    print(str(round(multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0], 3)))

    print(str(round(uni_trans_metrics[1]['acc'][0] - uni_orig_metrics[1]['acc'][0], 3)))
    print(str(round(uni_trans_metrics[0]['acc'][0] - uni_orig_metrics[0]['acc'][0], 3)))
    print(str(round(multi_trans_metrics['acc'][0] - multi_orig_metrics['acc'][0], 3)))

    try:
        print("causal metric")
        # percent = format((uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0]) / float(uni_orig_metrics[1]['causal'][0]), '.0%')
        percent = ''
        print(str(round(uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0], 3)) + "" + str(percent) + "")
        # percent = format((uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0]) / float(uni_orig_metrics[0]['causal'][0]), '.0%')
        percent = ''
        print(str(round(uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0], 3)) + "" + str(percent) + "")
        # percent = format((multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0]) / float(multi_orig_metrics['causal'][0]), '.0%')
        percent = ''
        print(str(round(multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0], 3)) + "" + str(percent) + "")

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