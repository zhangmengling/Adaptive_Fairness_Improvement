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
# banch_size=128, epoch=100
RW = [{'disp_imp': [0.7139676380110237], 'stat_par_diff': [-0.22730038366057315], 'average_odds_difference': [-0.1836198696658397], 'generalized_entropy_index': [0.11511141941222001], 'accuracy': [0.7300215982721382], 'acc': [0.7300215982721382], 'group': [0.2263669695520485], 'causal': [0.0658682634730539]},
      {'disp_imp': [0.8459794981381178], 'stat_par_diff': [-0.10514466260437827], 'average_odds_difference': [-0.05954290187859396], 'generalized_entropy_index': [0.124703825044343], 'accuracy': [0.7176025917926566], 'acc': [0.7176025917926566], 'group': [0.10390490268539049], 'causal': [0.11377245508982035]},
      {'disp_imp': [0.7926247763700756], 'stat_par_diff': [-0.1473282838763229], 'average_odds_difference': [-0.10599284369662185], 'generalized_entropy_index': [0.11511141941222001], 'accuracy': [0.7300215982721382], 'acc': [0.7300215982721382], 'group': [0.14640744992453092], 'causal': [0.021956087824351298]},
      {'disp_imp': [0.9200880841693175], 'stat_par_diff': [-0.05083004772774424], 'average_odds_difference': [-0.007252926274285171], 'generalized_entropy_index': [0.12241937970938294], 'accuracy': [0.7203023758099352], 'acc': [0.7203023758099352], 'group': [0.050593547229614566], 'causal': [0.10179640718562874]},
      {'disp_imp': [0.6307309023845732], 'stat_par_diff': [-0.30124584279153244], 'average_odds_difference': [-0.2454437370563614], 'generalized_entropy_index': [0.11511141941222001], 'accuracy': [0.7300215982721382], 'group': [0.2995092954599856], 'causal': [0.066], 'acc': [0.7300215982721382]},
      {'disp_imp': [0.8509610464045196], 'stat_par_diff': [-0.09609090429182288], 'average_odds_difference': [-0.05462692901907534], 'generalized_entropy_index': [0.12673370390332814], 'accuracy': [0.7084233261339092], 'group': [0.18954195159014436], 'causal': [0.126], 'acc': [0.7084233261339092]}]

# 3 disparate impact remover -- preprocessing (MinMaxScaler)
DI = [{'disp_imp': [0.854066985645933], 'stat_par_diff': [-0.10397727272727275], 'average_odds_difference': [-0.054739538313344466], 'generalized_entropy_index': [0.11607532889567031], 'group': [0.10344452304511076], 'acc': [0.7185185185185186]},
      {'disp_imp': [0.8314959294436907], 'stat_par_diff': [-0.11289772727272729], 'average_odds_difference': [-0.058877546281963955], 'generalized_entropy_index': [0.13529616533816397], 'group': [0.11357796279165383], 'acc': [0.7185185185185186]},
      {'disp_imp': [0.7718697921089825], 'stat_par_diff': [-0.16857752275835214], 'average_odds_difference': [-0.135824211277781], 'generalized_entropy_index': [0.11607532889567031], 'group': [0.1681996675074524], 'acc': [0.7180555555555556]},
      {'disp_imp': [0.7479104872872578], 'stat_par_diff': [-0.18148388440706587], 'average_odds_difference': [-0.15034510151743868], 'generalized_entropy_index': [0.12817170690725557], 'group': [0.18110429565084457], 'acc': [0.7180555555555556]},
      {'disp_imp': [0.7126078678634128], 'stat_par_diff': [-0.22207573846918105], 'average_odds_difference': [-0.16681436120658363], 'generalized_entropy_index': [0.11607532889567031], 'group': [0.2211533791262868], 'acc': [0.7219907407407408]},
      {'disp_imp': [0.7204833336470233], 'stat_par_diff': [-0.22700141388665984], 'average_odds_difference': [-0.1771243069130387], 'generalized_entropy_index': [0.11236887177699081], 'group': [0.2261842983171473], 'acc': [0.7090277777777778]}]

# 4 meta --inprocessing (maxabsscaler)
# META = [{'disp_imp': [0.712286920494222], 'stat_par_diff': [-0.22479981945384786], 'average_odds_difference': [-0.17907800238866955], 'generalized_entropy_index': [0.11849646161516098], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.22383809399591326], 'causal': [0.0755939524838013]},
#         {'disp_imp': [4.062288422477996], 'stat_par_diff': [0.36747461069735954], 'average_odds_difference': [0.3393405050457564], 'generalized_entropy_index': [0.45964803649991864], 'acc': [0.2845572354211663], 'group': [0.36680651565873945], 'causal': [0.23272138228941686]},
#         {'disp_imp': [0.7655373406193078], 'stat_par_diff': [-0.16694334924258147], 'average_odds_difference': [-0.1229905031593839], 'generalized_entropy_index': [0.11849646161516098], 'acc': [0.7294816414686826, 0.7294816414686826], 'group': [0.16603981596200673], 'causal': [0.03509719222462203]},
#         {'disp_imp': [6.812131147540984], 'stat_par_diff': [0.36785640174310025], 'average_odds_difference': [0.3520529410568475], 'generalized_entropy_index': [0.5465124166108078], 'acc': [0.30561555075593955], 'group': [0.3672894438167993], 'causal': [0.26457883369330454]},
#         {'disp_imp': [0.6135479609560388], 'stat_par_diff': [-0.31272105791057386], 'average_odds_difference': [-0.2549264482165726], 'generalized_entropy_index': [0.11849646161516098], 'acc': [0.7294816414686826], 'group': [0.3109590680603208], 'causal': [0.09287257019438445]},
#         {'disp_imp': [9.33801404212638], 'stat_par_diff': [0.43884284432244103], 'average_odds_difference': [0.4107176572069502], 'generalized_entropy_index': [0.529705303505907], 'acc': [0.28725701943844495], 'group': [0.43798372297135557], 'causal': [0.21436285097192226]}]
META = [{'disp_imp': [0.7139676380110237], 'stat_par_diff': [-0.22730038366057315], 'average_odds_difference': [-0.1836198696658397], 'generalized_entropy_index': [0.11511141941222001], 'accuracy': [0.7300215982721382], 'acc': [0.7300215982721382, 0.7300215982721382], 'group': [0.2263669695520485], 'causal': [0.0642548596112311]},
        {'disp_imp': [2.280942188830891], 'stat_par_diff': [0.2117824418867073], 'average_odds_difference': [0.1772287017009379], 'generalized_entropy_index': [0.539730560468887], 'accuracy': [0.2780777537796976], 'acc': [0.2780777537796976], 'group': [0.21159587264321844], 'causal': [0.05075593952483801]},
        {'disp_imp': [0.7926247763700756], 'stat_par_diff': [-0.1473282838763229], 'average_odds_difference': [-0.10599284369662185], 'generalized_entropy_index': [0.11511141941222001], 'accuracy': [0.7300215982721382], 'acc': [0.7300215982721382, 0.7300215982721382], 'group': [0.14640744992453092], 'causal': [0.017818574514038878]},
        {'disp_imp': [1.7126798260287721], 'stat_par_diff': [0.16576571902884416], 'average_odds_difference': [0.12977015429378433], 'generalized_entropy_index': [0.5311396092100674], 'accuracy': [0.2791576673866091], 'acc': [0.2791576673866091], 'group': [0.16572389880770522], 'causal': [0.004859611231101512]},
        {'disp_imp': [0.6307309023845732], 'stat_par_diff': [-0.30124584279153244], 'average_odds_difference': [-0.2454437370563614], 'generalized_entropy_index': [0.11511141941222001], 'accuracy': [0.7300215982721382], 'acc': [0.7300215982721382], 'group': [0.2995092954599856], 'causal': [0.06803455723542116]},
        {'disp_imp': [3.285456369107322], 'stat_par_diff': [0.300717943303595], 'average_odds_difference': [0.26153378302677], 'generalized_entropy_index': [0.5384883772448015], 'accuracy': [0.2775377969762419], 'acc': [0.2775377969762419], 'group': [0.30028059256895134], 'causal': [0.04697624190064795]}]

# 5 adversarial debias --inprocessing  (maxabsscaler)
AD = [{'disp_imp': [0.7428086515764242], 'stat_par_diff': [-0.2030097043556759], 'average_odds_difference': [-0.1638088416059689], 'generalized_entropy_index': [0.1125839397017497], 'acc': [0.7267818574514039, 0.7267818574514039], 'group': [0.2020491862672637], 'causal': [0.07451403887688984]},
      {'disp_imp': [2.5727826675693977], 'stat_par_diff': [0.2516452268111036], 'average_odds_difference': [0.21774079534383128], 'generalized_entropy_index': [0.5158849189610489], 'acc': [0.7257019438444925], 'group': [0.25149631175456144], 'causal': [0.09071274298056156]},
      {'disp_imp': [0.7976590869028732], 'stat_par_diff': [-0.14887424776924674], 'average_odds_difference': [-0.10811515400369803], 'generalized_entropy_index': [0.10993109995464187], 'acc': [0.7262419006479481, 0.7262419006479481], 'group': [0.14879437953480873], 'causal': [0.03023758099352052]},
      {'disp_imp': [1.261991767169784], 'stat_par_diff': [0.09410147333471675], 'average_odds_difference': [0.04965052754755936], 'generalized_entropy_index': [0.45705370920998134], 'acc': [0.7224622030237581], 'group': [0.09308375444786654], 'causal': [0.07019438444924406]},
      {'disp_imp': [0.6218340059548725], 'stat_par_diff': [-0.3159676397613894], 'average_odds_difference': [-0.26449661413163345], 'generalized_entropy_index': [0.11471767044752815], 'acc': [0.724622030237581], 'group': [0.31435676480757463], 'causal': [0.08099352051835854]},
      {'disp_imp': [1.8972472974478993], 'stat_par_diff': [0.21250593886923932], 'average_odds_difference': [0.14887385496344233], 'generalized_entropy_index': [0.5158754761953972], 'acc': [0.7300215982721382], 'group': [0.21038458469640153], 'causal': [0.0847732181425486]}]

# 6 prejudice remover --inprocssing (MaxAbsScaler)
PR = [{'disp_imp': [0.7299188816803992], 'stat_par_diff': [-0.2133353514386208], 'average_odds_difference': [-0.1698906752626613], 'generalized_entropy_index': [0.1157110571293651], 'acc': [0.7235421166306696, 0.7235421166306696], 'group': [0.21306214689265535], 'causal': [0.06371490280777538]},
      {'disp_imp': [2.3906608296852196], 'stat_par_diff': [0.24040679236579596], 'average_odds_difference': [0.2035673143847485], 'generalized_entropy_index': [0.5116237724784383], 'acc': [0.27429805615550756], 'group': [0.240225988700565], 'causal': [0.12904967602591794]},
      {'disp_imp': [0.7882459882459882], 'stat_par_diff': [-0.15269108613007187], 'average_odds_difference': [-0.11006632761496238], 'generalized_entropy_index': [0.1157900000000001], 'acc': [0.7235421166306696, 0.7235421166306696], 'group': [0.15260213374967474], 'causal': [0.029697624190064796]},
      {'disp_imp': [1.3147733294792119], 'stat_par_diff': [0.10176507006934898], 'average_odds_difference': [0.054494326774388285], 'generalized_entropy_index': [0.48973632842457415], 'acc': [0.26997840172786175], 'group': [0.10160031225605004], 'causal': [0.09017278617710583]},
      {'disp_imp': [0.6354423269809428], 'stat_par_diff': [-0.29980071794330354], 'average_odds_difference': [-0.24335343295575013], 'generalized_entropy_index': [0.1157110571293651], 'acc': [0.7240820734341252], 'group': [0.29910369956647787], 'causal': [0.10313174946004319]},
      {'disp_imp': [1.6212540059201996], 'stat_par_diff': [0.16757509370215912], 'average_odds_difference': [0.1223736377549362], 'generalized_entropy_index': [0.5201483985966501], 'acc': [0.2786177105831533], 'group': [0.19901226527732552], 'causal': [0.2602591792656587]}]

# 7 gradient reduction -- DemographicParity (MaxAbsScaler)
GR = [{'disp_imp': [0.718227588118488], 'stat_par_diff': [-0.22391514330850826], 'average_odds_difference': [-0.1785541826651185], 'generalized_entropy_index': [0.11412821786436768], 'acc': [0.730561555075594, 0.730561555075594], 'group': [0.2229794356767097], 'causal': [0.08045356371490281]},
      {'disp_imp': [1.6371722737141923], 'stat_par_diff': [0.1478239675016926], 'average_odds_difference': [0.1048377336598204], 'generalized_entropy_index': [0.5242102955253198], 'acc': [0.7219222462203023], 'group': [0.14746097995739316], 'causal': [0.032937365010799136]},
      {'disp_imp': [0.798393515644967], 'stat_par_diff': [-0.14322992322058514], 'average_odds_difference': [-0.09881459875845627], 'generalized_entropy_index': [0.11412821786436768], 'acc': [0.730561555075594, 0.730561555075594], 'group': [0.14230572720098705], 'causal': [0.03509719222462203]},
      {'disp_imp': [1.1914754098360654], 'stat_par_diff': [0.06059348412533716], 'average_odds_difference': [0.018071447126457973], 'generalized_entropy_index': [0.5161185912898559], 'acc': [0.7165226781857451], 'group': [0.06040127978949261], 'causal': [0.09935205183585313]},
      {'disp_imp': [0.6448197050166894], 'stat_par_diff': [-0.28507892097344667], 'average_odds_difference': [-0.2246457454319335], 'generalized_entropy_index': [0.11412821786436768], 'acc': [0.730561555075594], 'group': [0.28325221415463175], 'causal': [0.09719222462203024]},
      {'disp_imp': [0.9647302563427989], 'stat_par_diff': [-0.014154305020324165], 'average_odds_difference': [-0.07318239275043863], 'generalized_entropy_index': [0.500489548567512], 'acc': [0.7068034557235421], 'group': [0.08455806108200026], 'causal': [0.26457883369330454]}]
# GR = [{'disp_imp': [0.7139676380110237], 'stat_par_diff': [-0.22730038366057315], 'average_odds_difference': [-0.1836198696658397], 'generalized_entropy_index': [0.11511141941222001], 'accuracy': [0.7300215982721382], 'acc': [0.7300215982721382, 0.7300215982721382], 'group': [0.2263669695520485], 'causal': [0.0658682634730539]},
#       {'disp_imp': [1.7016877841183866], 'stat_par_diff': [0.1571780636425186], 'average_odds_difference': [0.11573046326940173], 'generalized_entropy_index': [0.524339482795473], 'accuracy': [0.2780777537796976], 'acc': [0.7240820734341252], 'group': [0.1568373838818602], 'causal': [0.029940119760479042]},
#       {'disp_imp': [0.7926247763700756], 'stat_par_diff': [-0.1473282838763229], 'average_odds_difference': [-0.10599284369662185], 'generalized_entropy_index': [0.11511141941222001], 'accuracy': [0.7300215982721382], 'acc': [0.7300215982721382, 0.7300215982721382], 'group': [0.14640744992453092], 'causal': [0.021956087824351298]},
#       {'disp_imp': [1.0313774942511684], 'stat_par_diff': [0.0109721934011206], 'average_odds_difference': [-0.033771782497709665], 'generalized_entropy_index': [0.5143684894493107], 'accuracy': [0.281317494600432], 'acc': [0.7192224622030238], 'group': [0.010713881763779742], 'causal': [0.19760479041916168]},
#       {'disp_imp': [0.6307309023845732], 'stat_par_diff': [-0.30124584279153244], 'average_odds_difference': [-0.2454437370563614], 'generalized_entropy_index': [0.11511141941222001], 'accuracy': [0.7300215982721382], 'acc': [0.7300215982721382], 'group': [0.2995092954599856], 'causal': [0.066]},
#       {'disp_imp': [1.076349047141424], 'stat_par_diff': [0.02511481813862637], 'average_odds_difference': [-0.03342633827853335], 'generalized_entropy_index': [0.531609453808431], 'accuracy': [0.291036717062635], 'acc': [0.7084233261339092], 'group': [0.14270595897101923], 'causal': [0.278]}]


# 8 calibrated euqodds CEO -- postprocessing  (MaxAbsScaler)
CEO = [{'disp_imp': [0.7577015512344331], 'stat_par_diff': [-0.18576214405360136], 'average_odds_difference': [-0.13624044548900036], 'generalized_entropy_index': [0.11761165061876161], 'acc': [0.725506072874494], 'group':[0.18420144296742802], 'causal':[0.0718562874251497]},
       {'disp_imp': [0.7577015512344331], 'stat_par_diff': [-0.18576214405360136], 'average_odds_difference': [-0.13624044548900036], 'generalized_entropy_index': [0.11761165061876161], 'acc': [0.725506072874494], 'group': [0.18420144296742802], 'causal': [0.4091816367265469]},
       {'disp_imp': [0.7805192710853088], 'stat_par_diff': [-0.15563178959405377], 'average_odds_difference': [-0.11383677984850288], 'generalized_entropy_index': [0.11845279037775125], 'acc': [0.728744939271255], 'group': [0.15654997905705093], 'causal': [0.021956087824351298]},
       {'disp_imp': [0.7805192710853088], 'stat_par_diff': [-0.15563178959405377], 'average_odds_difference': [-0.11383677984850288], 'generalized_entropy_index': [0.11845279037775125], 'acc': [0.728744939271255], 'group': [0.15654997905705093], 'causal': [0.4171656686626746]},
       {'disp_imp': [0.6675675675675676], 'stat_par_diff': [-0.2589473684210526], 'average_odds_difference': [-0.1874619055833477], 'generalized_entropy_index': [0.11826445154793479], 'acc': [0.7271255060728745], 'group': [0.25579451201521164], 'causal': [0.094]},
       {'disp_imp': [0.6675675675675676], 'stat_par_diff': [-0.2589473684210526], 'average_odds_difference': [-0.1874619055833477], 'generalized_entropy_index': [0.10263801063326096], 'acc': [0.6785425101214575], 'group': [0.47919876733436056], 'causal': [0.308]}]

# 9 eqodds EO -- postprocessing  (MaxAbsScaler)
EO = [{'disp_imp': [0.7614287654488658], 'stat_par_diff': [-0.18389865996649923], 'average_odds_difference': [-0.13074391606353522], 'generalized_entropy_index': [0.11426923052969566], 'acc': [0.7295546558704453], 'group': [0.18653342650042512], 'causal': [0.0658682634730539]},
      {'disp_imp': [0.7614287654488658], 'stat_par_diff': [-0.18389865996649923], 'average_odds_difference': [-0.13074391606353522], 'generalized_entropy_index': [0.11426923052969566], 'acc': [0.7295546558704453], 'group': [0.18653342650042512], 'causal': [0.40119760479041916]},
      {'disp_imp': [0.7891255832826131], 'stat_par_diff': [-0.148570611778159], 'average_odds_difference': [-0.10862046239029216], 'generalized_entropy_index': [0.12204010354699574], 'acc': [0.7206477732793523], 'group': [0.14947527871335703], 'causal': [0.03992015968063872]},
      {'disp_imp': [0.7891255832826131], 'stat_par_diff': [-0.148570611778159], 'average_odds_difference': [-0.10862046239029216], 'generalized_entropy_index': [0.12204010354699574], 'acc': [0.7206477732793523], 'group': [0.14947527871335703], 'causal': [0.41317365269461076]},
      {'disp_imp': [0.6379417879417879], 'stat_par_diff': [-0.2820242914979757], 'average_odds_difference': [-0.21051693404634578], 'generalized_entropy_index': [0.12025410252549955], 'acc': [0.728744939271255], 'group': [0.27890699275481107], 'causal': [0.092]},
      {'disp_imp': [0.6379417879417879], 'stat_par_diff': [-0.2820242914979757], 'average_odds_difference': [-0.21051693404634578], 'generalized_entropy_index': [0.12025410252549955], 'acc': [0.728744939271255], 'group': [0.27890699275481107], 'causal': [0.42]}]

# 10 reject_option -- postprocessing (MaxAbsScaler)
# RO = [{'disp_imp': [0.7752055073053127], 'stat_par_diff': [-0.1620393634840871], 'average_odds_difference': [-0.12012761974509678], 'generalized_entropy_index': [0.13068567896523722], 'acc': [0.6631578947368421], 'group':[0.1571310709444954], 'causal':[0.4121457489878543]},
#       {'disp_imp': [1.6539842067480257], 'stat_par_diff': [0.03814907872696817], 'average_odds_difference': [0.02247933410803183], 'generalized_entropy_index': [0.702261082646469], 'acc': [0.37732793522267205], 'group':[0.03800207100342634], 'causal':[0.08906882591093117]},
#       {'disp_imp': [0.8157191597813129], 'stat_par_diff': [-0.12815894797026872], 'average_odds_difference': [-0.08969028856676414], 'generalized_entropy_index': [0.12622622570682204], 'acc': [0.6495680345572354], 'group':[0.12902864880682569], 'causal':[0.297165991902834]},
#       {'disp_imp': [1.2769731007981082], 'stat_par_diff': [0.019618927973199335], 'average_odds_difference': [-0.001680329757514385], 'generalized_entropy_index': [0.7006084663815022], 'acc': [0.3797570850202429], 'group':[0.009889088436623193], 'causal':[0.9927125506072875]},
#       {'disp_imp': [0.6768047337278106], 'stat_par_diff': [-0.22113360323886644], 'average_odds_difference': [-0.16533695589672814], 'generalized_entropy_index': [0.140853755862977], 'acc': [0.6089068825910932], 'group':[0.23759844204759456], 'causal':[0.3894736842105263]},
#       {'disp_imp': [3.4589743589743587], 'stat_par_diff': [0.07765182186234817], 'average_odds_difference': [0.07398510723937668], 'generalized_entropy_index': [0.6962000710517672], 'acc': [0.3813765182186235], 'group':[0.07748418188374917], 'causal':[0.08825910931174089]}]
RO = [{'disp_imp': [0.7752055073053127], 'stat_par_diff': [-0.1620393634840871], 'average_odds_difference': [-0.12012761974509678], 'generalized_entropy_index': [0.13068567896523722], 'acc': [0.6631578947368421], 'group':[0.1571310709444954], 'causal':[0.4121457489878543]},
      {'disp_imp': [1.8024668798538146], 'stat_par_diff': [0.18389865996649915], 'average_odds_difference': [0.13074391606353522],'generalized_entropy_index': [0.5077792217168435], 'acc': [0.2704453441295547], 'group':[0.18653342650042515], 'causal':[0.37732793522267205]},
      {'disp_imp': [0.8157191597813129], 'stat_par_diff': [-0.12815894797026872], 'average_odds_difference': [-0.08969028856676414], 'generalized_entropy_index': [0.12622622570682204], 'acc': [0.6495680345572354], 'group':[0.12902864880682569], 'causal':[0.297165991902834]},
      {'disp_imp': [1.58687106918239], 'stat_par_diff': [0.17072612921669528], 'average_odds_difference': [0.12893515669386002], 'generalized_entropy_index': [0.4847600008984524], 'acc': [0.27125506072874495], 'group':[0.17166332918299543], 'causal':[0.4008097165991903]},
      {'disp_imp': [0.6768047337278106], 'stat_par_diff': [-0.22113360323886644], 'average_odds_difference': [-0.16533695589672814], 'generalized_entropy_index': [0.140853755862977], 'acc': [0.6089068825910932], 'group':[0.23759844204759456], 'causal':[0.3894736842105263]},
      {'disp_imp': [2.382307692307692], 'stat_par_diff': [0.2910121457489878], 'average_odds_difference': [0.22448249094359152], 'generalized_entropy_index': [0.5607173661973889], 'acc': [0.3255060728744939], 'group':[0.5007704160246533], 'causal':[0.28016]}]
#gender:
# {'disp_imp': [1.8024668798538146], 'stat_par_diff': [0.18389865996649915], 'average_odds_difference': [0.13074391606353522],
#  'generalized_entropy_index': [0.5077792217168435], 'acc': [0.2704453441295547], 'group':[0.18653342650042515], 'causal':[0.37732793522267205]}
# race:
# {'disp_imp': [1.58687106918239], 'stat_par_diff': [0.17072612921669528],
#  'average_odds_difference': [0.12893515669386002], 'generalized_entropy_index': [0.4847600008984524],
#  'acc': [0.27125506072874495], 'group':[0.17166332918299543], 'causal':[0.4008097165991903]}
# gender+rage:
# {'disp_imp': [2.382307692307692], 'stat_par_diff': [0.2910121457489878], 'average_odds_difference': [0.22448249094359152],
#  'generalized_entropy_index': [0.5607173661973889], 'acc': [0.3255060728744939], 'group':[0.5007704160246533], 'causal':[0.28016]}


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
    percent = format((uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0])/float(uni_orig_metrics[0]['group'][0]), '.0%')
    print(str(round(uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0], 3)) + "(" + str(percent) + ")")
    percent = format((uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0]) / float(uni_orig_metrics[1]['group'][0]),'.0%')
    print(str(round(uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0], 3)) + "(" + str(percent) + ")")
    percent = format((multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0]) / float(multi_orig_metrics['group'][0]),'.0%')
    print(str(round(multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0], 3)) + "(" + str(percent) + ")")

    print(str(round(uni_trans_metrics[0]['acc'][0] - uni_orig_metrics[0]['acc'][0], 3)))
    print(str(round(uni_trans_metrics[1]['acc'][0] - uni_orig_metrics[1]['acc'][0], 3)))
    print(str(round(multi_trans_metrics['acc'][0] - multi_orig_metrics['acc'][0], 3)))

    try:
        print("causal metric")
        percent = format((uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0]) / float(uni_orig_metrics[0]['causal'][0]), '.0%')
        percent = ''
        print(str(round(uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0], 3)) + "" + str( percent) + "")
        percent = format((uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0]) / float(uni_orig_metrics[1]['causal'][0]), '.0%')
        percent = ''
        print(str(round(uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0], 3)) + "" + str(percent) + "")
        percent = format((multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0]) / float(multi_orig_metrics['causal'][0]), '.0%')
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
Plot.plot_abs_acc_multi_metric(all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics,all_multi_trans_metrics)
# 3 images: one for 'race', one for 'sex', one for 'race,sex'
Plot.plot_abs_acc_individual_metric(all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics,all_multi_trans_metrics)
