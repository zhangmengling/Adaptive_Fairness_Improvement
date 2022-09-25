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
RW = [{'disp_imp': [0.9422764227642276], 'stat_par_diff': [-0.030380830124090707], 'average_odds_difference': [-0.061574197860962576], 'generalized_entropy_index': [0.1874132054033582], 'accuracy': [0.6333333333333333], 'acc': [0.6333333333333333], 'group': [0.026053864168618324], 'causal': [0.07777777777777778]},
      {'disp_imp': [0.9954832881662151], 'stat_par_diff': [-0.00213949507916128], 'average_odds_difference': [-0.04634692513368982], 'generalized_entropy_index': [0.17536506219578143], 'accuracy': [0.6777777777777778], 'acc': [0.6777777777777778], 'group': [0.002927400468384078], 'causal': [0.08888888888888889]},
      {'disp_imp': [1.2150638032990975], 'stat_par_diff': [0.09519217523074802], 'average_odds_difference': [0.13685113685113687], 'generalized_entropy_index': [0.1874132054033582], 'accuracy': [0.6333333333333333], 'acc': [0.6333333333333333], 'group': [0.10056497175141244], 'causal': [0.15]},
      {'disp_imp': [1.0252100840336134], 'stat_par_diff': [0.012398401983744312], 'average_odds_difference': [0.05426855426855426], 'generalized_entropy_index': [0.1923297902901465], 'accuracy': [0.6277777777777778], 'acc': [0.6277777777777778], 'group': [0.0], 'causal': [0.16666666666666666]},
      {'disp_imp': [1.1753424657534248], 'stat_par_diff': [0.07970112079701125], 'average_odds_difference': [0.24419191919191918], 'generalized_entropy_index': [0.1874132054033582], 'accuracy': [0.6333333333333333], 'group': [0.1333333333333333], 'causal': [0.17222222222222222], 'acc': [0.6333333333333333]},
      {'disp_imp': [1.0296803652968038], 'stat_par_diff': [0.016189290161892966], 'average_odds_difference': [0.01919191919191915], 'generalized_entropy_index': [0.15926275992438577], 'accuracy': [0.6666666666666666], 'group': [0.05555555555555558], 'causal': [0.16666666666666666], 'acc': [0.6666666666666666]}]


# 2 optim -- preprocessing

# 3 disparate impact remover -- preprocessing (MinMaxScaler)
DI = [{'disp_imp': [0.9651546068121175], 'stat_par_diff': [-0.016813511398348913], 'group': [0.022147377015717506], 'acc': [0.6047619047619047]},
      {'disp_imp': [1.08014440433213], 'stat_par_diff': [0.03642927469642271], 'group': [0.03842620943049602], 'acc': [0.6047619047619047]},
      {'disp_imp': [0.9768192826663591], 'stat_par_diff': [-0.011111418226042724], 'group': [0.008557046979865757], 'acc': [0.6333333333333333]},
      {'disp_imp': [1.071217784772772], 'stat_par_diff': [0.030017413416622896], 'group': [0.032997762863534674], 'acc': [0.6333333333333333]},
      {'disp_imp': [1.0998509687034277], 'stat_par_diff': [0.040680024286581684], 'group': [0.07257525083612043], 'acc': [0.6023809523809524]},
      {'disp_imp': [1.1950819672131148], 'stat_par_diff': [0.07225258044930177], 'group': [0.13151364764267987], 'acc': [0.6071428571428571]}]
# DI = [{'disp_imp': [1.516471119133574], 'stat_par_diff': [0.17336093509378708], 'average_odds_difference': [0.15712111730572306], 'generalized_entropy_index': [0.1705279572287814], 'accuracy': [0.6857142857142857], 'group': [0.16921820779750973], 'causal': [0.25476190476190474]},
#       {'disp_imp': [1.117121382493934], 'stat_par_diff': [0.049960869455454304], 'average_odds_difference': [0.036331263646657896], 'generalized_entropy_index': [0.1828605401732632], 'accuracy': [0.6571428571428571], 'group': [0.04506021637068791], 'causal': [0.11428571428571428]},
#       {'disp_imp': [0.7037952595608551], 'stat_par_diff': [-0.16891014124215703], 'average_odds_difference': [-0.13670625608108009], 'generalized_entropy_index': [0.1705279572287814], 'accuracy': [0.6857142857142857], 'group': [0.1756711409395973], 'causal': [0.2119047619047619]},
#       {'disp_imp': [0.8845198279980889], 'stat_par_diff': [-0.06680671107548575], 'average_odds_difference': [-0.03801366035947845], 'generalized_entropy_index': [0.1539853547681171], 'accuracy': [0.6666666666666666], 'group': [0.06493288590604018], 'causal': [0.04285714285714286]},
#       {'disp_imp': [1.0327868852459017], 'stat_par_diff': [0.014571948998178541], 'average_odds_difference': [0.05859807754969046], 'generalized_entropy_index': [0.1705279572287814], 'accuracy': [0.6857142857142857], 'group': [0.3085553997194951]},
#       {'disp_imp': [0.864168618266979], 'stat_par_diff': [-0.07043108682452942], 'average_odds_difference': [-0.044138970348647805], 'generalized_entropy_index': [0.18330118603028683], 'accuracy': [0.6452380952380953], 'group': [0.1545582047685835]}]


# 4 meta --inprocessing (maxabsscaler)
# META = [{'disp_imp': [0.9422764227642276], 'stat_par_diff': [-0.030380830124090707], 'acc': [0.6333333333333333, 0.6333333333333333], 'group': [0.026053864168618324], 'causal': [0.07777777777777778]},
#         {'disp_imp': [1.019512195121951], 'stat_par_diff': [0.011981172443303323], 'acc': [0.6833333333333333], 'group': [0.015807962529274078], 'causal': [0.027777777777777776]},
#         {'disp_imp': [1.2150638032990975], 'stat_par_diff': [0.09519217523074802], 'acc': [0.6333333333333333, 0.6333333333333333], 'group': [0.10056497175141244], 'causal': [0.15]},
#         {'disp_imp': [1.3489606368863334], 'stat_par_diff': [0.10869265739082518], 'acc': [0.6666666666666666], 'group': [0.11525423728813561], 'causal': [0.20555555555555555]},
#         {'disp_imp': [1.1753424657534248], 'stat_par_diff': [0.07970112079701125], 'acc': [0.6333333333333333], 'group': [0.1333333333333333], 'causal': [0.17222222222222222]},
#         {'disp_imp': [0.8664383561643835], 'stat_par_diff': [-0.09713574097135746], 'acc': [0.7], 'group': [0.18979591836734688], 'causal': [0.05]}]
META = [{'disp_imp': [0.9422764227642276], 'stat_par_diff': [-0.030380830124090707], 'average_odds_difference': [-0.061574197860962576], 'generalized_entropy_index': [0.1874132054033582], 'accuracy': [0.6333333333333333], 'acc': [0.6333333333333333, 0.6333333333333333], 'group': [0.026053864168618324], 'causal': [0.07777777777777778]},
        {'disp_imp': [1.0040650406504066], 'stat_par_diff': [0.0025673940949936247], 'average_odds_difference': [-0.03913435828877007], 'generalized_entropy_index': [0.12151926932501675], 'accuracy': [0.6833333333333333], 'acc': [0.6833333333333333], 'group': [0.006147540983606592], 'causal': [0.027777777777777776]},
        {'disp_imp': [1.2150638032990975], 'stat_par_diff': [0.09519217523074802], 'average_odds_difference': [0.13685113685113687], 'generalized_entropy_index': [0.1874132054033582], 'accuracy': [0.6333333333333333], 'acc': [0.6333333333333333, 0.6333333333333333], 'group': [0.10056497175141244], 'causal': [0.15]},
        {'disp_imp': [1.0422969187675069], 'stat_par_diff': [0.020801763328282097], 'average_odds_difference': [0.07646932646932644], 'generalized_entropy_index': [0.14764549930564336], 'accuracy': [0.7111111111111111], 'acc': [0.7111111111111111], 'group': [0.02514124293785308], 'causal': [0.011111111111111112]},
        {'disp_imp': [1.1753424657534248], 'stat_par_diff': [0.07970112079701125], 'average_odds_difference': [0.24419191919191918], 'generalized_entropy_index': [0.1874132054033582], 'accuracy': [0.6333333333333333], 'acc': [0.6333333333333333], 'group': [0.1333333333333333], 'causal': [0.17222222222222222]},
        {'disp_imp': [0.9902152641878669], 'stat_par_diff': [-0.0062266500622665255], 'average_odds_difference': [0.028787878787878834], 'generalized_entropy_index': [0.13402867245963654], 'accuracy': [0.6944444444444444], 'acc': [0.6944444444444444], 'group': [0.11479591836734693], 'causal': [0.05555555555555555]}]

# 5 adversarial debias --inprocessing  (maxabsscaler)
AD = [{'disp_imp': [0.8469301934398654], 'stat_par_diff': [-0.07787762088147199], 'acc': [0.6944444444444444], 'group': [0.07377049180327871], 'causal': [0.11666666666666667]},
      {'disp_imp': [0.8959349593495936], 'stat_par_diff': [-0.0547710740265297], 'acc': [0.6833333333333333], 'group': [0.050644028103044525], 'causal': [0.005555555555555556]},
      {'disp_imp': [0.9739495798319329], 'stat_par_diff': [-0.0128116820498691], 'acc': [0.6777777777777778], 'group': [0.008757062146892647], 'causal': [0.15]},
      {'disp_imp': [0.7859943977591037], 'stat_par_diff': [-0.10524865683978507], 'acc': [0.7166666666666667], 'group': [0.10197740112994352], 'causal': [0.05555555555555555]},
      {'disp_imp': [1.1753424657534248], 'stat_par_diff': [0.07970112079701125], 'acc': [0.6333333333333333], 'group': [0.1333333333333333], 'causal': [0.17222222222222222]},
      {'disp_imp': [0.5462328767123287], 'stat_par_diff': [-0.3300124533001246], 'acc': [0.6777777777777778], 'group': [0.31111111111111106], 'causal': [0.05555555555555555]}]
# AD = [{'disp_imp': [0.9654471544715447], 'stat_par_diff': [-0.02182284980744542], 'average_odds_difference': [-0.06293114973262035], 'generalized_entropy_index': [0.1290174471992653], 'accuracy': [0.6777777777777778], 'acc': [0.6777777777777778, 0.6777777777777778], 'group': [0.018442622950819665], 'causal': [0.1]},
#       {'disp_imp': [0.8959349593495936], 'stat_par_diff': [-0.0547710740265297], 'average_odds_difference': [-0.09669786096256683], 'generalized_entropy_index': [0.16710204081632665], 'accuracy': [0.6833333333333333], 'acc': [0.6833333333333333], 'group': [0.050644028103044525], 'causal': [0.005555555555555556]},
#       {'disp_imp': [0.7014595311808934], 'stat_par_diff': [-0.18597602975616478], 'average_odds_difference': [-0.11832368082368086], 'generalized_entropy_index': [0.15211146222349897], 'accuracy': [0.7055555555555556], 'acc': [0.7055555555555556, 0.7055555555555556], 'group': [0.1844632768361582], 'causal': [0.15]},
#       {'disp_imp': [0.7859943977591037], 'stat_par_diff': [-0.10524865683978507], 'average_odds_difference': [-0.04319498069498072], 'generalized_entropy_index': [0.16731905604275668], 'accuracy': [0.7166666666666667], 'acc': [0.7166666666666667], 'group': [0.10197740112994352], 'causal': [0.05555555555555555]},
#       {'disp_imp': [0.7345890410958904], 'stat_par_diff': [-0.1930261519302615], 'average_odds_difference': [0.07222222222222222], 'generalized_entropy_index': [0.12955254942767938], 'accuracy': [0.7222222222222222], 'acc': [0.7222222222222222], 'group': [0.16666666666666663], 'causal': [0.1388888888888889]},
#       {'disp_imp': [0.5462328767123287], 'stat_par_diff': [-0.3300124533001246], 'average_odds_difference': [-0.2578282828282828], 'generalized_entropy_index': [0.18239795918367344], 'accuracy': [0.6777777777777778], 'acc': [0.6777777777777778], 'group': [0.31111111111111106], 'causal': [0.05555555555555555]}]


# 6 prejudice remover --inprocssing (MaxAbsScaler)
PR = [{'disp_imp': [0.935483870967742], 'stat_par_diff': [-0.03448275862068961], 'average_odds_difference': [-0.0645293908161555], 'generalized_entropy_index': [0.1874132054033582], 'acc': [0.6277777777777778], 'group': [0.030448020878642845], 'causal': [0.07777777777777778]},
      {'disp_imp': [1.1147540983606556], 'stat_par_diff': [0.05737704918032782], 'average_odds_difference': [0.012679110105580732], 'generalized_entropy_index': [0.1423004568664339], 'acc': [0.7055555555555556], 'group': [0.062490938089024284], 'causal': [0.18333333333333332]},
      {'disp_imp': [1.2037037037037035], 'stat_par_diff': [0.09166666666666662], 'average_odds_difference': [0.13423590311462166], 'generalized_entropy_index': [0.1874132054033582], 'acc': [0.6277777777777778], 'group': [0.09713715994872524], 'causal': [0.15]},
      {'disp_imp': [0.8918918918918919], 'stat_par_diff': [-0.06666666666666665], 'average_odds_difference': [-0.03158454302619068], 'generalized_entropy_index': [0.15255731922398583], 'acc': [0.6611111111111111], 'group': [0.06395100413046584], 'causal': [0.26666666666666666]},
      {'disp_imp': [1.1753424657534248], 'stat_par_diff': [0.07970112079701125], 'average_odds_difference': [0.24419191919191918], 'generalized_entropy_index': [0.1874132054033582], 'acc': [0.6333333333333333], 'group': [0.1333333333333333], 'causal': [0.17222222222222222]},
      {'disp_imp': [0.8825831702544031], 'stat_par_diff': [-0.07471980074719797], 'average_odds_difference': [0.15277777777777776], 'generalized_entropy_index': [0.1423004568664339], 'acc': [0.7055555555555556], 'group': [0.1111111111111111], 'causal': [0.32222222222222224]}]

# 7 gradient reduction -- DemographicParity (MaxAbsScaler)
GR = [{'disp_imp': [0.9422764227642276], 'stat_par_diff': [-0.030380830124090707], 'average_odds_difference': [-0.061574197860962576], 'generalized_entropy_index': [0.1874132054033582], 'acc': [0.6333333333333333], 'group': [0.026053864168618324], 'causal': [0.07777777777777778]},
      {'disp_imp': [0.986624704956727], 'stat_par_diff': [-0.007274283269148474], 'average_odds_difference': [-0.0500233957219251], 'generalized_entropy_index': [0.14331285444234415], 'acc': [0.6777777777777778], 'group': [0.0029274004683840227], 'causal': [0.03888888888888889]},
      {'disp_imp': [1.2150638032990975], 'stat_par_diff': [0.09519217523074802], 'average_odds_difference': [0.13685113685113687], 'generalized_entropy_index': [0.1874132054033582], 'acc': [0.6333333333333333], 'group': [0.10056497175141244], 'causal': [0.15]},
      {'disp_imp': [1.3479614067849361], 'stat_par_diff': [0.15401570464251274], 'average_odds_difference': [0.20567889317889315], 'generalized_entropy_index': [0.15478451424397374], 'acc': [0.6777777777777778], 'group': [0.15988700564971747], 'causal': [0.19444444444444445]},
      {'disp_imp': [1.1753424657534248], 'stat_par_diff': [0.07970112079701125], 'average_odds_difference': [0.24419191919191918], 'generalized_entropy_index': [0.1874132054033582], 'acc': [0.6333333333333333], 'group': [0.1333333333333333], 'causal': [0.17222222222222222]},
      {'disp_imp': [0.9041095890410958], 'stat_par_diff': [-0.06102117061021173], 'average_odds_difference': [-0.029166666666666646], 'generalized_entropy_index': [0.14331285444234415], 'acc': [0.6777777777777778], 'group': [0.13061224489795914], 'causal': [0.027777777777777776]}]

# 8 calibrated euqodds CEO -- postprocessing  (no scaler) (postprocessing_ceo_demo.py)
CEO = [{'disp_imp': [1.2447368421052631], 'stat_par_diff': [0.11124401913875598], 'average_odds_difference': [0.029896574014221036], 'generalized_entropy_index': [0.11471346420781285], 'acc': [0.7055555555555556], 'group': [0.10821705426356593], 'causal': [0.06666666666666667]},
       {'disp_imp': [0.9842105263157895], 'stat_par_diff': [-0.0071770334928229484], 'average_odds_difference': [-0.0856496444731739], 'generalized_entropy_index': [0.14635272391505072], 'acc': [0.7], 'group': [0.011782945736434125], 'causal': [0.45]},
       {'disp_imp': [0.7906976744186046], 'stat_par_diff': [-0.12927496580027364],'average_odds_difference': [-0.08151476251604622], 'generalized_entropy_index': [0.11471346420781285], 'acc': [0.7055555555555556], 'group': [0.11194295900178253], 'causal': [0.1]},
       {'disp_imp': [0.6212624584717608], 'stat_par_diff': [-0.23392612859097128], 'average_odds_difference': [-0.19018684923691342], 'generalized_entropy_index': [0.16481994459833793], 'acc': [0.6833333333333333], 'group': [0.217825311942959], 'causal': [0.45], 'acc':[0.7]},
       {'disp_imp': [0.823529411764706], 'stat_par_diff': [-0.11764705882352933], 'average_odds_difference': [0.02115987460815047], 'generalized_entropy_index': [0.11471346420781285], 'acc': [0.7055555555555556], 'group': [0.21323529411764708], 'causal': [0.08333333333333333]},
       {'disp_imp': [0.14705882352941177], 'stat_par_diff': [-0.5686274509803921], 'average_odds_difference': [-0.41379310344827586], 'generalized_entropy_index': [0.3450704225352112], 'group': [0.5686274509803921], 'causal': [0.09166666666666666], 'acc':[0.5333333333333333]}]

# CEO = [{},
#        {'disp_imp': [1.6939759036144577], 'stat_par_diff': [0.1875610550309345], 'average_odds_difference': [0.11304008858356684], 'generalized_entropy_index': [0.20196111438431122], 'acc': [0.6583333333333333], 'group':[0.1856368563685637], 'causal':[0.6]},
#        {},
#        {'disp_imp': [0.7028571428571428], 'stat_par_diff': [-0.20634920634920634],
#         'average_odds_difference': [-0.19649591894173937], 'generalized_entropy_index': [0.16588133176266345],
#         'acc': [0.625], 'group':[0.2323580034423408], 'causal':[0.45]},
#        {},
#        {'disp_imp': [0.5], 'stat_par_diff': [-0.5], 'average_odds_difference': [-0.5268817204301075], 'generalized_entropy_index': [0.26559546313799626], 'acc': [0.6333333333333333], 'group':[0.49019607843137253], 'causal':[0.25833333333333336]}
#        ]

# 9 eqodds EO -- postprocessing  (MaxAbsScaler)
EO = [{'disp_imp': [1.2447368421052631], 'stat_par_diff': [0.11124401913875598], 'average_odds_difference': [0.029896574014221036], 'generalized_entropy_index': [0.11471346420781285], 'acc': [0.7055555555555556], 'group': [0.10821705426356593], 'causal': [0.06666666666666667]},
         {'disp_imp': [1.2447368421052631], 'stat_par_diff': [0.11124401913875598], 'average_odds_difference': [0.029896574014221036], 'generalized_entropy_index': [0.11471346420781285], 'group': [0.10821705426356593], 'causal': [0.525], 'acc':[0.6888888888888889]},
         {'disp_imp': [0.7906976744186046], 'stat_par_diff': [-0.12927496580027364], 'average_odds_difference': [-0.08151476251604622], 'generalized_entropy_index': [0.11471346420781285], 'acc': [0.7055555555555556], 'group': [0.11194295900178253], 'causal': [0.1]},
         {'disp_imp': [0.7906976744186046], 'stat_par_diff': [-0.12927496580027364], 'average_odds_difference': [-0.08151476251604622], 'generalized_entropy_index': [0.11471346420781285], 'group': [0.11194295900178253], 'causal': [0.525], 'acc':[0.7166666666666667]},
         {'disp_imp': [0.823529411764706], 'stat_par_diff': [-0.11764705882352933], 'average_odds_difference': [0.02115987460815047], 'generalized_entropy_index': [0.11471346420781285], 'acc': [0.7055555555555556], 'group': [0.21323529411764708], 'causal': [0.08333333333333333]},
        {'disp_imp': [0.823529411764706], 'stat_par_diff': [-0.11764705882352933], 'average_odds_difference': [0.02115987460815047], 'generalized_entropy_index': [0.11471346420781285], 'group': [0.21323529411764708], 'causal': [0.525], 'acc':[0.6111111111111112]}]

# EO = [{},
#       {'disp_imp': [1.5399780941949615], 'stat_par_diff': [0.16053402800390748], 'average_odds_difference': [0.09220675525023352], 'generalized_entropy_index': [0.20413223140495884], 'acc': [0.65], 'group':[0.1578590785907859], 'causal':[0.5916666666666667]},
#       {},
#       {'disp_imp': [0.7028571428571428], 'stat_par_diff': [-0.20634920634920634], 'average_odds_difference': [-0.19649591894173937], 'generalized_entropy_index': [0.16588133176266345], 'acc': [0.625], 'group':[0.2323580034423408], 'causal':[0.45]},
#       {},
#       {'disp_imp': [0.5], 'stat_par_diff': [-0.5], 'average_odds_difference': [-0.5268817204301075], 'generalized_entropy_index': [0.26559546313799626], 'acc': [0.6333333333333333], 'group':[0.49019607843137253], 'causal':[0.25833333333333336]}]

# 10 reject_option -- postprocessing (MaxAbsScaler)
RO = [{'disp_imp': [1.2447368421052631], 'stat_par_diff': [0.11124401913875598], 'average_odds_difference': [0.029896574014221036], 'generalized_entropy_index': [0.11471346420781285], 'acc': [0.7055555555555556], 'group':[0.10821705426356593], 'causal':[0.06666666666666667]},
      {'disp_imp': [1.2447368421052631], 'stat_par_diff': [0.11124401913875598], 'average_odds_difference': [0.029896574014221036], 'generalized_entropy_index': [0.11471346420781285], 'acc': [0.7111111111111111], 'group':[0.08837209302325577], 'causal':[0.5583333333333333]},
      {'disp_imp': [0.7906976744186046], 'stat_par_diff': [-0.12927496580027364], 'average_odds_difference': [-0.08151476251604622], 'generalized_entropy_index': [0.11471346420781285], 'acc': [0.7055555555555556], 'group':[0.11194295900178253], 'causal':[0.1]},
      {'disp_imp': [0.7906976744186046], 'stat_par_diff': [-0.12927496580027364], 'average_odds_difference': [-0.08151476251604622], 'generalized_entropy_index': [0.11471346420781285], 'acc': [0.6833333333333333], 'group':[0.12192513368983954], 'causal':[0.4583333333333333]},
      {'disp_imp': [0.823529411764706], 'stat_par_diff': [-0.11764705882352933], 'average_odds_difference': [0.02115987460815047], 'generalized_entropy_index': [0.11471346420781285], 'acc': [0.7055555555555556], 'group':[0.21323529411764708], 'causal':[0.08333333333333333]},
      {'disp_imp': [0.823529411764706], 'stat_par_diff': [-0.11764705882352933], 'average_odds_difference': [0.02115987460815047], 'generalized_entropy_index': [0.11471346420781285], 'acc': [0.7055555555555556], 'group':[0.25980392156862747], 'causal':[0.4666666666666667]}]

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


print("-->all_uni_orig_metrics", all_uni_orig_metrics)
print("-->all_uni_trans_metrics", all_uni_trans_metrics)
print("-->all_multi_orig_metrics", all_multi_orig_metrics)
print("-->all_multi_trans_metrics", all_multi_trans_metrics)

# processing_names = [RW_processing_name, OP_processing_name, DI_processing_name]
processing_names = ["RW", "DI", "META", "AD", "PR", "GR", "CEO", "EO", "RO"]
# processing_names = ["RW", "OP", "AD", "META"]
# dataset_name = "Adult income"
# sens_attrs = ["race", "sex"]
dataset_name = "German credit"
sens_attrs = ["sex", "age"]
# dataset_name = "Compas"
# sens_attrs = ["sex", "race"]

#
# for i in range(0, len(processing_names)):
#     process_name = processing_names[i]
#     print("-->process_name", process_name)
#     uni_orig_metrics = all_uni_orig_metrics[i]
#     uni_trans_metrics = all_uni_trans_metrics[i]
#     multi_orig_metrics = all_multi_orig_metrics[i]
#     multi_trans_metrics = all_multi_trans_metrics[i]
#     print("group metric")
#     try:
#         percent = format((multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0]) / float(multi_orig_metrics['group'][0]), '.0%')
#     except:
#         percent = multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0]
#     print(str(round(multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0], 3)) + "(" + str(percent)+ ")")
#     try:
#         percent = format((uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0]) / float(uni_orig_metrics[0]['group'][0]),'.0%')
#     except:
#         percent = uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0]
#     print(str(round(uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0], 3)) + "(" + str(percent) + ")")
#     try:
#         percent = format((uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0])/float(uni_orig_metrics[1]['group'][0]), '.0%')
#     except:
#         percent = uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0]
#     print(str(round(uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0], 3)) + "(" + str(percent) + ")")
#     print("causal metric")
#     try:
#         try:
#             percent = format((multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0]) / float(
#                 multi_orig_metrics['causal'][0]), '.0%')
#         except:
#             percent = multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0]
#         print(
#             str(round(multi_orig_metrics['causal'][0] - multi_trans_metrics['causal'][0], 3)) + "(" + str(percent) + ")")
#         try:
#             percent = format((uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0]) / float(
#                 uni_orig_metrics[0]['causal'][0]), '.0%')
#         except:
#             percent = uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0]
#         print(str(round(uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0], 3)) + "(" + str(
#             percent) + ")")
#         try:
#             percent = format((uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0]) / float(
#                 uni_orig_metrics[1]['causal'][0]), '.0%')
#         except:
#             percent = uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0]
#         print(str(round(uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0], 3)) + "(" + str(
#             percent) + ")")
#     except:
#         print("no causal metric")

for i in range(0, len(processing_names)):
    process_name = processing_names[i]
    print("-->process_name", process_name)
    uni_orig_metrics = all_uni_orig_metrics[i]
    uni_trans_metrics = all_uni_trans_metrics[i]
    multi_orig_metrics = all_multi_orig_metrics[i]
    multi_trans_metrics = all_multi_trans_metrics[i]
    print("group metric")
    percent = format((uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0])/float(uni_orig_metrics[0]['group'][0]), '.0%')
    print(str(round(uni_orig_metrics[0]['group'][0] - uni_trans_metrics[0]['group'][0], 3)))
    percent = format((uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0]) / float(uni_orig_metrics[1]['group'][0]),'.0%')
    print(str(round(uni_orig_metrics[1]['group'][0] - uni_trans_metrics[1]['group'][0], 3)))
    percent = format((multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0]) / float(multi_orig_metrics['group'][0]),'.0%')
    print(str(round(multi_orig_metrics['group'][0] - multi_trans_metrics['group'][0], 3)))

    print(str(round(uni_trans_metrics[0]['acc'][0] - uni_orig_metrics[0]['acc'][0], 3)))
    print(str(round(uni_trans_metrics[1]['acc'][0] - uni_orig_metrics[1]['acc'][0], 3)))
    print(str(round(multi_trans_metrics['acc'][0] - multi_orig_metrics['acc'][0], 3)))

    try:
        print("causal metric")
        # percent = format((uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0]) / float(uni_orig_metrics[0]['causal'][0]), '.0%')
        percent = ''
        print(str(round(uni_orig_metrics[0]['causal'][0] - uni_trans_metrics[0]['causal'][0], 3)) + "" + str( percent) + "")
        # percent = format((uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0]) / float(uni_orig_metrics[1]['causal'][0]), '.0%')
        percent = ''
        print(str(round(uni_orig_metrics[1]['causal'][0] - uni_trans_metrics[1]['causal'][0], 3)) + "" + str(percent) + "")
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
# Plot.plot_abs_acc_multi_metric(all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics, all_multi_trans_metrics)
# 3 images: one for 'race', one for 'sex', one for 'race,sex'
Plot.plot_abs_acc_individual_metric(all_uni_orig_metrics, all_uni_trans_metrics, all_multi_orig_metrics, all_multi_trans_metrics)
