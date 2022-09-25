import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append("../")
from aif360.algorithms.inprocessing import GerryFairClassifier
from aif360.algorithms.inprocessing.gerryfair.clean import array_to_tuple
from aif360.algorithms.inprocessing.gerryfair.auditor import Auditor
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.datasets import AdultDataset, CompasDataset, GermanDataset
from aif360.datasets.compas_dataset1 import CompasDataset_1
from sklearn import svm
from sklearn import tree
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
from aif360.metrics import BinaryLabelDatasetMetric
from IPython.display import Image
import pickle
import matplotlib.pyplot as plt

from collections import defaultdict
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric


def metric_test_without_thresh(dataset, dataset_pred):
    metric = ClassificationMetric(dataset,
                                  dataset_pred,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
    metric_arrs = defaultdict(list)
    metric_arrs['tpr'].append(metric.true_positive_rate())
    metric_arrs['tnr'].append(metric.true_negative_rate())
    metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                   + metric.true_negative_rate()) / 2)
    metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
    metric_arrs['disp_imp'].append(metric.disparate_impact())
    metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
    metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
    metric_arrs['theil_ind'].append(metric.theil_index())
    return metric_arrs


def metric_test1(dataset, model, thresh_arr):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        all_classes = np.array([0, 1])
        pos_ind = np.where(all_classes == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        # changed coding
        pos_ind = 1
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        # changed coding
        # dataset_pred.labels = model.predict_classes(dataset.features)

        metric = ClassificationMetric(
            dataset, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
        metric_arrs['tpr'].append(metric.true_positive_rate())
        metric_arrs['tnr'].append(metric.true_negative_rate())
        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    return metric_arrs


# Make a function to print out accuracy and fairness metrics. This will be used throughout the tutorial.
def describe_metrics(metrics, thresh_arr):
    best_ind = np.argmax(metrics['bal_acc'])
    print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
    print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
#     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
    disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
    print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
    print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
    print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
    print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
    print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))



# load data set
data_set = AdultDataset()

dataset_train, dataset_test = data_set.split([0.7], shuffle=True)


# data_set = load_preproc_data_adult(sub_samp=1000, balance=True)
# max_iterations = 500
max_iterations = 50

# %% md

# %%

C = 100
print_flag = True
gamma = .005

fair_model = GerryFairClassifier(C=C, printflag=print_flag, gamma=gamma, fairness_def='FP',
                                 max_iters=max_iterations, heatmapflag=False)

# fit method
print("-->fitting model")
fair_model.fit(dataset_train, early_termination=True)

# predict method. If threshold in (0, 1) produces binary predictions
print("-->predicting model")
dataset_yhat = fair_model.predict(dataset_test, threshold=False)
print("-->dataset_yhat", dataset_yhat)



# output heatmap (brute force)
# replace None with the relative path if you want to save the plot
# print("-->plotting heatmap")
# fair_model.heatmapflag = True
# fair_model.heatmap_path = 'heatmap'
# fair_model.generate_heatmap(dataset_test, dataset_yhat.labels)
# Image(filename='{}.png'.format(fair_model.heatmap_path))

# %% md

# %%

gerry_metric = BinaryLabelDatasetMetric(data_set)
gamma_disparity = gerry_metric.rich_subgroup(array_to_tuple(dataset_yhat.labels), 'FP')
print("-->gamma_disparity", gamma_disparity)


# set to 50 iterations for fast running of notebook - set >= 1000 when running real experiments
pareto_iters = 50

def multiple_classifiers_pareto(dataset, gamma_list=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1], save_results=False,
                                iters=pareto_iters):
    ln_predictor = linear_model.LinearRegression()
    svm_predictor = svm.LinearSVR()
    # tree_predictor = tree.DecisionTreeRegressor(max_depth=3)
    # kernel_predictor = KernelRidge(alpha=1.0, gamma=1.0, kernel='rbf')
    predictor_dict = {'Linear': {'predictor': ln_predictor, 'iters': iters},
                      'SVR': {'predictor': svm_predictor, 'iters': iters}}
    # predictor_dict = {'Linear': {'predictor': ln_predictor, 'iters': iters},
    #                   'SVR': {'predictor': svm_predictor, 'iters': iters},
    #                   'Tree': {'predictor': tree_predictor, 'iters': iters},
    #                   'Kernel': {'predictor': kernel_predictor, 'iters': iters}}

    results_dict = {}

    for pred in predictor_dict:
        print('Curr Predictor: {}'.format(pred))
        predictor = predictor_dict[pred]['predictor']
        max_iters = predictor_dict[pred]['iters']
        fair_clf = GerryFairClassifier(C=100, printflag=True, gamma=1, predictor=predictor, max_iters=max_iters)
        fair_clf.printflag = False
        fair_clf.max_iters = max_iters
        errors, fp_violations, fn_violations = fair_clf.pareto(dataset, gamma_list)
        results_dict[pred] = {'errors': errors, 'fp_violations': fp_violations, 'fn_violations': fn_violations}
        plt.plot(errors, fp_violations, label=pred)

    if save_results:
        pickle.dump(results_dict, open('results_dict_' + str(gamma_list) + '_gammas' + str(gamma_list) + '.pkl', 'wb'))

    plt.xlabel('Error')
    plt.ylabel('Unfairness')
    plt.legend()
    plt.title('Error vs. Unfairness\n(Adult Dataset)')
    plt.savefig('gerryfair_pareto.png')
    plt.close()


multiple_classifiers_pareto(data_set)
Image(filename='gerryfair_pareto.png')

# %% md

def fp_vs_fn(dataset, gamma_list, iters):
    fp_auditor = Auditor(dataset, 'FP')
    fn_auditor = Auditor(dataset, 'FN')
    fp_violations = []
    fn_violations = []
    for g in gamma_list:
        print('gamma: {} '.format(g), end=" ")
        fair_model = GerryFairClassifier(C=100, printflag=False, gamma=g, max_iters=iters)
        fair_model.gamma = g
        fair_model.fit(dataset)
        predictions = array_to_tuple((fair_model.predict(dataset)).labels)
        _, fp_diff = fp_auditor.audit(predictions)
        _, fn_diff = fn_auditor.audit(predictions)
        fp_violations.append(fp_diff)
        fn_violations.append(fn_diff)

    plt.plot(fp_violations, fn_violations, label='adult')
    plt.xlabel('False Positive Disparity')
    plt.ylabel('False Negative Disparity')
    plt.legend()
    plt.title('FP vs FN Unfairness')
    plt.savefig('gerryfair_fp_fn.png')
    plt.close()


gamma_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05]
fp_vs_fn(data_set, gamma_list, pareto_iters)
Image(filename='gerryfair_fp_fn.png')

# %%


