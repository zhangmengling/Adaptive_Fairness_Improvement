import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization
# import torch
from sklearn.metrics import accuracy_score
from collections import defaultdict
import matplotlib.pyplot as plt
from aif360.metrics import BinaryLabelDatasetMetric, utils
import random
seed = 1
random.seed(seed)
np.random.seed(seed)
# tf.random.set_random_seed(seed)

from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.datasets.multiclass_label_dataset import MulticlassLabelDataset
import math
from itertools import permutations, combinations_with_replacement

nb_classes = 2
input_num = 1000
# SAMPLING_THRE = 500  # 500 for causal discrimination test/ 200 for group discrimination test
SAMPLING_THRE = 10
ERROR_THRE = 0.05
Z_VALUE = 1.96  # when conf=95%
# Z_VALUE= 2.576  # when conf=99%

class CausalClassficationMetric(ClassificationMetric):
    def __init__(self, dataset, classified_dataset,
                 unprivileged_groups=None, privileged_groups=None):
        if not isinstance(dataset, BinaryLabelDataset) and not isinstance(dataset, MulticlassLabelDataset):
            raise TypeError("'dataset' should be a BinaryLabelDataset or a MulticlassLabelDataset")

        # sets self.dataset, self.unprivileged_groups, self.privileged_groups
        # print("-->dataset", dataset)
        # print("-->unprivileged_groups", unprivileged_groups)
        # print("-->privileged_groups", privileged_groups)
        super(ClassificationMetric, self).__init__(dataset,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)

        if isinstance(classified_dataset, BinaryLabelDataset) or isinstance(classified_dataset, MulticlassLabelDataset):
            self.classified_dataset = classified_dataset
        else:
            raise TypeError("'classified_dataset' should be a "
                            "BinaryLabelDataset or a MulticlassLabelDataset.")

        if isinstance(self.classified_dataset, MulticlassLabelDataset):
            fav_label_value = 1.
            unfav_label_value = 0.

            self.classified_dataset = self.classified_dataset.copy()
            # Find all the labels which match any of the favorable labels
            fav_idx = np.logical_or.reduce(
                np.equal.outer(self.classified_dataset.favorable_label, self.classified_dataset.labels))
            # Replace labels with corresponding values
            self.classified_dataset.labels = np.where(fav_idx, fav_label_value, unfav_label_value)

            self.classified_dataset.favorable_label = float(fav_label_value)
            self.classified_dataset.unfavorable_label = float(unfav_label_value)

        # Verify if everything except the predictions and metadata are the same
        # for the two datasets
        with self.dataset.temporarily_ignore('labels', 'scores'):
            if self.dataset != self.classified_dataset:
                raise ValueError("The two datasets are expected to differ only "
                                 "in 'labels' or 'scores'.")

    def margin_of_error(self, inputs, input_labels):
        num_positive = input_labels.count(1)
        num_inputs = len(inputs)
        pro_positive = float(num_positive)/float(num_inputs)
        # print("-->positive num: %d and inputs num: %d" % (num_positive, num_inputs))
        margin_error = Z_VALUE*math.sqrt((pro_positive*(1-pro_positive))/num_inputs)
        return margin_error, pro_positive

    def sensitive_value_filter(self, sensitive_indexs, sensitive_values, all_inputs, all_labels):
        filter_inputs = []
        filter_labels = []
        for i in range(0, len(all_inputs)):
            input = all_inputs[i]
            label = all_labels[i]
            satisfy =True
            for i in range(0, len(sensitive_indexs)):
                index = sensitive_indexs[i]
                value = sensitive_values[i]
                if input[index] != value:
                    satisfy = False
            if satisfy == True:
                filter_inputs.append(input)
                filter_labels.append(label)
        return filter_inputs, filter_labels

    def univariate_group_discrimination(self, *args):
        probabilities = []
        sensitive_attribute = list(self.privileged_groups[0].keys())[0]
        sensitive_index = self.dataset.feature_names.index(sensitive_attribute)
        print("-->sensitive_index:", sensitive_index)
        sensitive_values = set([data[sensitive_index] for data in self.dataset.features])
        for value in sensitive_values:
            print("-->value", value)
            all_inputs, all_labels = self.sensitive_value_filter(sensitive_indexs=[sensitive_index], sensitive_values=[value],
                                                     all_inputs=self.dataset.features, all_labels=self.classified_dataset.labels)
            print("-->length for attribute %s value %d is %d" % (sensitive_attribute, value, len(all_inputs)))
            for i in range(SAMPLING_THRE, len(all_inputs)):
                inputs = all_inputs[:i]
                labels = all_labels[:i]
                margin_error, probability = self.margin_of_error(inputs=list(inputs), input_labels=list(labels))
                # if margin_error < ERROR_THRE:
                #     break
            probabilities.append(probability)
            print("-->margin_error", margin_error)
        print("-->probabilities", probabilities)
        return max(probabilities) - min(probabilities)

    def multivariate_group_discrimination(self, *args):
        probabilities = []
        sensitive_attributes = []
        sensitive_indexs = []
        sensitive_values = []
        for group in self.privileged_groups:
            sensitive_attribute = list(group.keys())[0]
            sensitive_attributes.append(sensitive_attribute)
            sensitive_index = self.dataset.feature_names.index(sensitive_attribute)
            sensitive_indexs.append(sensitive_index)
            sensitive_values.append(set([data[sensitive_index] for data in self.dataset.features]))
            # print("-->length of dataset", len(self.dataset.features))

        print("-->sensitive_attributes", sensitive_attributes)
        print("-->sensitive_indexs", sensitive_indexs)
        print("-->sensitive_values", sensitive_values)

        indexs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for condition in indexs:
            # e.g. [0, 0]
            values = []
            for i in range(0, len(condition)):
                values.append(list(sensitive_values[i])[condition[i]])
            print("-->values' condition:", values)
            all_inputs, all_labels = self.sensitive_value_filter(sensitive_indexs=sensitive_indexs,
                                                                 sensitive_values=values,
                                                                 all_inputs=self.dataset.features,
                                                                 all_labels=self.classified_dataset.labels)
            print("-->length for attribute %s value %s is %d" % (sensitive_attributes, values, len(all_inputs)))
            if len(all_inputs) == 0 or SAMPLING_THRE >= len(all_inputs):
            # if len(all_inputs) == 0:
                print("-->length", len(all_inputs))
                print("-->have no satisfied input!")
                continue
            for i in range(SAMPLING_THRE, len(all_inputs)):
                inputs = all_inputs[:i]
                labels = all_labels[:i]
                margin_error, probability = self.margin_of_error(inputs=list(inputs), input_labels=list(labels))
                # if margin_error < ERROR_THRE:
                #     break
            probabilities.append(probability)
            print("-->probability", probability)
            print("-->margin_error", margin_error)
        print("-->probabilities:", probabilities)
        return max(probabilities) - min(probabilities)

    def causal_distort(self, input, sensitive_indexs, sensitive_values):
        distorted_inputs = []
        if isinstance(sensitive_values[0], set):
            value_conditions = [[0, 0], [0, 1], [1, 0], [1, 1]]
        else:
            value_conditions = [[0], [1]]
        for condition in value_conditions:
            new_input = input.copy()
            for i in range(0, len(condition)):
                new_input[sensitive_indexs] = condition[i]
            distorted_inputs.append(new_input)
            # if list(new_input) != list(input):
            #     distorted_inputs.append(new_input)
        return distorted_inputs
    
    def causal_distort_scaler(self, input, sensitive_indexs, sensitive_values, scaler_protected_feautres):
        print("-->causal_distort_scaler")
        distorted_inputs = []
        if isinstance(sensitive_values[0], set):
            value_conditions = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
            scaler_value_conditions = []
            for condition in value_conditions:
                scaler_condition = scaler_protected_feautres[str(condition)]
                scaler_value_conditions.append(scaler_condition)
            value_conditions = scaler_value_conditions
        else:
            value_conditions = [[0.0], [1.0]]
            scaler_value_conditions = []
            for condition in value_conditions:
                scaler_condition = scaler_protected_feautres[str(condition)]
                scaler_value_conditions.append(scaler_condition)
            value_conditions = scaler_value_conditions
            print("-->value_conditions", value_conditions)
        for condition in value_conditions:
            new_input = input.copy()
            for i in range(0, len(condition)):
                new_input[sensitive_indexs] = condition[i]
            # print("-->input", input, type(input))
            # print("-->new_input", new_input, type(new_input))
            if list(new_input) != list(input):
                distorted_inputs.append(new_input)
        return distorted_inputs

    def univariate_causal_discrimination(self, model, thresh_arr, **kwargs):
        sampling_thre = 100
        try:
            scaler_protected_features = kwargs['scaler_protected_features']
        except:
            print("-->no scaler protected features in univariate_causal_discrimination")
        count = 0
        sensitive_attribute = list(self.privileged_groups[0].keys())[0]
        sensitive_index = self.dataset.feature_names.index(sensitive_attribute)
        print("-->sensitive_index:", sensitive_index)
        sensitive_values = list(set([data[sensitive_index] for data in self.dataset.features]))
        print("-->sensitive_values", sensitive_values, type(sensitive_values))
        print("-->lenghth of dataset", len(self.dataset.features))
        all_inputs = []
        for i in range(0, len(self.dataset.features)):
            # print("-->i", i)
            predicted_labels = []
            input = self.dataset.features[i]
            all_inputs.append(input)
            label = self.classified_dataset.labels[i]
            # print("-->input", input)
            try:
                distorted_inputs = self.causal_distort_scaler(input=input,
                                                              sensitive_indexs=sensitive_index,
                                                              sensitive_values=sensitive_values,
                                                              scaler_protected_feautres=scaler_protected_features)
            except:
                distorted_inputs = self.causal_distort(input=input,
                                                       sensitive_indexs=sensitive_index,
                                                       sensitive_values=sensitive_values)
            for distorted_input in distorted_inputs:
                try:
                    # sklearn classifier
                    y_val_pred_prob = model.predict_proba(np.array([distorted_input]))
                except:
                    try:
                        # aif360 inprocessing algorithm
                        y_val_pred_prob = model.predict([distorted_input]).scores
                    except:
                        distorted_input.astype(int)
                        distorted_input_dataset = self.dataset.copy()
                        distorted_input_dataset.features = [distorted_input]
                        distorted_input_dataset.labels = [label]
                        try:
                            y_val_pred_prob = model.predict_one_input(distorted_input_dataset).scores
                        except:
                            try:
                                y_val_pred_prob = model.predict(distorted_input_dataset).scores
                            except:
                                # print("-->inprocessing: prejudice remover's rediction")
                                distorted_input.astype(int)
                                distorted_input_dataset = self.dataset.copy()
                                distorted_input_dataset.features = distorted_input_dataset.features[:2]
                                distorted_input_dataset.labels = distorted_input_dataset.labels[:2]
                                distorted_input_dataset.protected_attributes = distorted_input_dataset.protected_attributes[
                                                                               :2]
                                distorted_input_dataset.features[0] = distorted_input
                                distorted_input_dataset.labels[0] = 1
                                sensitive_indexs = [distorted_input_dataset.feature_names.index(attr) for attr in
                                                    self.classified_dataset.protected_attribute_names]
                                distorted_input_dataset.protected_attributes[0] = [distorted_input[index] for index in
                                                                                   sensitive_indexs]
                                # y_val_pred_prob = model.predict(distorted_input_dataset).scores[0]
                                y_val_pred_prob = model.predict(distorted_input_dataset.features)

                if thresh_arr == None:
                    distorted_input.astype(int)
                    distorted_input_dataset = self.dataset.copy()
                    distorted_input_dataset.features = [distorted_input]
                    distorted_input_dataset.labels = [label]
                    dataset_transf_valid_pred = model.predict(distorted_input_dataset)
                    predicted_label = dataset_transf_valid_pred.labels[0]
                    predicted_labels.append(predicted_label)
                else:
                    for thresh in thresh_arr:
                        # changed coding
                        pos_ind = 1
                        try:
                            y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
                        except:
                            y_val_pred = (np.array(y_val_pred_prob) > thresh).astype(np.float64)
                        predicted_label = y_val_pred[0]
                        predicted_labels.append(predicted_label)
            if predicted_labels.count(label) < len(predicted_labels):
                count += 1
            margin_error, probability = self.margin_of_error(inputs=list(all_inputs), input_labels=[1]*count)
            if margin_error < ERROR_THRE and i >= sampling_thre:
                break
        print("%d out of %d inputs can be distorted"% (count, len(all_inputs)))
        print("-->margin of error", margin_error)
        return probability

    def multivariate_causal_discrimination(self, model, thresh_arr, **kwargs):
        sampling_thre = 100
        try:
            scaler_protected_features = kwargs['scaler_protected_features']
        except:
            print("-->no scaler protected features in multivariate_causal_discrimination")
        count = 0
        sensitive_attributes = []
        sensitive_indexs = []
        sensitive_values = []
        for group in self.privileged_groups:
            sensitive_attribute = list(group.keys())[0]
            sensitive_attributes.append(sensitive_attribute)
            sensitive_index = self.dataset.feature_names.index(sensitive_attribute)
            sensitive_indexs.append(sensitive_index)
            sensitive_values.append(set([data[sensitive_index] for data in self.dataset.features]))
        all_inputs = []
        # print("-->sensitive_indexs", sensitive_indexs)
        # print("-->sensitive_values", sensitive_values)
        print("-->length:", len(self.dataset.features))
        for i in range(0, len(self.dataset.features)):
            print("-->i", i)
            predicted_labels = []
            input = self.dataset.features[i]
            # print("-->sensitive features", [input[i] for i in sensitive_indexs])
            all_inputs.append(input)
            label = self.classified_dataset.labels[i]
            try:
                distorted_inputs = self.causal_distort_scaler(input=input,
                                                       sensitive_indexs=sensitive_indexs,
                                                       sensitive_values=sensitive_values,
                                                       scaler_protected_feautres=scaler_protected_features)
            except:
                distorted_inputs = self.causal_distort(input=input,
                                                       sensitive_indexs=sensitive_indexs,
                                                       sensitive_values=sensitive_values)
            # print("-->distorted_inputs feature", [[input[i] for i in sensitive_indexs] for input in distorted_inputs])
            for distorted_input in distorted_inputs:
                try:
                    # sklearn classifier
                    y_val_pred_prob = model.predict_proba(np.array([distorted_input]))
                except:
                    try:
                        # aif360 inprocessing algorithm
                        y_val_pred_prob = model.predict([distorted_input]).scores
                    except:
                        try:
                            distorted_input.astype(int)
                            distorted_input_dataset = self.dataset.copy()
                            distorted_input_dataset.features = [distorted_input]
                            distorted_input_dataset.labels = [label]
                            y_val_pred_prob = model.predict(distorted_input_dataset).scores
                        except:
                            try:
                                distorted_input.astype(int)
                                distorted_input_dataset = self.dataset.copy()
                                distorted_input_dataset.features = [distorted_input]
                                distorted_input_dataset.labels = [label]
                                # y_val_pred_prob = model.predict_one_input(distorted_input_dataset).scores
                                y_val_pred_prob = model.predict(distorted_input_dataset.features)
                            except:
                                # inprocessing: prejudice remover
                                # print("-->inprocessing: prejudice remover's rediction")
                                distorted_input.astype(int)
                                distorted_input_dataset = self.dataset.copy()
                                distorted_input_dataset.features = distorted_input_dataset.features[:2]
                                distorted_input_dataset.labels = distorted_input_dataset.labels[:2]
                                distorted_input_dataset.protected_attributes = distorted_input_dataset.protected_attributes[:2]
                                distorted_input_dataset.features[0] = distorted_input
                                distorted_input_dataset.labels[0] = 1
                                sensitive_indexs = [distorted_input_dataset.feature_names.index(attr) for attr in self.classified_dataset.protected_attribute_names]
                                distorted_input_dataset.protected_attributes[0] = [distorted_input[index] for index in sensitive_indexs]
                                # y_val_pred_prob = model.predict(distorted_input_dataset).scores[0]
                                y_val_pred_prob = model.predict(distorted_input_dataset.features)

                if thresh_arr == None:
                    distorted_input.astype(int)
                    distorted_input_dataset = self.dataset.copy()
                    distorted_input_dataset.features = [distorted_input]
                    distorted_input_dataset.labels = [label]
                    dataset_transf_valid_pred = model.predict(distorted_input_dataset)
                    predicted_label = dataset_transf_valid_pred.labels[0]
                    predicted_labels.append(predicted_label)
                else:
                    for thresh in thresh_arr:
                        # changed coding
                        pos_ind = 1
                        try:
                            y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
                        except:
                            y_val_pred = (np.array(y_val_pred_prob) > thresh).astype(np.float64)
                        predicted_label = y_val_pred[0]
                        predicted_labels.append(predicted_label)
            # print("-->predicted_labels", predicted_labels)
            # print("-->label", label)
            if predicted_labels.count(label) != len(predicted_labels):
                # print("-->predicted_labels", predicted_labels)
                # print("-->label", label)
                count += 1
            margin_error, probability = self.margin_of_error(inputs=list(all_inputs), input_labels=[1]*count)
            if margin_error < ERROR_THRE and len(all_inputs) >= sampling_thre:
                break
        print("%d out of %d inputs can be distorted"% (count, len(all_inputs)))
        print("-->margin of error", margin_error)
        return probability


    def binary_confusion_matrix(self, privileged=None):
        """Compute the number of true/false positives/negatives, optionally
        conditioned on protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Returns:
            dict: Number of true positives, false positives, true negatives,
            false negatives (optionally conditioned).
        """
        condition = self._to_condition(privileged)

        return utils.compute_num_TF_PN(self.dataset.protected_attributes,
                                       self.dataset.labels, self.classified_dataset.labels,
                                       self.dataset.instance_weights,
                                       self.dataset.protected_attribute_names,
                                       self.dataset.favorable_label, self.dataset.unfavorable_label,
                                       condition=condition)

    def generalized_binary_confusion_matrix(self, privileged=None):
        """Compute the number of generalized true/false positives/negatives,
        optionally conditioned on protected attributes. Generalized counts are
        based on scores and not on the hard predictions.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Returns:
            dict: Number of generalized true positives, generalized false
            positives, generalized true negatives, generalized false negatives
            (optionally conditioned).
        """
        condition = self._to_condition(privileged)

        return utils.compute_num_gen_TF_PN(self.dataset.protected_attributes,
                                           self.dataset.labels, self.classified_dataset.scores,
                                           self.dataset.instance_weights,
                                           self.dataset.protected_attribute_names,
                                           self.dataset.favorable_label, self.dataset.unfavorable_label,
                                           condition=condition)

    def num_true_positives(self, privileged=None):
        r"""Return the number of instances in the dataset where both the
        predicted and true labels are 'favorable',
        :math:`TP = \sum_{i=1}^n \mathbb{1}[y_i = \text{favorable}]\mathbb{1}[\hat{y}_i = \text{favorable}]`,
        optionally conditioned on protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.binary_confusion_matrix(privileged=privileged)['TP']

    def num_false_positives(self, privileged=None):
        r""":math:`FP = \sum_{i=1}^n \mathbb{1}[y_i = \text{unfavorable}]\mathbb{1}[\hat{y}_i = \text{favorable}]`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.binary_confusion_matrix(privileged=privileged)['FP']

    def num_false_negatives(self, privileged=None):
        r""":math:`FN = \sum_{i=1}^n \mathbb{1}[y_i = \text{favorable}]\mathbb{1}[\hat{y}_i = \text{unfavorable}]`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.binary_confusion_matrix(privileged=privileged)['FN']

    def num_true_negatives(self, privileged=None):
        r""":math:`TN = \sum_{i=1}^n \mathbb{1}[y_i = \text{unfavorable}]\mathbb{1}[\hat{y}_i = \text{unfavorable}]`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.binary_confusion_matrix(privileged=privileged)['TN']

    def num_generalized_true_positives(self, privileged=None):
        """Return the generalized number of true positives, :math:`GTP`, the
        weighted sum of predicted scores where true labels are 'favorable',
        optionally conditioned on protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.generalized_binary_confusion_matrix(
            privileged=privileged)['GTP']

    def num_generalized_false_positives(self, privileged=None):
        """Return the generalized number of false positives, :math:`GFP`, the
        weighted sum of predicted scores where true labels are 'unfavorable',
        optionally conditioned on protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups` must be
                must be provided at initialization to condition on them.
        """
        return self.generalized_binary_confusion_matrix(
            privileged=privileged)['GFP']

    def num_generalized_false_negatives(self, privileged=None):
        """Return the generalized number of false negatives, :math:`GFN`, the
        weighted sum of 1 - predicted scores where true labels are 'favorable',
        optionally conditioned on protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.generalized_binary_confusion_matrix(
            privileged=privileged)['GFN']

    def num_generalized_true_negatives(self, privileged=None):
        """Return the generalized number of true negatives, :math:`GTN`, the
        weighted sum of 1 - predicted scores where true labels are 'unfavorable',
        optionally conditioned on protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.generalized_binary_confusion_matrix(
            privileged=privileged)['GTN']

    def performance_measures(self, privileged=None):
        """Compute various performance measures on the dataset, optionally
        conditioned on protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Returns:
            dict: True positive rate, true negative rate, false positive rate,
            false negative rate, positive predictive value, negative predictive
            value, false discover rate, false omission rate, and accuracy
            (optionally conditioned).
        """
        TP = self.num_true_positives(privileged=privileged)
        FP = self.num_false_positives(privileged=privileged)
        FN = self.num_false_negatives(privileged=privileged)
        TN = self.num_true_negatives(privileged=privileged)
        GTP = self.num_generalized_true_positives(privileged=privileged)
        GFP = self.num_generalized_false_positives(privileged=privileged)
        GFN = self.num_generalized_false_negatives(privileged=privileged)
        GTN = self.num_generalized_true_negatives(privileged=privileged)
        P = self.num_positives(privileged=privileged)
        N = self.num_negatives(privileged=privileged)

        return dict(
            TPR=TP / P, TNR=TN / N, FPR=FP / N, FNR=FN / P,
            GTPR=GTP / P, GTNR=GTN / N, GFPR=GFP / N, GFNR=GFN / P,
            PPV=TP / (TP + FP) if (TP + FP) > 0.0 else np.float64(0.0),
            NPV=TN / (TN + FN) if (TN + FN) > 0.0 else np.float64(0.0),
            FDR=FP / (FP + TP) if (FP + TP) > 0.0 else np.float64(0.0),
            FOR=FN / (FN + TN) if (FN + TN) > 0.0 else np.float64(0.0),
            ACC=(TP + TN) / (P + N) if (P + N) > 0.0 else np.float64(0.0)
        )

    def true_positive_rate(self, privileged=None):
        """Return the ratio of true positives to positive examples in the
        dataset, :math:`TPR = TP/P`, optionally conditioned on protected
        attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['TPR']

    def false_positive_rate(self, privileged=None):
        """:math:`FPR = FP/N`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['FPR']

    def false_negative_rate(self, privileged=None):
        """:math:`FNR = FN/P`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['FNR']

    def true_negative_rate(self, privileged=None):
        """:math:`TNR = TN/N`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['TNR']

    def generalized_true_positive_rate(self, privileged=None):
        """Return the ratio of generalized true positives to positive examples
        in the dataset, :math:`GTPR = GTP/P`, optionally conditioned on
        protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['GTPR']

    def generalized_false_positive_rate(self, privileged=None):
        """:math:`GFPR = GFP/N`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['GFPR']

    def generalized_false_negative_rate(self, privileged=None):
        """:math:`GFNR = GFN/P`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['GFNR']

    def generalized_true_negative_rate(self, privileged=None):
        """:math:`GTNR = GTN/N`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['GTNR']

    def positive_predictive_value(self, privileged=None):
        """:math:`PPV = TP/(TP + FP)`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['PPV']

    def false_discovery_rate(self, privileged=None):
        """:math:`FDR = FP/(TP + FP)`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['FDR']

    def false_omission_rate(self, privileged=None):
        """:math:`FOR = FN/(TN + FN)`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['FOR']

    def negative_predictive_value(self, privileged=None):
        """:math:`NPV = TN/(TN + FN)`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['NPV']

    def accuracy(self, privileged=None):
        """:math:`ACC = (TP + TN)/(P + N)`.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.performance_measures(privileged=privileged)['ACC']

    def error_rate(self, privileged=None):
        """:math:`ERR = (FP + FN)/(P + N)`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return 1. - self.accuracy(privileged=privileged)

    def true_positive_rate_difference(self):
        r""":math:`TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}`
        """
        return self.difference(self.true_positive_rate)

    def false_positive_rate_difference(self):
        r""":math:`FPR_{D = \text{unprivileged}} - FPR_{D = \text{privileged}}`
        """
        return self.difference(self.false_positive_rate)

    def false_negative_rate_difference(self):
        r""":math:`FNR_{D = \text{unprivileged}} - FNR_{D = \text{privileged}}`
        """
        return self.difference(self.false_negative_rate)

    def false_omission_rate_difference(self):
        r""":math:`FOR_{D = \text{unprivileged}} - FOR_{D = \text{privileged}}`
        """
        return self.difference(self.false_omission_rate)

    def false_discovery_rate_difference(self):
        r""":math:`FDR_{D = \text{unprivileged}} - FDR_{D = \text{privileged}}`
        """
        return self.difference(self.false_discovery_rate)

    def false_positive_rate_ratio(self):
        r""":math:`\frac{FPR_{D = \text{unprivileged}}}{FPR_{D = \text{privileged}}}`
        """
        return self.ratio(self.false_positive_rate)

    def false_negative_rate_ratio(self):
        r""":math:`\frac{FNR_{D = \text{unprivileged}}}{FNR_{D = \text{privileged}}}`
        """
        return self.ratio(self.false_negative_rate)

    def false_omission_rate_ratio(self):
        r""":math:`\frac{FOR_{D = \text{unprivileged}}}{FOR_{D = \text{privileged}}}`
        """
        return self.ratio(self.false_omission_rate)

    def false_discovery_rate_ratio(self):
        r""":math:`\frac{FDR_{D = \text{unprivileged}}}{FDR_{D = \text{privileged}}}`
        """
        return self.ratio(self.false_discovery_rate)

    def average_odds_difference(self):
        r"""Average of difference in FPR and TPR for unprivileged and privileged
        groups:

        .. math::

           \tfrac{1}{2}\left[(FPR_{D = \text{unprivileged}} - FPR_{D = \text{privileged}})
           + (TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}))\right]

        A value of 0 indicates equality of odds.
        """
        return 0.5 * (self.difference(self.false_positive_rate)
                      + self.difference(self.true_positive_rate))

    def average_abs_odds_difference(self):
        r"""Average of absolute difference in FPR and TPR for unprivileged and
        privileged groups:

        .. math::

           \tfrac{1}{2}\left[|FPR_{D = \text{unprivileged}} - FPR_{D = \text{privileged}}|
           + |TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}|\right]

        A value of 0 indicates equality of odds.
        """
        return 0.5 * (np.abs(self.difference(self.false_positive_rate))
                      + np.abs(self.difference(self.true_positive_rate)))

    def error_rate_difference(self):
        r"""Difference in error rates for unprivileged and privileged groups,
        :math:`ERR_{D = \text{unprivileged}} - ERR_{D = \text{privileged}}`.
        """
        return self.difference(self.error_rate)

    def error_rate_ratio(self):
        r"""Ratio of error rates for unprivileged and privileged groups,
        :math:`\frac{ERR_{D = \text{unprivileged}}}{ERR_{D = \text{privileged}}}`.
        """
        return self.ratio(self.error_rate)

    def num_pred_positives(self, privileged=None):
        r""":math:`\sum_{i=1}^n \mathbb{1}[\hat{y}_i = \text{favorable}]`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        condition = self._to_condition(privileged)

        return utils.compute_num_pos_neg(
            self.classified_dataset.protected_attributes,
            self.classified_dataset.labels,
            self.classified_dataset.instance_weights,
            self.classified_dataset.protected_attribute_names,
            self.classified_dataset.favorable_label,
            condition=condition)

    def num_pred_negatives(self, privileged=None):
        r""":math:`\sum_{i=1}^n \mathbb{1}[\hat{y}_i = \text{unfavorable}]`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        condition = self._to_condition(privileged)

        return utils.compute_num_pos_neg(
            self.classified_dataset.protected_attributes,
            self.classified_dataset.labels,
            self.classified_dataset.instance_weights,
            self.classified_dataset.protected_attribute_names,
            self.classified_dataset.unfavorable_label,
            condition=condition)

    def selection_rate(self, privileged=None):
        r""":math:`Pr(\hat{Y} = \text{favorable})`

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return (self.num_pred_positives(privileged=privileged)
                / self.num_instances(privileged=privileged))

    def disparate_impact(self):
        r"""
        .. math::
           \frac{Pr(\hat{Y} = 1 | D = \text{unprivileged})}
           {Pr(\hat{Y} = 1 | D = \text{privileged})}
        """
        return self.ratio(self.selection_rate)

    def statistical_parity_difference(self):
        r"""
        .. math::
           Pr(\hat{Y} = 1 | D = \text{unprivileged})
           - Pr(\hat{Y} = 1 | D = \text{privileged})
        """
        return self.difference(self.selection_rate)

    def generalized_entropy_index(self, alpha=2):
        r"""Generalized entropy index is proposed as a unified individual and
        group fairness measure in [3]_.  With :math:`b_i = \hat{y}_i - y_i + 1`:

        .. math::

           \mathcal{E}(\alpha) = \begin{cases}
               \frac{1}{n \alpha (\alpha-1)}\sum_{i=1}^n\left[\left(\frac{b_i}{\mu}\right)^\alpha - 1\right],& \alpha \ne 0, 1,\\
               \frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu},& \alpha=1,\\
               -\frac{1}{n}\sum_{i=1}^n\ln\frac{b_{i}}{\mu},& \alpha=0.
           \end{cases}

        Args:
            alpha (int): Parameter that regulates the weight given to distances
                between values at different parts of the distribution.

        References:
            .. [3] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
               "A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via Inequality Indices,"
               ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.
        """
        y_pred = self.classified_dataset.labels.ravel()
        y_true = self.dataset.labels.ravel()
        y_pred = (y_pred == self.classified_dataset.favorable_label).astype(
            np.float64)
        y_true = (y_true == self.dataset.favorable_label).astype(np.float64)
        b = 1 + y_pred - y_true

        if alpha == 1:
            # moving the b inside the log allows for 0 values
            return np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b))
        elif alpha == 0:
            return -np.mean(np.log(b / np.mean(b)) / np.mean(b))
        else:
            return np.mean((b / np.mean(b)) ** alpha - 1) / (alpha * (alpha - 1))

    def _between_group_generalized_entropy_index(self, groups, alpha=2):
        r"""Between-group generalized entropy index is proposed as a group
        fairness measure in [2]_ and is one of two terms that the generalized
        entropy index decomposes to.

        Args:
            groups (list): A list of groups over which to calculate this metric.
                Groups should be disjoint. By default, this will use the
                `privileged_groups` and `unprivileged_groups` as the only two
                groups.
            alpha (int): See :meth:`generalized_entropy_index`.

        References:
            .. [2] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
               "A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via Inequality Indices,"
               ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.
        """
        b = np.zeros(self.dataset.labels.size, dtype=np.float64)

        for group in groups:
            classified_group = utils.compute_boolean_conditioning_vector(
                self.classified_dataset.protected_attributes,
                self.classified_dataset.protected_attribute_names,
                condition=group)
            true_group = utils.compute_boolean_conditioning_vector(
                self.dataset.protected_attributes,
                self.dataset.protected_attribute_names,
                condition=group)
            # ignore if there are no members of this group present
            if not np.any(true_group):
                continue
            y_pred = self.classified_dataset.labels[classified_group].ravel()
            y_true = self.dataset.labels[true_group].ravel()
            y_pred = (y_pred == self.classified_dataset.favorable_label).astype(
                np.float64)
            y_true = (y_true == self.dataset.favorable_label).astype(np.float64)
            b[true_group] = np.mean(1 + y_pred - y_true)

        if alpha == 1:
            return np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b))
        elif alpha == 0:
            return -np.mean(np.log(b / np.mean(b)) / np.mean(b))
        else:
            return np.mean((b / np.mean(b)) ** alpha - 1) / (alpha * (alpha - 1))

    def between_all_groups_generalized_entropy_index(self, alpha=2):
        """Between-group generalized entropy index that uses all combinations of
        groups based on `self.dataset.protected_attributes`. See
        :meth:`_between_group_generalized_entropy_index`.

        Args:
            alpha (int): See :meth:`generalized_entropy_index`.
        """
        all_values = list(map(np.concatenate, zip(
            self.dataset.privileged_protected_attributes,
            self.dataset.unprivileged_protected_attributes)))
        groups = [[dict(zip(self.dataset.protected_attribute_names, vals))]
                  for vals in product(*all_values)]
        return self._between_group_generalized_entropy_index(groups=groups,
                                                             alpha=alpha)

    def between_group_generalized_entropy_index(self, alpha=2):
        """Between-group generalized entropy index that uses
        `self.privileged_groups` and `self.unprivileged_groups` as the only two
        groups. See :meth:`_between_group_generalized_entropy_index`.

        Args:
            alpha (int): See :meth:`generalized_entropy_index`.
        """
        groups = [self._to_condition(False), self._to_condition(True)]
        return self._between_group_generalized_entropy_index(groups=groups,
                                                             alpha=alpha)

    def theil_index(self):
        r"""The Theil index is the :meth:`generalized_entropy_index` with
        :math:`\alpha = 1`.
        """
        return self.generalized_entropy_index(alpha=1)

    def coefficient_of_variation(self):
        r"""The coefficient of variation is two times the square root of the
        :meth:`generalized_entropy_index` with :math:`\alpha = 2`.
        """
        return 2 * np.sqrt(self.generalized_entropy_index(alpha=2))

    def between_group_theil_index(self):
        r"""The between-group Theil index is the
        :meth:`between_group_generalized_entropy_index` with :math:`\alpha = 1`.
        """
        return self.between_group_generalized_entropy_index(alpha=1)

    def between_group_coefficient_of_variation(self):
        r"""The between-group coefficient of variation is two times the square
        root of the :meth:`between_group_generalized_entropy_index` with
        :math:`\alpha = 2`.
        """
        return 2 * np.sqrt(self.between_group_generalized_entropy_index(alpha=2))

    def between_all_groups_theil_index(self):
        r"""The between-group Theil index is the
        :meth:`between_all_groups_generalized_entropy_index` with
        :math:`\alpha = 1`.
        """
        return self.between_all_groups_generalized_entropy_index(alpha=1)

    def between_all_groups_coefficient_of_variation(self):
        r"""The between-group coefficient of variation is two times the square
        root of the :meth:`between_all_groups_generalized_entropy_index` with
        :math:`\alpha = 2`.
        """
        return 2 * np.sqrt(self.between_all_groups_generalized_entropy_index(
            alpha=2))

    def differential_fairness_bias_amplification(self, concentration=1.0):
        """Bias amplification is the difference in smoothed EDF between the
        classifier and the original dataset. Positive values mean the bias
        increased due to the classifier.

        Args:
            concentration (float, optional): Concentration parameter for
                Dirichlet smoothing. Must be non-negative.
        """
        ssr = self._smoothed_base_rates(self.classified_dataset.labels,
                                        concentration)

        def pos_ratio(i, j):
            return abs(np.log(ssr[i]) - np.log(ssr[j]))

        def neg_ratio(i, j):
            return abs(np.log(1 - ssr[i]) - np.log(1 - ssr[j]))

        edf_clf = max(max(pos_ratio(i, j), neg_ratio(i, j))
                      for i in range(len(ssr)) for j in range(len(ssr)) if i != j)
        edf_data = self.smoothed_empirical_differential_fairness(concentration)

        return edf_clf - edf_data

        # ============================== ALIASES ===================================

    def equal_opportunity_difference(self):
        """Alias of :meth:`true_positive_rate_difference`."""
        return self.true_positive_rate_difference()

    def power(self, privileged=None):
        """Alias of :meth:`num_true_positives`."""
        return self.num_true_positives(privileged=privileged)

    def precision(self, privileged=None):
        """Alias of :meth:`positive_predictive_value`."""
        return self.positive_predictive_value(privileged=privileged)

    def recall(self, privileged=None):
        """Alias of :meth:`true_positive_rate`."""
        return self.true_positive_rate(privileged=privileged)

    def sensitivity(self, privileged=None):
        """Alias of :meth:`true_positive_rate`."""
        return self.true_positive_rate(privileged=privileged)

    def specificity(self, privileged=None):
        """Alias of :meth:`true_negative_rate`."""
        return self.true_negative_rate(privileged=privileged)
