import os
import pandas as pd
from aif360.datasets import StandardDataset
import random
import numpy as np
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.metrics.dataset_metric import DatasetMetric
# need to use DatasetMetric._to_condition(privileged)
seed = 1
random.seed(seed)
np.random.seed(seed)
import operator



default_mappings = {
    'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},
                                 {1.0: 'Old', 0.0: 'Young'}],
}

def default_preprocessing(df):
    """Adds a derived sex attribute based on personal_status."""
    # TODO: ignores the value of privileged_classes for 'sex'
    status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
    # status_map = {'91': 'male', '93': 'male', '94': 'male',
    #               '92': 'female', '95': 'female'}
    df['sex'] = df['personal_status'].replace(status_map)

    return df

class GermanDataset(StandardDataset):
    """German credit Dataset.

    See :file:`aif360/data/raw/german/README.md`.
    """

    def __init__(self, label_name='credit',
                 favorable_classes=[1],
                 protected_attribute_names=['sex', 'age'],
                 # privileged_classes=[['male'], lambda x: x > 25],
                 privileged_classes=[[1], lambda x: x > 3],
                 instance_weights_name=None,
                 # categorical_features=['status', 'credit_history', 'purpose',
                 #     'savings', 'employment', 'other_debtors', 'property',
                 #     'installment_plans', 'housing', 'skill_level', 'telephone',
                 #     'foreign_worker'],
                 categorical_features=['checking_status', 'credit_history', 'purpose',
                                       'savings_status', 'employment', 'other_parties', 'property_magnitude',
                                       'other_payment_plans', 'housing', 'job', 'own_telephone',
                                       'foreign_worker'],
                 features_to_keep=[],
                 # features_to_drop=['personal_status'],
                 features_to_drop=[],
                 na_values=[],
                 # custom_preprocessing=default_preprocessing,
                 # metadata=default_mappings
                 custom_preprocessing=None,
                 metadata=None):
        """See :obj:`StandardDataset` for a description of the arguments.

        By default, this code converts the 'age' attribute to a binary value
        where privileged is `age > 25` and unprivileged is `age <= 25` as
        proposed by Kamiran and Calders [1]_.

        References:
            .. [1] F. Kamiran and T. Calders, "Classifying without
               discriminating," 2nd International Conference on Computer,
               Control and Communication, 2009.

        Examples:
            In some cases, it may be useful to keep track of a mapping from
            `float -> str` for protected attributes and/or labels. If our use
            case differs from the default, we can modify the mapping stored in
            `metadata`:

            >>> label_map = {1.0: 'Good Credit', 0.0: 'Bad Credit'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> gd = GermanDataset(protected_attribute_names=['sex'],
            ... privileged_classes=[['male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})

            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'data', 'raw', 'german', 'credit.csv')

        # filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                         '..', 'data', 'raw', 'german', 'german.data')

        # filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                         '..', 'data', 'raw', 'german', 'german_numeric.data')
        # as given by german.doc
        # column_names = ['status', 'month', 'credit_history',
        #     'purpose', 'credit_amount', 'savings', 'employment',
        #     'investment_as_income_percentage', 'personal_status',
        #     'other_debtors', 'residence_since', 'property', 'age',
        #     'installment_plans', 'housing', 'number_of_credits',
        #     'skill_level', 'people_liable_for', 'telephone',
        #     'foreign_worker', 'credit']

        column_names = ['checking_status', 'duration', 'credit_history',
                        'purpose', 'credit_amount', 'savings_status', 'employment',
                        'installment_commitment', 'sex',
                        'other_parties', 'residence', 'property_magnitude', 'age',
                        'other_payment_plans', 'housing', 'existing_credits',
                        'job', 'num_dependents', 'own_telephone',
                        'foreign_worker', 'credit']


        try:
            # df = pd.read_csv(filepath, sep=' ', header=None, names=column_names,
            #                  na_values=na_values)
            df = pd.read_csv(filepath, sep=';', na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following files:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
            print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc")
            print("\nand place them, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
                os.path.abspath(__file__), '..', '..', 'data', 'raw', 'german'))))
            import sys
            sys.exit(1)

        super(GermanDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)

        self.categorical_features = categorical_features
        self.column_names = [feature for feature in column_names if feature not in features_to_drop]
        try:
            self.column_names.remove(label_name)
        except:
            print("-->cannot remove label_name")
        ranges = self.get_non_categorical_range()
        self.ranges = ranges
        protected_attribute_indexs = [self.feature_names.index(attribute) for attribute in
                                      self.protected_attribute_names]
        self.protected_attribute_indexs = protected_attribute_indexs

    def get_non_categorical_range(self):
        categorical_features = [feature for feature in self.categorical_features if
                                feature not in self.protected_attribute_names]
        non_categorical_features = [feature for feature in self.feature_names if feature not in categorical_features]
        ranges = []
        for feature in non_categorical_features:
            index = self.feature_names.index(feature)
            feature_range = [min(self.features[:, index]), max(self.features[:, index])]
            ranges.append(feature_range)
        return ranges

    def clip(self, input):
        """
        Clip the generating instance with each feature to make sure it is valid
        """
        categorical_features = [feature for feature in self.categorical_features if
                                feature not in self.protected_attribute_names]
        non_categorical_features = [feature for feature in self.feature_names if feature not in categorical_features]
        for i in range(0, len(non_categorical_features)):
            feature = non_categorical_features[i]
            index = self.feature_names.index(feature)
            input[index] = max(input[index], self.ranges[i][0])
            input[index] = min(input[index], self.ranges[i][1])
        return list(input)

    def random_input(self):
        categorical_features = [feature for feature in self.categorical_features if feature not in self.protected_attribute_names]
        non_categorical_features = [feature for feature in self.feature_names if feature not in categorical_features]
        new_input = []
        for feature in non_categorical_features:
            index = self.feature_names.index(feature)
            # print("-->index", index)
            # print("-->dataset of selected feature")
            # print(self.features[:, index])
            feature_range = [min(self.features[:, index]), max(self.features[:, index])]
            new_input.append(random.randint(feature_range[0], feature_range[1]))

        new_input.extend(0 for _ in range(len(self.feature_names)-len(new_input)))

        for feature in categorical_features:
            indexes = [self.feature_names.index(f) for f in self.feature_names if feature in f]
            index = random.choice(indexes)
            new_input[index] = 1
        return new_input

    def distort_input(self, *args):
        priviledged_group = args[0]
        if_priviledge = args[1]  # True
        # try:
        #     priviledged_group = args[0]
        #     if_priviledge = args[1]  # True
        # except:
        #     priviledged_group = self.privileged_protected_attributes
        #     if_priviledge = True

        categorical_features = [feature for feature in self.categorical_features if
                                feature not in self.protected_attribute_names]
        # print("-->categorical_features", categorical_features)
        noncategorical_features = [feature for feature in self.feature_names if feature not in categorical_features]
        non_categorical_features = [feature for feature in noncategorical_features if feature not in self.protected_attribute_names]
        non_protected_features = [feature for feature in self.feature_names if feature not in self.protected_attribute_names]
        # print("-->non_categorical_features", non_categorical_features)

        not_priviledge = operator.not_(if_priviledge)
        # print("-->not_priviledge", not_priviledge)
        cond_vec = [not_priviledge]  # false
        protected_attribute_indexs = [self.feature_names.index(attribute) for attribute in
                                      self.protected_attribute_names]
        # print("-->protected_attribute_indexs", protected_attribute_indexs)
        # while new_input not satisfy priviledged_group or unpriviledged group, continue random.choice
        # condition if necessary
        # condition = [{'sex': 1, 'age': 1}, {'sex': 0}]

        while(cond_vec[0] == not_priviledge):
            new_input = random.choice(self.features)
            # print("-->new_input", new_input)
            protected_attributes = np.array([[new_input[index] for index in protected_attribute_indexs]])
            # print("-->protected_attributes", protected_attributes)
            # print(type(protected_attributes))
            # print("-->priviledged_group", priviledged_group)
            cond_vec = compute_boolean_conditioning_vector(protected_attributes, self.protected_attribute_names,
                                                           priviledged_group)
            # print("-->cond_vec", cond_vec)

        pert_category = random.choice([1, 1])
        if pert_category == 1:  # for non-categorical features
            pert_feature = random.choice(non_categorical_features)
            pert_feature_index = self.feature_names.index(pert_feature)
            pert_range = [1, 5]
            pert_num = random.randint(pert_range[0], pert_range[1])
            pert_dir = random.choice([-1, 1])
            new_input[pert_feature_index] = new_input[pert_feature_index] + pert_dir*pert_num
        else:   # for categorical features
            feature = random.choice(categorical_features)
            indexes = [self.feature_names.index(f) for f in self.feature_names if feature in f]
            for i in indexes:
                new_input[i] = 0
            index = random.choice(indexes)
            new_input[index] = 1
        # print("-->clipped new_input", self.clip(new_input))
        return self.clip(new_input)

    def generate_inputs(self, max_num, if_random, privileged_groups, if_priviledge):
        # print("-->generate_inputs function:", self.feature_names)
        # print(self.protected_attributes)
        # print("-->check first 2 3 features")
        # print(self.features[1])
        # print(self.features[2])
        # print(self.protected_attributes[1])
        # print(self.protected_attributes[2])
        new_inputs = []
        while len(new_inputs) < max_num:
            if if_random == True:
                new_inputs.append(self.random_input())
            else:
                # print("-->distort_input")
                new_inputs.append(self.distort_input(privileged_groups, if_priviledge))
            # print("-->type", type(new_inputs))
            # new_inputs = list(set(new_inputs))

        return new_inputs



