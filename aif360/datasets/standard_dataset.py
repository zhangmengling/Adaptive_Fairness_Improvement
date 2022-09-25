from logging import warning

import numpy as np
import pandas as pd

from aif360.datasets import BinaryLabelDataset


class StandardDataset(BinaryLabelDataset):
    """Base class for every :obj:`BinaryLabelDataset` provided out of the box by
    aif360.

    It is not strictly necessary to inherit this class when adding custom
    datasets but it may be useful.

    This class is very loosely based on code from
    https://github.com/algofairness/fairness-comparison.
    """

    def __init__(self, df, label_name, favorable_classes,
                 protected_attribute_names, privileged_classes,
                 instance_weights_name='', scores_name='',
                 categorical_features=[], features_to_keep=[],
                 features_to_drop=[], na_values=[], custom_preprocessing=None,
                 metadata=None):
        """
        Subclasses of StandardDataset should perform the following before
        calling `super().__init__`:

            1. Load the dataframe from a raw file.

        Then, this class will go through a standard preprocessing routine which:

            2. (optional) Performs some dataset-specific preprocessing (e.g.
               renaming columns/values, handling missing data).

            3. Drops unrequested columns (see `features_to_keep` and
               `features_to_drop` for details).

            4. Drops rows with NA values.

            5. Creates a one-hot encoding of the categorical variables.

            6. Maps protected attributes to binary privileged/unprivileged
               values (1/0).

            7. Maps labels to binary favorable/unfavorable labels (1/0).

        Args:
            df (pandas.DataFrame): DataFrame on which to perform standard
                processing.
            label_name: Name of the label column in `df`.
            favorable_classes (list or function): Label values which are
                considered favorable or a boolean function which returns `True`
                if favorable. All others are unfavorable. Label values are
                mapped to 1 (favorable) and 0 (unfavorable) if they are not
                already binary and numerical.
            protected_attribute_names (list): List of names corresponding to
                protected attribute columns in `df`.
            privileged_classes (list(list or function)): Each element is
                a list of values which are considered privileged or a boolean
                function which return `True` if privileged for the corresponding
                column in `protected_attribute_names`. All others are
                unprivileged. Values are mapped to 1 (privileged) and 0
                (unprivileged) if they are not already numerical.
            instance_weights_name (optional): Name of the instance weights
                column in `df`.
            categorical_features (optional, list): List of column names in the
                DataFrame which are to be expanded into one-hot vectors.
            features_to_keep (optional, list): Column names to keep. All others
                are dropped except those present in `protected_attribute_names`,
                `categorical_features`, `label_name` or `instance_weights_name`.
                Defaults to all columns if not provided.
            features_to_drop (optional, list): Column names to drop. *Note: this
                overrides* `features_to_keep`.
            na_values (optional): Additional strings to recognize as NA. See
                :func:`pandas.read_csv` for details.
            custom_preprocessing (function): A function object which
                acts on and returns a DataFrame (f: DataFrame -> DataFrame). If
                `None`, no extra preprocessing is applied.
            metadata (optional): Additional metadata to append.
        """
        # 2. Perform dataset-specific preprocessing
        if custom_preprocessing:
            df = custom_preprocessing(df)

        # 3. Drop unrequested columns
        print("-->features_to_keep", features_to_keep)
        print("-->df.columns.tolist()", df.columns.tolist())
        features_to_keep = features_to_keep or df.columns.tolist()
        keep = (set(features_to_keep) | set(protected_attribute_names)
              | set(categorical_features) | set([label_name]))
        if instance_weights_name:
            keep |= set([instance_weights_name])
        print("-->keep", keep)
        print("-->features_to_drop", features_to_drop)
        df = df[sorted(keep - set(features_to_drop), key=df.columns.get_loc)]
        categorical_features = sorted(set(categorical_features) - set(features_to_drop), key=df.columns.get_loc)
        print("-->df", df)


        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        if count > 0:
            warning("Missing Data: {} rows removed from {}.".format(count,
                    type(self).__name__))
        df = dropped

        # 5. Create a one-hot encoding of the categorical variables.
        print("-->categorical_features", categorical_features)
        print("-->df", df)
        for colum_name in df.columns.tolist():
            print("-->colum_name", colum_name)
            print(type(df[str(colum_name)][0]))
            if type(df[colum_name][0]) == str and colum_name != df.columns.tolist()[-1]:
                if colum_name in protected_attribute_names:
                    continue
                print("-->colum_name", colum_name)
                variables = list(set(list(df[colum_name])))
                num_vaiables = list(range(0, len(variables)))
                variable_map = dict(zip(variables, num_vaiables))
                print("-->variable_map", variable_map)
                # df[colum_name].map(variable_map)
                df[colum_name] = df[colum_name].apply(lambda x: variable_map[x])

        # df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')
        print("-->df column names in standard_dataset:", df.columns.tolist())

        # 6.0. Make unprivileged feature binary
        for attr, vals in zip(protected_attribute_names, privileged_classes):
            privileged_values = [1.]
            unprivileged_values = [0.]
            if attr == "age":
                continue
            eql = lambda x: x == vals[0]
            print("-->after making binary")
            print(df[attr])
            df[attr] = df[attr].apply(eql)
            print(df[attr])

        print("-->df", df)
        # 6. Map protected attributes to privileged/unprivileged
        privileged_protected_attributes = []
        unprivileged_protected_attributes = []
        for attr, vals in zip(protected_attribute_names, privileged_classes):
            privileged_values = [1.]
            unprivileged_values = [0.]
            if callable(vals):
                df[attr] = df[attr].apply(vals)
            elif np.issubdtype(df[attr].dtype, np.number):
                # this attribute is numeric; no remapping needed
                privileged_values = vals
                unprivileged_values = list(set(df[attr]).difference(vals))
            else:
                # find all instances which match any of the attribute values
                try:
                    priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
                except:
                    vals = [1.0]
                    priv = np.logical_or.reduce(np.equal.outer(vals, df[attr].to_numpy()))
                df.loc[priv, attr] = privileged_values[0]
                df.loc[~priv, attr] = unprivileged_values[0]

                # 6.1. Fix unprivileged feature to 0.0
                if len(unprivileged_values) != 1:
                    df[attr] = 0.0

            privileged_protected_attributes.append(
                np.array(privileged_values, dtype=np.float64))
            unprivileged_protected_attributes.append(
                np.array(unprivileged_values, dtype=np.float64))

        print("-->privileged_protected_attributes", privileged_protected_attributes)
        print("-->unprivileged_protected_attributes", unprivileged_protected_attributes)




        # 7. Make labels binary
        favorable_label = 1.
        unfavorable_label = 0.
        if callable(favorable_classes):
            df[label_name] = df[label_name].apply(favorable_classes)
        elif np.issubdtype(df[label_name], np.number) and len(set(df[label_name])) == 2:
            # labels are already binary; don't change them
            favorable_label = favorable_classes[0]
            unfavorable_label = set(df[label_name]).difference(favorable_classes).pop()
        else:
            # find all instances which match any of the favorable classes
            pos = np.logical_or.reduce(np.equal.outer(favorable_classes, 
                                                      df[label_name].to_numpy()))
            df.loc[pos, label_name] = favorable_label
            df.loc[~pos, label_name] = unfavorable_label

        super(StandardDataset, self).__init__(df=df, label_names=[label_name],
            protected_attribute_names=protected_attribute_names,
            privileged_protected_attributes=privileged_protected_attributes,
            unprivileged_protected_attributes=unprivileged_protected_attributes,
            instance_weights_name=instance_weights_name,
            scores_names=[scores_name] if scores_name else [],
            favorable_label=favorable_label,
            unfavorable_label=unfavorable_label, metadata=metadata)
