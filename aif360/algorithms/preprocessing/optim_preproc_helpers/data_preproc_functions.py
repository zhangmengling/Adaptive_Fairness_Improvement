from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset
from aif360.datasets.compas_dataset1 import CompasDataset_1
import pandas as pd
import numpy as np

def load_preproc_data_adult(protected_attributes=None, sub_samp=False, balance=False):
    # def custom_preprocessing(df):
    #     """The custom pre-processing function is adapted from
    #         https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb
    #         If sub_samp != False, then return smaller version of dataset truncated to tiny_test data points.
    #     """
    #
    #     # Group age by decade
    #     df['Age (decade)'] = df['age'].apply(lambda x: x//10*10)
    #     # df['Age (decade)'] = df['age'].apply(lambda x: np.floor(x/10.0)*10.0)
    #
    #     def group_edu(x):
    #         if x <= 5:
    #             return '<6'
    #         elif x >= 13:
    #             return '>12'
    #         else:
    #             return x
    #
    #     def age_cut(x):
    #         if x >= 70:
    #             return '>=70'
    #         else:
    #             return x
    #
    #     def group_race(x):
    #         if x == "White":
    #             return 1.0
    #         else:
    #             return 0.0
    #
    #     # Cluster education and age attributes.
    #     # Limit education range
    #     df['Education Years'] = df['education-num'].apply(lambda x: group_edu(x))
    #     df['Education Years'] = df['Education Years'].astype('category')
    #
    #     # Limit age range
    #     df['Age (decade)'] = df['Age (decade)'].apply(lambda x: age_cut(x))
    #
    #     # Rename income variable
    #     df['Income Binary'] = df['income-per-year']
    #     df['Income Binary'] = df['Income Binary'].replace(to_replace='>50K.', value='>50K', regex=True)
    #     df['Income Binary'] = df['Income Binary'].replace(to_replace='<=50K.', value='<=50K', regex=True)
    #
    #     # Recode sex and race
    #     df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
    #     df['race'] = df['race'].apply(lambda x: group_race(x))
    #
    #     if sub_samp and not balance:
    #         df = df.sample(sub_samp)
    #     if sub_samp and balance:
    #         df_0 = df[df['Income Binary'] == '<=50K']
    #         df_1 = df[df['Income Binary'] == '>50K']
    #         df_0 = df_0.sample(int(sub_samp/2))
    #         df_1 = df_1.sample(int(sub_samp/2))
    #         df = pd.concat([df_0, df_1])
    #
    #     # print("-->df:", df)
    #     workclass_dic = {'Private': 1, 'Self-emp-inc': 2, 'Never-worked': 3, 'Self-emp-not-inc': 4, 'Local-gov': 5,
    #                      'Federal-gov': 6, 'State-gov': 7, 'Without-pay': 8}
    #     for key in workclass_dic:
    #         df.replace(key, workclass_dic[key], inplace=True)
    #     education_dic = {'HS-grad': 1, 'Masters': 2, 'Some-college': 3, '1st-4th': 4, 'Prof-school': 5, 'Bachelors': 6,
    #                      'Assoc-acdm': 7, 'Doctorate': 8, '9th': 9, 'Preschool': 10, '10th': 11, '11th': 12, '12th': 13,
    #                      '5th-6th': 14, 'Assoc-voc': 15, '7th-8th': 16}
    #     for key in education_dic:
    #         df.replace(key, education_dic[key], inplace=True)
    #     marital_dic = {'Divorced': 1, 'Never-married': 2, 'Married-spouse-absent': 3, 'Widowed': 4, 'Married-civ-spouse': 5,
    #                    'Separated': 6, 'Married-AF-spouse': 7}
    #     for key in marital_dic:
    #         df.replace(key, marital_dic[key], inplace=True)
    #     occupation_dic = {'Craft-repair': 1, 'Sales': 2, 'Adm-clerical': 3, 'Other-service': 4, 'Protective-serv': 5,
    #                       'Armed-Forces': 6, 'Handlers-cleaners': 7, 'Tech-support': 8, 'Priv-house-serv': 9,
    #                       'Exec-managerial': 10, 'Machine-op-inspct': 11, 'Farming-fishing': 12, 'Transport-moving': 13, 'Prof-specialty': 14}
    #     for key in occupation_dic:
    #         df.replace(key, occupation_dic[key], inplace=True)
    #     relationship_dic = {'Own-child': 1, 'Not-in-family': 2, 'Unmarried': 3, 'Other-relative': 4, 'Wife': 5, 'Husband': 6}
    #     for key in relationship_dic:
    #         df.replace(key, relationship_dic[key], inplace=True)
    #     # print("-->df:", df)
    #     return df

    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb
            If sub_samp != False, then return smaller version of dataset truncated to tiny_test data points.
        """
        return df

    XD_features = ['age', 'education', 'sex', 'race']
    # XD_features = ['Age (decade)', 'workclass', 'Education Years', 'occupation', 'race', 'sex', 'Income Binary']
    # ['Age (decade)', 'Education Years', 'Income', 'Gender', 'Income Binary']
    # XC_features =  ["age", "workclass", "fnlwgt", "education", "marital_status", "occupation", "relationship", "race",
    # "sex", "captital_gain", "capital_loss", "hours_per_week", "native_country", 'income-per-year']

    D_features = ['sex', 'race'] if protected_attributes is None else protected_attributes
    # Y_features = ['Income Binary']
    Y_features = ['income-per-year']
    X_features = list(set(XD_features)-set(D_features))
    # categorical_features = ['Age (decade)', 'Education Years']
    categorical_features = ['age', 'education']


    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}
    # unprivileged_classes
    all_unprivileged_classes = {"sex": [0.0],
                              "race": [0.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                    "race": {1.0: 'White', 0.0: 'Non-white'}}

    return AdultDataset(
        label_name=Y_features[0],
        # favorable_classes=['>50K', '>50K.'],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        # unprivileged_classes=[all_unprivileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=['?'],
        custom_preprocessing=custom_preprocessing)
        # metadata={'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
        #           'protected_attribute_maps': [all_protected_attribute_maps[x]
        #                         for x in D_features]},
        # custom_preprocessing=custom_preprocessing)

def load_preproc_data_compas(protected_attributes=None):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
        """

        print("-->df columns", df.columns.tolist())

        df.rename(columns={'start': 'c_jail_in'}, inplace=True)
        df.rename(columns={'end': 'c_jail_out'}, inplace=True)

        print("-->df columns", df.columns.tolist())

        # columns[
        #     'sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count',
        #     'priors_count', 'days_b_screening_arrest', 'c_days_from_compas', 'c_charge_degree', 'is_recid', 'r_charge_degree',
        #     'r_days_from_arrest', 'is_violent_recid', 'vr_charge_degree', 'decile_score.1', 'score_text', 'v_decile_score',
        #     'v_score_text', 'priors_count.1', 'start', 'end', 'event', 'two_year_recid']

        df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text',
                 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score',
                 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

        # Indices of data samples to keep
        ix = df['days_b_screening_arrest'] <= 30
        ix = (df['days_b_screening_arrest'] >= -30) & ix
        ix = (df['is_recid'] != -1) & ix
        ix = (df['c_charge_degree'] != "O") & ix
        ix = (df['score_text'] != 'N/A') & ix
        df = df.loc[ix,:]
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-
                                pd.to_datetime(df['c_jail_in'])).apply(
                                                        lambda x: x.days)

        # Restrict races to African-American and Caucasian
        dfcut = df.loc[~df['race'].isin(['Native American','Hispanic','Asian','Other']),:]

        # Restrict the features to use
        dfcutQ = dfcut[['sex','race','age_cat','c_charge_degree','score_text','priors_count','is_recid',
                'two_year_recid','length_of_stay']].copy()

        # Quantize priors count between 0, 1-3, and >3
        def quantizePrior(x):
            if x <=0:
                return '0'
            elif 1<=x<=3:
                return '1 to 3'
            else:
                return 'More than 3'

        # Quantize length of stay
        def quantizeLOS(x):
            if x<= 7:
                return '<week'
            if 8<x<=93:
                return '<3months'
            else:
                return '>3 months'

        # Quantize length of stay
        def adjustAge(x):
            if x == '25 - 45':
                return '25 to 45'
            else:
                return x

        # Quantize score_text to MediumHigh
        def quantizeScore(x):
            if (x == 'High')| (x == 'Medium'):
                return 'MediumHigh'
            else:
                return x

        def group_race(x):
            if x == "Caucasian":
                return 1.0
            else:
                return 0.0

        dfcutQ['priors_count'] = dfcutQ['priors_count'].apply(lambda x: quantizePrior(x))
        dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(lambda x: quantizeLOS(x))
        dfcutQ['score_text'] = dfcutQ['score_text'].apply(lambda x: quantizeScore(x))
        dfcutQ['age_cat'] = dfcutQ['age_cat'].apply(lambda x: adjustAge(x))

        # Recode sex and race
        dfcutQ['sex'] = dfcutQ['sex'].replace({'Female': 1.0, 'Male': 0.0})
        dfcutQ['race'] = dfcutQ['race'].apply(lambda x: group_race(x))

        features = ['two_year_recid',
                    'sex', 'race',
                    'age_cat', 'priors_count', 'c_charge_degree']

        # Pass vallue to df
        df = dfcutQ[features]
        print("-->age_cat:", set(list(df['age_cat'])))

        return df

    XD_features = ['age_cat', 'c_charge_degree', 'priors_count', 'sex', 'race']
    D_features = ['sex', 'race']  if protected_attributes is None else protected_attributes
    Y_features = ['two_year_recid']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['age_cat', 'priors_count', 'c_charge_degree']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {0.0: 'Male', 1.0: 'Female'},
                                    "race": {1.0: 'Caucasian', 0.0: 'Not Caucasian'}}


    # changed CompasDataset --> CompasDataset_1
    return CompasDataset_1(
        label_name=Y_features[0],
        favorable_classes=[0],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        # features_to_keep=X_features+Y_features+D_features,
        features_to_keep=D_features + X_features + Y_features,
        na_values=[],
        metadata={'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)

def load_preproc_data_german(protected_attributes=None):
    """
    Load and pre-process german credit dataset.
    Args:
        protected_attributes(list or None): If None use all possible protected
            attributes, else subset the protected attributes to the list.

    Returns:
        GermanDataset: An instance of GermanDataset with required pre-processing.

    """
    def custom_preprocessing(df):
        """ Custom pre-processing for German Credit Data
        """
        def group_credit_hist(x):
            if x in ['A30', 'A31', 'A32']:
                return 'None/Paid'
            elif x == 'A33':
                return 'Delay'
            elif x == 'A34':
                return 'Other'
            else:
                return 'NA'

        def group_employ(x):
            if x == 'A71':
                return 'Unemployed'
            elif x in ['A72', 'A73']:
                return '1-4 years'
            elif x in ['A74', 'A75']:
                return '4+ years'
            else:
                return 'NA'

        def group_savings(x):
            if x in ['A61', 'A62']:
                return '<500'
            elif x in ['A63', 'A64']:
                return '500+'
            elif x == 'A65':
                return 'Unknown/None'
            else:
                return 'NA'

        def group_status(x):
            if x in ['A11', 'A12']:
                return '<200'
            elif x in ['A13']:
                return '200+'
            elif x == 'A14':
                return 'None'
            else:
                return 'NA'

        def group_amount(x):
            if x < 1000:
                return 0
            elif 1000 <= x < 2000:
                return 1
            elif 2000 <= x < 3000:
                return 2
            elif 3000 <= x < 4000:
                return 3
            elif 4000 <= x < 5000:
                return 4
            elif 5000 <= x < 6000:
                return 5
            elif x >=6000:
                return 6


        status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
                    'A92': 0.0, 'A95': 0.0}
        df['sex'] = df['personal_status'].replace(status_map)


        # group credit history, savings, and employment
        df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
        df['savings'] = df['savings'].apply(lambda x: group_savings(x))
        df['employment'] = df['employment'].apply(lambda x: group_employ(x))
        df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
        df['status'] = df['status'].apply(lambda x: group_status(x))

        df['credit_amount'] = df['credit_amount'].apply(lambda x: group_amount(x))
        df['credit_history'] = df['credit_history'].replace({'Other': 1.0, 'Delay': 2.0, 'None/Paid': 3.0})
        df['property'] = df['property'].replace({'A124':1.0, 'A123':2.0, 'A122':3.0, 'A121':4.0})
        df['skill_level'] = df['skill_level'].replace({'A174':0.0, 'A173':1.0, 'A171':2.0, 'A172':3.0})
        df['foreign_worker'] = df['foreign_worker'].replace({'A201':0.0, 'A202':1.0})
        #
        # print("-->df", df)
        # print(list(df[0:1]))
        # print("-->credit_history", set(list(df.iloc[:,2])))
        # print("-->credit_amount", set(list(df.iloc[:, 4])))
        # print("-->property", set(list(df.iloc[:, 11])))
        # print("-->number_of_credits", set(list(df.iloc[:, 15])))
        # print("-->skill_level", set(list(df.iloc[:, 16])))
        # print("-->foreign_worker", set(list(df.iloc[:, 19])))
        return df

    # Feature partitions
    XD_features = ['credit_history', 'savings', 'employment', 'sex', 'age']
    # XD_features = ['credit_history', 'savings', 'employment', 'skill_level', 'foreign_worker', 'sex', 'age']
    # 'status', 'month', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment',
    # 'investment_as_income_percentage', 'personal_status', 'other_debtors', 'residence_since', 'property', 'age',
    # 'installment_plans', 'housing', 'number_of_credits', 'skill_level', 'people_liable_for', 'telephone', 'foreign_worker', 'credit', 'sex'
    D_features = ['sex', 'age'] if protected_attributes is None else protected_attributes
    Y_features = ['credit']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['credit_history', 'savings', 'employment']
    # categorical_features = ['credit_history', 'savings', 'employment', 'skill_level', 'foreign_worker']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "age": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                    "age": {1.0: 'Old', 0.0: 'Young'}}

    return GermanDataset(
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={ 'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
                   'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)


def load_preproc_data_bank(protected_attributes=None, sub_samp=False, balance=False):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb
            If sub_samp != False, then return smaller version of dataset truncated to tiny_test data points.
        """

        # Group age by decade
        df['age'] = df['age'].apply(lambda x: x >= 25)

        # # Limit age range
        # df['Age (decade)'] = df['Age (decade)'].apply(lambda x: age_cut(x))

        return df

    # XD_features = ['age', 'marital', 'contact', 'duration', 'previous', 'poutcome']
    XD_features = ['age', 'duration', 'previous', 'poutcome']
    # XD_features = ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'duration',
    #                'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
    #                'cons.conf.idx', 'euribor3m', 'nr.employed']

    D_features = ['age'] if protected_attributes is None else protected_attributes
    Y_features = ['y']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['age', 'contact', 'poutcome']

    # privileged classes
    all_privileged_classes = {"age": [1.0]}

    print("-->feature to keep", X_features+Y_features+D_features)

    # protected attribute maps
    # all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
    #                                 "race": {1.0: 'White', 0.0: 'Non-white'}}

    return BankDataset(
        label_name=Y_features[0],
        favorable_classes=['yes'],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        # unprivileged_classes=[all_unprivileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=['?'],
        metadata=None,
        custom_preprocessing=custom_preprocessing)
        # metadata={'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
        #           'protected_attribute_maps': [all_protected_attribute_maps[x]
        #                         for x in D_features]})
        # , custom_preprocessing=custom_preprocessing)
