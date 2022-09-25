# Contains the helper functions for the optim_preproc class
import numpy as np
import pandas as pd


def get_distortion_adult(vold, vnew):
    """Distortion function for the adult dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """

    # Define local functions to adjust education and age
    def adjustEdu(v):
        if v == '>12':
            return 13
        elif v == '<6':
            return 5
        else:
            return int(v)

    def adjustAge(a):
        if a == '>=70':
            return 70.0
        else:
            return float(a)

    def adjustInc(a):
        if a == "<=50K":
            return 0
        elif a == ">50K":
            return 1
        else:
            return int(a)

    # value that will be returned for events that should not occur
    bad_val = 3.0

    # Adjust education years
    eOld = adjustEdu(vold['Education Years'])
    eNew = adjustEdu(vnew['Education Years'])

    # Education cannot be lowered or increased in more than 1 year
    if (eNew < eOld) | (eNew > eOld+1):
        return bad_val

    # adjust age
    aOld = adjustAge(vold['Age (decade)'])
    aNew = adjustAge(vnew['Age (decade)'])

    # Age cannot be increased or decreased in more than a decade
    if np.abs(aOld-aNew) > 10.0:
        return bad_val

    # Penalty of 2 if age is decreased or increased
    if np.abs(aOld-aNew) > 0:
        return 2.0

    # Adjust income
    incOld = adjustInc(vold['Income Binary'])
    incNew = adjustInc(vnew['Income Binary'])

    # final penalty according to income
    if incOld > incNew:
        return 1.0
    else:
        return 0.0


def get_distortion_german(vold, vnew):
    """Distortion function for the german dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """

    # Distortion cost
    distort = {}
    distort['credit_history'] = pd.DataFrame(
                                {'None/Paid': [0., 1., 2.],
                                'Delay':      [1., 0., 1.],
                                'Other':      [2., 1., 0.]},
                                index=['None/Paid', 'Delay', 'Other'])
    distort['employment'] = pd.DataFrame(
                            {'Unemployed':    [0., 1., 2.],
                            '1-4 years':      [1., 0., 1.],
                            '4+ years':       [2., 1., 0.]},
                            index=['Unemployed', '1-4 years', '4+ years'])
    distort['savings'] = pd.DataFrame(
                            {'Unknown/None':  [0., 1., 2.],
                            '<500':           [1., 0., 1.],
                            '500+':           [2., 1., 0.]},
                            index=['Unknown/None', '<500', '500+'])
    distort['status'] = pd.DataFrame(
                            {'None':          [0., 1., 2.],
                            '<200':           [1., 0., 1.],
                            '200+':           [2., 1., 0.]},
                            index=['None', '<200', '200+'])
    distort['credit'] = pd.DataFrame(
                        {'Bad Credit':    [0., 1.],
                         'Good Credit':    [2., 0.]},
                         index=['Bad Credit', 'Good Credit'])

    # distort['credit'] = pd.DataFrame(
    #     {'Bad Credit' : [0., 1.],
    #      'Good Credit': [2., 0.],
    #      0.0: [0., 1.],
    #      1.0: [2., 0.]},
    #     index=['Bad Credit', 'Good Credit', 0.0, 1.0])

    distort['sex'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])
    distort['age'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])

    distort['skill_level'] = pd.DataFrame(
                        {0.0:    [0., 1., 2., 3.],
                         1.0:    [1., 0., 2., 3.],
                         2.0:    [2., 1., 0., 3.],
                         3.0:    [3., 2., 1., 0.]},
                         index=[0.0, 1.0, 2.0, 3.0])
    distort['foreign_worker'] = pd.DataFrame(
        {0.0: [0., 2.],
         1.0: [2., 0.]},
        index=[0.0, 1.0])

    total_cost = 0.0
    for k in vold:
        if k in vnew:
            # print("-->k", k)
            # print(distort[k])
            # print(vnew)
            # print(vold)
            # print(vnew[k])
            # print(vold[k])
            # print(type(vnew[k]))
            if k == "credit":
                if vnew[k] == 0.0:
                    vnew[k] = 'Bad Credit'
                elif vnew[k] == 1.0:
                    vnew[k] = 'Good Credit'
                else:
                    vnew[k] = vnew[k]
                if vold[k] == 0.0:
                    vold[k] = 'Bad Credit'
                elif vold[k] == 1.0:
                    vold[k] = 'Good Credit'
                else:
                    vold[k] = vold[k]
            elif k == "credit_history":
                # print(vnew)
                # print(vold)
                # print(vnew[k])
                # print(vold[k])
                # print(type(vnew[k]))
                if vnew[k] == '1.0' or '0.0' or '2.0' or '3.0':
                    if vnew[k] == "1.0":
                        vnew[k] = 'Other'
                    elif vnew[k] == "2.0":
                        vnew[k] = 'Delay'
                    elif vnew[k] == "3.0":
                        vnew[k] = "None/Paid"
                    else:
                        vnew[k] = vnew[k]
                if vold[k] == '1.0' or '0.0' or '2.0' or '3.0':
                    if vold[k] == "1.0":
                        vold[k] = 'Other'
                    elif vold[k] == "2.0":
                        vold[k] = 'Delay'
                    elif vold[k] == "3.0":
                        vold[k] = "None/Paid"
                    else:
                        vold[k] = vold[k]
            elif k == "skill_level":
                # print(vnew)
                # print(vold)
                # print(vnew[k])
                # print(vold[k])
                # print(type(vnew[k]))
                if vnew[k] == '1.0' or '0.0' or '2.0' or '3.0':
                    vnew[k] = float(vnew[k])
                if vold[k] == '1.0' or '0.0' or '2.0' or '3.0':
                    vold[k] = float(vold[k])
            elif k == "foreign_worker":
                # print(vnew)
                # print(vold)
                # print(vnew[k])
                # print(vold[k])
                # print(type(vnew[k]))
                if vnew[k] == '1.0' or '0.0' or '2.0' or '3.0':
                    vnew[k] = float(vnew[k])
                if vold[k] == '1.0' or '0.0' or '2.0' or '3.0':
                    vold[k] = float(vold[k])
                # df['credit_history'] = df['credit_history'].replace({'Other': 1.0, 'Delay': 2.0, 'None/Paid': 3.0})
            total_cost += distort[k].loc[vnew[k], vold[k]]

    return total_cost

def get_distortion_compas(vold, vnew):
    """Distortion function for the compas dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """
    # Distortion cost
    distort = {}
    distort['two_year_recid'] = pd.DataFrame(
                                {'No recid.':     [0., 2.],
                                'Did recid.':     [2., 0.]},
                                index=['No recid.', 'Did recid.'])

    # distort['two_year_recid'] = pd.DataFrame(
    #     {0.0: [0., 2.],
    #      1.0: [2., 0.]},
    #     index=[0.0, 1.0])

    distort['age_cat'] = pd.DataFrame(
                            {'Less than 25':    [0., 1., 2.],
                            '25 to 45':         [1., 0., 1.],
                            'Greater than 45':  [2., 1., 0.]},
                            index=['Less than 25', '25 to 45', 'Greater than 45'])

    # distort['age_cat'] = pd.DataFrame(
    #                         {'Less than 25':    [0., 1., 2.],
    #                         '25 - 45':         [1., 0., 1.],
    #                         'Greater than 45':  [2., 1., 0.]},
    #                         index=['Less than 25', '25 - 45', 'Greater than 45'])

    distort['c_charge_degree'] = pd.DataFrame(
                            {'M':   [0., 2.],
                            'F':    [1., 0.]},
                            index=['M', 'F'])
    distort['priors_count'] = pd.DataFrame(
                            {'0':           [0., 1., 2.],
                            '1 to 3':       [1., 0., 1.],
                            'More than 3':  [2., 1., 0.]},
                            index=['0', '1 to 3', 'More than 3'])
    distort['sex'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])
    distort['race'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])

    total_cost = 0.0
    for k in vold:
        if k in vnew:
            # print("-->k", k)
            # print(distort[k])
            # print(vnew)
            # print(vold)
            # print(vnew[k])
            # print(vold[k])
            # if k == 'priors_count':
            #     try:
            #         if 1 <= int(vnew[k]) <= 3:
            #             vnew[k] = '1 to 3'
            #         elif int(vnew[k]) > 3:
            #             vnew[k] = 'More than 3'
            #         elif int(vnew[k]) == 0:
            #             vnew[k] = '0'
            #     except:
            #         vnew[k] = vnew[k]
            #
            #     try:
            #         if 1 <= int(vold[k]) <= 3:
            #             vold[k] = '1 to 3'
            #         elif int(vold[k]) > 3:
            #             vold[k] = 'More than 3'
            #         elif int(vold[k]) == 0:
            #             vold[k] = '0'
            #     except:
            #         vold[k] = vold[k]

            total_cost += distort[k].loc[vnew[k], vold[k]]

    return total_cost
