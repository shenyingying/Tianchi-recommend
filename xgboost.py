# -*- coding:utf-8 -*- 
#Author: shenying
#Date: 18-8-8 下午9:13

path_df_part_1_uic_label = "../recomm/data/divide/df_part_1_uic_label.csv"
path_df_part_2_uic_label = "../recomm/data/divide/df_part_2_uic_label.csv"
path_df_part_3_uic = "../recomm/data/divide/df_part_3_uic.csv"

path_df_part_1_U = '../recomm/data/feature/u_11.csv'
path_df_part_1_I = '../recomm/data/feature/i_11.csv'
path_df_part_1_C = '../recomm/data/feature/c_11.csv'
path_df_part_1_IC = '../recomm/data/feature/ic_11.csv'
path_df_part_1_UI = '../recomm/data/feature/ui_11.csv'
path_df_part_1_UC = '../recomm/data/feature/uc_11.csv'

path_df_part_2_U = "../recomm/data/feature/df_part_2_U.csv"
path_df_part_2_I = "../recomm/data/feature/df_part_2_I.csv"
path_df_part_2_C = "../recomm/data/feature/df_part_2_C.csv"
path_df_part_2_IC = "../recomm/data/feature/df_part_2_IC.csv"
path_df_part_2_UI = "../recomm/data/feature/df_part_2_UI.csv"
path_df_part_2_UC = "../recomm/data/feature/df_part_2_UC.csv"

path_df_part_3_U = "../recomm/data/feature/df_part_3_U.csv"
path_df_part_3_I = "../recomm/data/feature/df_part_3_I.csv"
path_df_part_3_C = "../recomm/data/feature/df_part_3_C.csv"
path_df_part_3_IC = "../recomm/data/feature/df_part_3_IC.csv"
path_df_part_3_UI = "../recomm/data/feature/df_part_3_UI.csv"
path_df_part_3_UC = "../recomm/data/feature/df_part_3_UC.csv"

path_df_part_1_uic_label_cluster = "../recomm/data/subsample/df_part_1_uic_label_cluster_.csv"
path_df_part_2_uic_label_cluster = "../recomm/data/subsample/df_part_2_uic_label_cluster_.csv"

# item_sub_set P
path_df_P = '../recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv'

##### output file
path_df_result = '../recomm/data/xgboost/res_xgb1.csv'
path_df_result_tmp = '../recomm/data/xgboost/res_xgb1_tmp.csv'

# depending package
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

import time

rcParams['figure.figsize'] = 12, 4
# train = pd.read_csv('train_modified.csv')
target = 'Disbursed'
IDcol = 'ID'


# some functions
def df_read(path, mode='r'):
    '''the definition of dataframe loading function
    '''
    data_file = open(path, mode)
    try:
        df = pd.read_csv(data_file, index_col=False)
    finally:
        data_file.close()
    return df


def subsample(df, sub_size):
    '''the definition of sub-sampling function
    @param df: dataframe
    @param sub_size: sub_sample set size

    @return sub-dataframe with the same formation of df
    '''
    if sub_size >= len(df):
        return df
    else:
        return df.sample(n=sub_size)


##### loading data of part 1 & 2
df_part_1_uic_label_cluster = df_read(path_df_part_1_uic_label_cluster)
df_part_2_uic_label_cluster = df_read(path_df_part_2_uic_label_cluster)

df_part_1_U = df_read(path_df_part_1_U)
df_part_1_I = df_read(path_df_part_1_I)
df_part_1_C = df_read(path_df_part_1_C)
df_part_1_IC = df_read(path_df_part_1_IC)
df_part_1_UI = df_read(path_df_part_1_UI)
df_part_1_UC = df_read(path_df_part_1_UC)

df_part_2_U = df_read(path_df_part_2_U)
df_part_2_I = df_read(path_df_part_2_I)
df_part_2_C = df_read(path_df_part_2_C)
df_part_2_IC = df_read(path_df_part_2_IC)
df_part_2_UI = df_read(path_df_part_2_UI)
df_part_2_UC = df_read(path_df_part_2_UC)


##### generation and splitting to training set & valid set
def valid_train_set_construct(valid_ratio=0.5, valid_sub_ratio=0.5, train_np_ratio=1, train_sub_ratio=0.5):
    '''
    # generation of train set
    @param valid_ratio: float ~ [0~1], the valid set ratio in total set and the rest is train set
    @param valid_sub_ratio: float ~ (0~1), random sample ratio of valid set
    @param train_np_ratio:(1~1200), the sub-sample ratio of training set for N/P balanced.
    @param train_sub_ratio: float ~ (0~1), random sample ratio of train set after N/P subsample

    @return valid_X, valid_y, train_X, train_y
    '''
    msk_1 = np.random.rand(len(df_part_1_uic_label_cluster)) < valid_ratio
    msk_2 = np.random.rand(len(df_part_2_uic_label_cluster)) < valid_ratio

    valid_df_part_1_uic_label_cluster = df_part_1_uic_label_cluster.loc[msk_1]
    valid_df_part_2_uic_label_cluster = df_part_2_uic_label_cluster.loc[msk_2]

    valid_part_1_uic_label = valid_df_part_1_uic_label_cluster[valid_df_part_1_uic_label_cluster['class'] == 0].sample(
        frac=valid_sub_ratio)
    valid_part_2_uic_label = valid_df_part_2_uic_label_cluster[valid_df_part_2_uic_label_cluster['class'] == 0].sample(
        frac=valid_sub_ratio)

    ### constructing valid set
    for i in range(1, 1001, 1):
        valid_part_1_uic_label_0_i = valid_df_part_1_uic_label_cluster[valid_df_part_1_uic_label_cluster['class'] == i]
        if len(valid_part_1_uic_label_0_i) != 0:
            valid_part_1_uic_label_0_i = valid_part_1_uic_label_0_i.sample(frac=valid_sub_ratio)
            valid_part_1_uic_label = pd.concat([valid_part_1_uic_label, valid_part_1_uic_label_0_i])

        valid_part_2_uic_label_0_i = valid_df_part_2_uic_label_cluster[valid_df_part_2_uic_label_cluster['class'] == i]
        if len(valid_part_2_uic_label_0_i) != 0:
            valid_part_2_uic_label_0_i = valid_part_2_uic_label_0_i.sample(frac=valid_sub_ratio)
            valid_part_2_uic_label = pd.concat([valid_part_2_uic_label, valid_part_2_uic_label_0_i])

    valid_part_1_df = pd.merge(valid_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_I, how='left', on=['item_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_C, how='left', on=['item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_IC, how='left', on=['item_id', 'item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UC, how='left', on=['user_id', 'item_category'])

    valid_part_2_df = pd.merge(valid_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_I, how='left', on=['item_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_C, how='left', on=['item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_IC, how='left', on=['item_id', 'item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UC, how='left', on=['user_id', 'item_category'])

    valid_df = pd.concat([valid_part_1_df, valid_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    valid_df.fillna(-1, inplace=True)

    # using all the features for valid rf model
    valid_X = valid_df.as_matrix(
        ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
         'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
         'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
         'u_b4_rate', 'u_b4_diff_hours',
         'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
         'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
         'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
         'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
         'i_b4_rate', 'i_b4_diff_hours',
         'c_u_count_in_6', 'c_u_count_in_3', 'c_u_count_in_1',
         'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
         'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
         'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
         'c_b4_rate', 'c_b4_diff_hours',
         'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
         'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
         'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
         'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
         'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
         'ui_b1_last_hours', 'ui_b2_last_hours', 'ui_b3_last_hours', 'ui_b4_last_hours',
         'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
         'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
         'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
         'uc_b_count_rank_in_u',
         'uc_b1_last_hours', 'uc_b2_last_hours', 'uc_b3_last_hours', 'uc_b4_last_hours'])
    valid_y = valid_df['label'].values
    print("valid subset is generated.")

    ### constructing training set
    train_df_part_1_uic_label_cluster = df_part_1_uic_label_cluster.loc[~msk_1]
    train_df_part_2_uic_label_cluster = df_part_2_uic_label_cluster.loc[~msk_2]

    train_part_1_uic_label = train_df_part_1_uic_label_cluster[train_df_part_1_uic_label_cluster['class'] == 0].sample(
        frac=train_sub_ratio)
    train_part_2_uic_label = train_df_part_2_uic_label_cluster[train_df_part_2_uic_label_cluster['class'] == 0].sample(
        frac=train_sub_ratio)

    frac_ratio = train_sub_ratio * train_np_ratio / 1200
    for i in range(1, 1001, 1):
        train_part_1_uic_label_0_i = train_df_part_1_uic_label_cluster[train_df_part_1_uic_label_cluster['class'] == i]
        if len(train_part_1_uic_label_0_i) != 0:
            train_part_1_uic_label_0_i = train_part_1_uic_label_0_i.sample(frac=frac_ratio)
            train_part_1_uic_label = pd.concat([train_part_1_uic_label, train_part_1_uic_label_0_i])

        train_part_2_uic_label_0_i = train_df_part_2_uic_label_cluster[train_df_part_2_uic_label_cluster['class'] == i]
        if len(train_part_2_uic_label_0_i) != 0:
            train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac=frac_ratio)
            train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])

    # constructing training set
    train_part_1_df = pd.merge(train_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_I, how='left', on=['item_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_C, how='left', on=['item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_IC, how='left', on=['item_id', 'item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UC, how='left', on=['user_id', 'item_category'])

    train_part_2_df = pd.merge(train_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_I, how='left', on=['item_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_C, how='left', on=['item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_IC, how='left', on=['item_id', 'item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UC, how='left', on=['user_id', 'item_category'])

    train_df = pd.concat([train_part_1_df, train_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    train_df.fillna(-1, inplace=True)

    # using all the features for training rf model
    train_X = train_df.as_matrix(
        ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
         'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
         'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
         'u_b4_rate', 'u_b4_diff_hours',
         'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
         'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
         'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
         'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
         'i_b4_rate', 'i_b4_diff_hours',
         'c_u_count_in_6', 'c_u_count_in_3', 'c_u_count_in_1',
         'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
         'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
         'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
         'c_b4_rate', 'c_b4_diff_hours',
         'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
         'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
         'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
         'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
         'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
         'ui_b1_last_hours', 'ui_b2_last_hours', 'ui_b3_last_hours', 'ui_b4_last_hours',
         'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
         'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
         'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
         'uc_b_count_rank_in_u',
         'uc_b1_last_hours', 'uc_b2_last_hours', 'uc_b3_last_hours', 'uc_b4_last_hours'])
    train_y = train_df['label'].values
    print("train subset is generated.")

    return valid_X, valid_y, train_X, train_y


##### generation of training set & valid set
def train_set_construct(np_ratio=1, sub_ratio=1):
    '''
    # generation of train set
    @param np_ratio: int, the sub-sample rate of training set for N/P balanced.
    @param sub_ratio: float ~ (0~1], the further sub-sample rate of training set after N/P balanced.
    '''
    train_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(
        frac=sub_ratio)
    train_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(
        frac=sub_ratio)

    frac_ratio = sub_ratio * np_ratio / 1200
    for i in range(1, 1001, 1):
        train_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        train_part_1_uic_label_0_i = train_part_1_uic_label_0_i.sample(frac=frac_ratio)
        train_part_1_uic_label = pd.concat([train_part_1_uic_label, train_part_1_uic_label_0_i])

        train_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac=frac_ratio)
        train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])
    print("training subset uic_label keys is selected.")

    # constructing training set
    train_part_1_df = pd.merge(train_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_I, how='left', on=['item_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_C, how='left', on=['item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_IC, how='left', on=['item_id', 'item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UC, how='left', on=['user_id', 'item_category'])

    train_part_2_df = pd.merge(train_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_I, how='left', on=['item_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_C, how='left', on=['item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_IC, how='left', on=['item_id', 'item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UC, how='left', on=['user_id', 'item_category'])

    train_df = pd.concat([train_part_1_df, train_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    train_df.fillna(-1, inplace=True)

    # using all the features for training rf model
    train_X = train_df.as_matrix(
        ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
         'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
         'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
         'u_b4_rate', 'u_b4_diff_hours',
         'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
         'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
         'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
         'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
         'i_b4_rate', 'i_b4_diff_hours',
         'c_u_count_in_6', 'c_u_count_in_3', 'c_u_count_in_1',
         'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
         'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
         'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
         'c_b4_rate', 'c_b4_diff_hours',
         'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
         'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
         'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
         'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
         'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
         'ui_b1_last_hours', 'ui_b2_last_hours', 'ui_b3_last_hours', 'ui_b4_last_hours',
         'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
         'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
         'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
         'uc_b_count_rank_in_u',
         'uc_b1_last_hours', 'uc_b2_last_hours', 'uc_b3_last_hours', 'uc_b4_last_hours'])
    train_y = train_df['label'].values
    print("train subset is generated.")
    return train_X, train_y


def valid_set_construct(sub_ratio=0.1):
    '''
    # generation of valid set
    @param sub_ratio: float ~ (0~1], the sub-sample rate of original valid set
    '''
    valid_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(
        frac=sub_ratio)
    valid_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(
        frac=sub_ratio)

    for i in range(1, 1001, 1):
        valid_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        valid_part_1_uic_label_0_i = valid_part_1_uic_label_0_i.sample(frac=sub_ratio)
        valid_part_1_uic_label = pd.concat([valid_part_1_uic_label, valid_part_1_uic_label_0_i])

        valid_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        valid_part_2_uic_label_0_i = valid_part_2_uic_label_0_i.sample(frac=sub_ratio)
        valid_part_2_uic_label = pd.concat([valid_part_2_uic_label, valid_part_2_uic_label_0_i])

    # constructing valid set
    valid_part_1_df = pd.merge(valid_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_I, how='left', on=['item_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_C, how='left', on=['item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_IC, how='left', on=['item_id', 'item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UC, how='left', on=['user_id', 'item_category'])

    valid_part_2_df = pd.merge(valid_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_I, how='left', on=['item_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_C, how='left', on=['item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_IC, how='left', on=['item_id', 'item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UC, how='left', on=['user_id', 'item_category'])

    valid_df = pd.concat([valid_part_1_df, valid_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    valid_df.fillna(-1, inplace=True)

    # using all the features for valid rf model
    valid_X = valid_df.as_matrix(
        ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
         'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
         'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
         'u_b4_rate', 'u_b4_diff_hours',
         'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
         'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
         'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
         'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
         'i_b4_rate', 'i_b4_diff_hours',
         'c_u_count_in_6', 'c_u_count_in_3', 'c_u_count_in_1',
         'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
         'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
         'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
         'c_b4_rate', 'c_b4_diff_hours',
         'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
         'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
         'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
         'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
         'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
         'ui_b1_last_hours', 'ui_b2_last_hours', 'ui_b3_last_hours', 'ui_b4_last_hours',
         'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
         'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
         'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
         'uc_b_count_rank_in_u',
         'uc_b1_last_hours', 'uc_b2_last_hours', 'uc_b3_last_hours', 'uc_b4_last_hours'])
    valid_y = valid_df['label'].values
    print("valid subset is generated.")

    return valid_X, valid_y

# 调参:
# step1:n_estimators:最佳迭代次数
# cv_params={'n_estimators':[400,500,600,700,800,900,1000]} 参数的最佳取值：{'n_estimators': 400} 最佳模型得分:0.116402764827392
# cv_params={'n_estimators':[200,250,300,350,400,450]} 参数的最佳取值：{'n_estimators': 200} 最佳模型得分:0.1270196359263699
# cv_params={'n_estimators':[50,100,150,200,250]}  参数的最佳取值：{'n_estimators': 150}  最佳模型得分:0.1308298446194594
# cv_params={'n_estimators':[130,140,150,160,170]} 参数的最佳取值：{'n_estimators': 140}  最佳模型得分:0.1301806460283527
# step2:min_child_weight,max_depth
# cv_params={'max_depth':[3,4,5,6,7,8,9,10],'min_child_weight':[1,2,3,4,5,6,7,8]}
# # step3:gamma
# cv_params={'gamma':[0.1,0.2,0.3,0.4,0.5,0.6]} 参数的最佳取值：{'gamma': 0.1}  最佳模型得分:0.1368594531153595
# cv_params={'gamma':[0,0.05,0.1]} 参数的最佳取值：{'gamma': 0.05}  最佳模型得分:0.1391325809983267
# # step4:
# cv_params={'subsample':[0.6,0.7,0.8,0.9],'colsample_bytree':[0.6,0.7,0.8,0.9]} 参数的最佳取值：{'colsample_bytree': 0.8, 'subsample': 0.6} 最佳模型得分:0.1383508979562909
# # step5:
# cv_params={'reg_alpha':[0.05,0.1,1,2,3],'reg_lambda':[0.05,0.1,1,2,3]} 参数的最佳取值：{'reg_lambda': 0.1, 'reg_alpha': 0.05}  最佳模型得分:0.13678680215318179
# # step6:
# cv_params={'learning_rate':[0.01, 0.05, 0.07, 0.1, 0.2]} 参数的最佳取值：{'learning_rate': 0.01} 最佳模型得分:0.13978412607191046
# other_params={'n_estimators':140,
#               'max_depth':10,
#               'min_child_weight':8,
#               'gamma':0.05,
#               'subsample':0.6,
#               'colsample_bytree':0.8,
#               'reg_alpha':0.05,
#               'reg_lambda':0.1,
#               'learning_rate':0.01,
#               'objective':'binary:logistic',
#               'nthread':5,
#               'scale_pos_weight':1,
#               'seed':27}
# model=xgb.XGBRegressor(**other_params)
# optimized_GBM=GridSearchCV(estimator=model,param_grid=cv_params,scoring='r2',cv=5,verbose=1,n_jobs=4)
# optimized_GBM.fit(train_X,train_y)
# evalute_result=optimized_GBM.grid_scores_
# print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

# build model and fitting
train_X, train_y = train_set_construct(np_ratio=5, sub_ratio=1)
xgb1=XGBClassifier(learning_rate=0.01,
                   n_estimators=140,
                   max_depth=10,
                   min_child_weight=8,
                   gamma=0.05,
                   subsample=0.6,
                   colsample_bytree=0.8,
                   reg_alpha=0.05,
                   objective='binary:logistic',
                   nthread=4,
                   scale_pos_weight=1,
                   seed=27)
xgb1.fit(train_X,train_y)
df_part_3_U  = df_read(path_df_part_3_U )
df_part_3_I  = df_read(path_df_part_3_I )
df_part_3_C  = df_read(path_df_part_3_C )
df_part_3_IC = df_read(path_df_part_3_IC)
df_part_3_UI = df_read(path_df_part_3_UI)
df_part_3_UC = df_read(path_df_part_3_UC)

# process by chunk as ui-pairs size is too big
batch = 0
for pred_uic in pd.read_csv(open(path_df_part_3_uic, 'r'), chunksize = 100000):
    try:
        # construct of prediction sample set
        pred_df = pd.merge(pred_uic, df_part_3_U,  how='left', on=['user_id'])
        pred_df = pd.merge(pred_df,  df_part_3_I,  how='left', on=['item_id'])
        pred_df = pd.merge(pred_df,  df_part_3_C,  how='left', on=['item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_IC, how='left', on=['item_id','item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_UI, how='left', on=['user_id','item_id','item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_UC, how='left', on=['user_id','item_category'])

        # fill the missing value as -1 (missing value are time features)
        pred_df.fillna(-1, inplace=True)

        # using all the features for training RF model
        pred_X = pred_df.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6',
                                    'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3',
                                    'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1',
                                    'u_b4_rate','u_b4_diff_hours',
                                    'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                    'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6',
                                    'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                    'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1',
                                    'i_b4_rate','i_b4_diff_hours',
                                    'c_u_count_in_6','c_u_count_in_3','c_u_count_in_1',
                                    'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                    'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                    'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                    'c_b4_rate','c_b4_diff_hours',
                                    'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c',
                                    'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                    'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                    'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1',
                                    'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                    'ui_b1_last_hours','ui_b2_last_hours','ui_b3_last_hours','ui_b4_last_hours',
                                    'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6',
                                    'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3',
                                    'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                    'uc_b_count_rank_in_u',
                                    'uc_b1_last_hours','uc_b2_last_hours','uc_b3_last_hours','uc_b4_last_hours'])

        # predicting
        # pred_y = (RF_clf.predict_proba(pred_X)[:,1] > 0.75).astype(int)
        pred_y = (xgb1.predict_proba(pred_X)[:, 1] > 0.75).astype(int)

        # generation of U-I pairs those predicted to buy
        pred_df['pred_label'] = pred_y
        # add to result csv
        pred_df[pred_df['pred_label'] == 1].to_csv(path_df_result_tmp,
                                                   columns=['user_id','item_id'],
                                                   index=False, header=False, mode='a')

        batch += 1
        print("prediction chunk %d done." % batch)

    except StopIteration:
        print("prediction finished.")
        break

df_P = df_read(path_df_P)
df_P_item = df_P.drop_duplicates(['item_id'])[['item_id']]
df_pred = pd.read_csv(open(path_df_result_tmp,'r'), index_col=False, header=None)
df_pred.columns = ['user_id', 'item_id']

# output result
df_pred_P = pd.merge(df_pred, df_P_item, on=['item_id'], how='inner')[['user_id', 'item_id']]
df_pred_P.to_csv(path_df_result, index=False)