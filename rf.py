# -*- coding:utf-8 -*- 
#Author: shenying
#Date: 18-8-1 下午5:20

#用随机森林rf进行预测
# input
# 用聚类的label去
path_df_part_1_uic_label_cluster='../recomm/data/subsample/path_df_part1_uic_label_cluster.csv'
path_df_part_2_uic_label_cluster='../recomm/data/subsample/path_df_part2_uic_label_cluster.csv'
path_df_part_3_uic='../recomm/data/divide/df_part_3_uic.csv'
# feature
path_df_part_1_U='../recomm/data/feature/u_1.csv'
path_df_part_1_I= '../recomm/data/feature/i_1.csv'
path_df_part_1_C='../recomm/data/feature/c_1.csv'
path_df_part_1_IC='../recomm/data/feature/ic_1.csv'
path_df_part_1_UI='../recomm/data/feature/ui_1.csv'
path_df_part_1_UC='../recomm/data/feature/uc_1.csv'

path_df_part_2_U='../recomm/data/feature/u_2.csv'
path_df_part_2_I='../recomm/data/feature/i_2.csv'
path_df_part_2_C='../recomm/data/feature/c_2.csv'
path_df_part_2_IC='../recomm/data/feature/ic_2.csv'
path_df_part_2_UI='../recomm/data/feature/ui_2.csv'
path_df_part_2_UC='../recomm/data/feature/uc_2.csv'

path_df_part_3_U='../recomm/data/feature/u_3.csv'
path_df_part_3_I='../recomm/data/feature/i_3.csv'
path_df_part_3_C='../recomm/data/feature/c_3.csv'
path_df_part_3_IC='../recomm/data/feature/ic_3.csv'
path_df_part_3_UI='../recomm/data/feature/ui_3.csv'
path_df_part_3_UC='../recomm/data/feature/uc_3.csv'
# normalize scalar
path_df_part1_scaler='../recomm/data/subsample/path_df_part1_scaler'
path_df_part2_scaler='../recomm/data/subsample/path_df_part2_scaler'

#p
path_df_P='../recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv'

#output
path_df_result='../recomm/data/rf/res_lr.csv'
path_df_result_tmp='../recomm/data/rf/res_lr_tmp.csv'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import time

def df_read(path, mode = 'r'):
    data_file = open(path, mode)
    try:     df = pd.read_csv(data_file, index_col = False)
    finally: data_file.close()
    return   df
def subsample(df, sub_size):
    if sub_size >= len(df):
        return df
    else:
        return df.sample(n=sub_size)
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
    df_part_1_uic_label_cluster = df_read(path_df_part_1_uic_label_cluster)
    df_part_2_uic_label_cluster = df_read(path_df_part_2_uic_label_cluster)
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
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UI, how='left', on=['user_id', 'item_id', 'item_category', 'label'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UC, how='left', on=['user_id', 'item_category'])

    valid_part_2_df = pd.merge(valid_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_I, how='left', on=['item_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_C, how='left', on=['item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_IC, how='left', on=['item_id', 'item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UI, how='left',
                               on=['user_id', 'item_id', 'item_category', 'label'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UC, how='left', on=['user_id', 'item_category'])

    valid_df = pd.concat([valid_part_1_df, valid_part_2_df],sort=False)

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

    train_df = pd.concat([train_part_1_df, train_part_2_df],sort=False)

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

def selNP():
    f1_scores = []
    np_ratios = []
    for np_ratio in [1, 5, 10, 30, 50, 70, 100]:
        t1 = time.time()
        # generation of training and valid set

        valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio=0.2,
                                                                       valid_sub_ratio=1,
                                                                       train_np_ratio=np_ratio,
                                                                       train_sub_ratio=1)
        # generation of rf model and fit
        rf_clf = RandomForestClassifier(max_depth=35,
                                        n_estimators=100,
                                        max_features="sqrt",
                                        verbose=True)
        rf_clf.fit(train_X, train_y)

        # validation and evaluation
        valid_y_pred = rf_clf.predict(valid_X)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        np_ratios.append(np_ratio)
        print('rf_clf [NP ratio = %d] is fitted' % np_ratio)

        t2 = time.time()
        print('time used %d s' % (t2 - t1))
    f1 = plt.figure(1)
    plt.plot(np_ratios, f1_scores, label="md=35,nt=100")
    plt.xlabel('NP ratio')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of NP ratio - RF')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()

def selNestimators():
    f1_scores = []
    n_trees = []
    valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio=0.2,
                                                                   valid_sub_ratio=1,
                                                                   train_np_ratio=5,
                                                                   train_sub_ratio=1)
    for nt in [10, 20, 40, 80, 120, 160, 200, 300, 400, 500, 600, 700, 800]:
        t1 = time.time()
        # generation of training and valid set
        # generation of RF model and fit
        RF_clf = RandomForestClassifier(n_estimators=nt,
                                        max_depth=35,
                                        max_features="sqrt",
                                        verbose=True)
        RF_clf.fit(train_X, train_y)

        # validation and evaluation
        valid_y_pred = RF_clf.predict(valid_X)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        n_trees.append(nt)
        print('RF_clf [n_estimators = %d] is fitted' % nt)

        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    # plot the result
    f1 = plt.figure(1)
    plt.plot(n_trees, f1_scores, label="md=35,np_ratio=5")
    plt.xlabel('n_trees')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of RF n_trees')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()

def selMaxDepths():
    f1_scores = []
    max_depths = []

    valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio=0.2,
                                                                   valid_sub_ratio=1,
                                                                   train_np_ratio=10,
                                                                   train_sub_ratio=1)

    for md in [1, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
        t1 = time.time()

        # generation of RF model and fit
        RF_clf = RandomForestClassifier(max_depth=md, n_estimators=150, max_features="sqrt", verbose=True)
        RF_clf.fit(train_X, train_y)

        # validation and evaluation
        valid_y_pred = RF_clf.predict(valid_X)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        max_depths.append(md)
        print('RF_clf [max_depths = %d] is fitted' % md)

        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    # plot the result
    f1 = plt.figure(1)
    plt.plot(max_depths, f1_scores, label="np_ratio=10,nt=150")
    plt.xlabel('max_depths')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of RF max_depths')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()

def selMinSamples():
    valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio=0.2,
                                                                   valid_sub_ratio=1,
                                                                   train_np_ratio=10,
                                                                   train_sub_ratio=1)
    min_samples_splits = []
    f1_scores = []
    for mss in [2, 4, 6, 8, 10, 15, 20, 18, 20, 25, 30, 40, 50, 60, 70, 80, 90]:
        t1 = time.time()
        # generation of training and valid set

        # generation of RF model and fit
        RF_clf = RandomForestClassifier(min_samples_split=mss,
                                        max_depth=30,
                                        n_estimators=150,
                                        max_features="sqrt",
                                        verbose=True)
        RF_clf.fit(train_X, train_y)

        # validation and evaluation
        valid_y_pred = RF_clf.predict(valid_X)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        min_samples_splits.append(mss)
        print('RF_clf [min_samples_splits = %d] is fitted' % mss)

        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    # plot the result
    f1 = plt.figure(1)
    plt.plot(min_samples_splits, f1_scores, label="np=10,nt=150,md=30")
    plt.xlabel('min_samples_split')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of RF min_samples_split')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()

def selCutOff():
    f1_scores = []
    cut_offs = []

    valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio=0.2,
                                                                   valid_sub_ratio=1,
                                                                   train_np_ratio=10,
                                                                   train_sub_ratio=1)

    # generation of RF model and fit
    RF_clf = RandomForestClassifier(max_depth=30,
                                    n_estimators=150,
                                    max_features="sqrt",
                                    verbose=True)
    RF_clf.fit(train_X, train_y)

    for cutoff in np.arange(0.1, 1, 0.05):
        # validation and evaluation
        valid_y_pred = (RF_clf.predict_proba(valid_X)[:, 1] > cutoff).astype(int)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        cut_offs.append(cutoff)
        print('RF_clf [cutoff = %.2f] is fitted' % cutoff)

    # plot the result
    f1 = plt.figure(1)
    plt.plot(cut_offs, f1_scores, label="np_ratio=10,nt=150,md=30")
    plt.xlabel('cut_offs')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of RF cut_offs')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()



def trainRF():
    # build model and fitting
    RF_clf = RandomForestClassifier(max_depth=30, n_estimators=150, max_features="sqrt")
    train_X, train_y = train_set_construct(np_ratio=5, sub_ratio=1)
    RF_clf.fit(train_X, train_y)

    ##### predicting
    # loading feature data
    df_part_3_U = df_read(path_df_part_3_U)
    df_part_3_I = df_read(path_df_part_3_I)
    df_part_3_C = df_read(path_df_part_3_C)
    df_part_3_IC = df_read(path_df_part_3_IC)
    df_part_3_UI = df_read(path_df_part_3_UI)
    df_part_3_UC = df_read(path_df_part_3_UC)

    # process by chunk as ui-pairs size is too big
    batch = 0
    for pred_uic in pd.read_csv(open(path_df_part_3_uic, 'r'), chunksize=100000):
        try:
            # construct of prediction sample set
            pred_df = pd.merge(pred_uic, df_part_3_U, how='left', on=['user_id'])
            pred_df = pd.merge(pred_df, df_part_3_I, how='left', on=['item_id'])
            pred_df = pd.merge(pred_df, df_part_3_C, how='left', on=['item_category'])
            pred_df = pd.merge(pred_df, df_part_3_IC, how='left', on=['item_id', 'item_category'])
            pred_df = pd.merge(pred_df, df_part_3_UI, how='left', on=['user_id', 'item_id', 'item_category'])
            pred_df = pd.merge(pred_df, df_part_3_UC, how='left', on=['user_id', 'item_category'])

            # fill the missing value as -1 (missing value are time features)
            pred_df.fillna(-1, inplace=True)

            # using all the features for training RF model
            pred_X = pred_df.as_matrix(
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

            # predicting
            pred_y = (RF_clf.predict_proba(pred_X)[:, 1] > 0.75).astype(int)

            # generation of U-I pairs those predicted to buy
            pred_df['pred_label'] = pred_y
            # add to result csv
            pred_df[pred_df['pred_label'] == 1].to_csv(path_df_result_tmp,
                                                       columns=['user_id', 'item_id'],
                                                       index=False, header=False, mode='a')

            batch += 1
            print("prediction chunk %d done." % batch)

        except StopIteration:
            print("prediction finished.")
            break

if __name__=="__main__":
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

    selNP()
    selNestimators()
    selMaxDepths()
    selMinSamples()
    selCutOff()
    trainRF()
    df_P = df_read(path_df_P)
    df_P_item = df_P.drop_duplicates(['item_id'])[['item_id']]
    df_pred = pd.read_csv(open(path_df_result_tmp, 'r'), index_col=False, header=None)
    df_pred.columns = ['user_id', 'item_id']

    # output result
    df_pred_P = pd.merge(df_pred, df_P_item, on=['item_id'], how='inner')[['user_id', 'item_id']]
    df_pred_P.to_csv(path_df_result, index=False)

