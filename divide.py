# -*- coding:utf-8 -*- 
#Author: shenying
#Date: 18-7-31 下午2:41

import pandas as pd
import numpy as np
#input
path_df_D='../recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv'
#output
path_df_part_1= '../recomm/data/divide/df_part_1.csv'
path_df_part_2='../recomm/data/divide/df_part_2.csv'
path_df_part_3='../recomm/data/divide/df_part_3.csv'

path_df_part_1_tar='../recomm/data/divide/df_part_tar_1.csv'
path_df_part_2_tar='../recomm/data/divide/df_part_tar_2.csv'

path_df_part_1_uic_label='../recomm/data/divide/df_part_1_uic_label.csv'
path_df_part_2_uic_label='../recomm/data/divide/df_part_2_uic_label.csv'
path_df_part_3_uic='../recomm/data/divide/df_part_3_uic.csv'

def divideData():
    batch = 0
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H')
    for df in pd.read_csv(open(path_df_D, 'r'),
                          parse_dates=['time'],
                          index_col=['time'],
                          date_parser=dateparse,
                          chunksize=100000):  # operation on chunk as the data file is too large
        try:
            df_part_1 = df['2014-11-22':'2014-11-27']
            df_part_1_tar = df['2014-11-28']
            df_part_2 = df['2014-11-29':'2014-12-04']
            df_part_2_tar = df['2014-12-05']
            df_part_3 = df['2014-12-13':'2014-12-18']

            df_part_1.to_csv(path_df_part_1,
                             columns=['user_id', 'item_id', 'behavior_type', 'item_category'],
                             header=False, mode='a')
            df_part_1_tar.to_csv(path_df_part_1_tar,
                                 columns=['user_id', 'item_id', 'behavior_type', 'item_category'],
                                 header=False, mode='a')
            df_part_2.to_csv(path_df_part_2,
                             columns=['user_id', 'item_id', 'behavior_type', 'item_category'],
                             header=False, mode='a')
            df_part_2_tar.to_csv(path_df_part_2_tar,
                                 columns=['user_id', 'item_id', 'behavior_type', 'item_category'],
                                 header=False, mode='a')
            df_part_3.to_csv(path_df_part_3,
                             columns=['user_id', 'item_id', 'behavior_type', 'item_category'],
                             header=False, mode='a')

            batch += 1
            print('chunk %d done.' % batch)

        except StopIteration:
            print("divide the data set finish.")
            break
def divideUic1():
    data_file = open(path_df_part_1, 'r')
    try:
        df_part_1 = pd.read_csv(data_file, index_col=False)
        df_part_1.columns = ['time', 'user_id', 'item_id', 'behavior_type', 'item_category']
    finally:
        data_file.close()
    df_part_1_uic = df_part_1.drop_duplicates(['user_id', 'item_id', 'item_category'])[
        ['user_id', 'item_id', 'item_category']]

    data_file = open(path_df_part_1_tar, 'r')
    try:
        df_part_1_tar = pd.read_csv(data_file, index_col=False, parse_dates=[0])
        df_part_1_tar.columns = ['time', 'user_id', 'item_id', 'behavior_type', 'item_category']
    finally:
        data_file.close()

    # uic + label
    df_part_1_uic_label_1 = df_part_1_tar[df_part_1_tar['behavior_type'] == 4][['user_id', 'item_id', 'item_category']]
    df_part_1_uic_label_1.drop_duplicates(['user_id', 'item_id'], 'last', inplace=True)
    df_part_1_uic_label_1['label'] = 1
    df_part_1_uic_label = pd.merge(df_part_1_uic,
                                   df_part_1_uic_label_1,
                                   on=['user_id', 'item_id', 'item_category'],
                                   how='left').fillna(0).astype('int')
    df_part_1_uic_label.to_csv(path_df_part_1_uic_label, index=False)
def divideUic2():
    data_file = open(path_df_part_2, 'r')
    try:
        df_part_2 = pd.read_csv(data_file, index_col=False)
        df_part_2.columns = ['time', 'user_id', 'item_id', 'behavior_type', 'item_category']
    finally:
        data_file.close()
    df_part_2_uic = df_part_2.drop_duplicates(['user_id', 'item_id', 'item_category'])[
        ['user_id', 'item_id', 'item_category']]

    data_file = open(path_df_part_2_tar, 'r')
    try:
        df_part_2_tar = pd.read_csv(data_file, index_col=False, parse_dates=[0])
        df_part_2_tar.columns = ['time', 'user_id', 'item_id', 'behavior_type', 'item_category']
    finally:
        data_file.close()

    # uic + label
    df_part_2_uic_label_1 = df_part_2_tar[df_part_2_tar['behavior_type'] == 4][['user_id', 'item_id', 'item_category']]
    df_part_2_uic_label_1.drop_duplicates(['user_id', 'item_id'], 'last', inplace=True)
    df_part_2_uic_label_1['label'] = 1
    df_part_2_uic_label = pd.merge(df_part_2_uic,
                                   df_part_2_uic_label_1,
                                   on=['user_id', 'item_id', 'item_category'],
                                   how='left').fillna(0).astype('int')
    df_part_2_uic_label.to_csv(path_df_part_2_uic_label, index=False)
def divideUic3():
    data_file = open(path_df_part_3, 'r')
    try:
        df_part_3 = pd.read_csv(data_file, index_col=False)
        df_part_3.columns = ['time', 'user_id', 'item_id', 'behavior_type', 'item_category']
    finally:
        data_file.close()
    df_part_3_uic = df_part_3.drop_duplicates(['user_id', 'item_id', 'item_category'])[
        ['user_id', 'item_id', 'item_category']]
    df_part_3_uic.to_csv(path_df_part_3_uic, index=False)
if __name__=='__main__':
    divideData()
    divideUic1()
    divideUic2()
    divideUic3()
