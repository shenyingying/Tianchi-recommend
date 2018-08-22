# -*- coding:utf-8 -*- 
#Author: shenying
#Date: 18-7-31 下午6:49

#对聚类的负样本进行下采样以获的和正样本差不多的比率,进行lr

# input
# 用聚类的label去
path_df_part1_uic_label_cluster='../recomm/data/subsample/path_df_part1_uic_label_cluster.csv'
path_df_part2_uic_label_cluster='../recomm/data/subsample/path_df_part2_uic_label_cluster.csv'
path_df_part3_uic='../recomm/data/divide/df_part_3_uic.csv'
# feature
path_1_u='../recomm/data/feature/u_1.csv'
path_1_i= '../recomm/data/feature/i_1.csv'
path_1_c='../recomm/data/feature/c_1.csv'
path_1_ui='../recomm/data/feature/ui_1.csv'
path_1_uc='../recomm/data/feature/uc_1.csv'
path_1_ic='../recomm/data/feature/ic_1.csv'

path_2_u='../recomm/data/feature/u_2.csv'
path_2_i='../recomm/data/feature/i_2.csv'
path_2_c='../recomm/data/feature/c_2.csv'
path_2_ui='../recomm/data/feature/ui_2.csv'
path_2_uc='../recomm/data/feature/uc_2.csv'
path_2_ic='../recomm/data/feature/ic_2.csv'

path_3_u='../recomm/data/feature/u_3.csv'
path_3_i='../recomm/data/feature/i_3.csv'
path_3_c='../recomm/data/feature/c_3.csv'
path_3_ui='../recomm/data/feature/ui_3.csv'
path_3_uc='../recomm/data/feature/uc_3.csv'
path_3_ic='../recomm/data/feature/ic_3.csv'
# normalize scalar
path_df_part1_scaler='../recomm/data/subsample/path_df_part1_scaler'
path_df_part2_scaler='../recomm/data/subsample/path_df_part2_scaler'

#p
path_df_p='../recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv'

#output
path_df_result='../recomm/data/lr/res_lr.csv'
path_df_result_tmp='../recomm/data/lr/res_lr_tmp.csv'

import pandas as pd
import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt
import time

def df_read(path,mode='r'):
    path_df=open(path,mode)
    try:
        df=pd.read_csv(path_df,index_col=False)
    finally:
        path_df.close()
    return df

def subsample(df,sub_size):
    if sub_size>=len(df):
        return df
    else:
        return df.sample(n=sub_size)

def train_set_construct(np_ratio=1,sub_ratio=1):
    train_part1_uic_label=df_part1_uic_label_cluster[df_part1_uic_label_cluster['class']==0].sample(frac=sub_ratio)
    train_part2_uic_label=df_part2_uic_label_cluster[df_part2_uic_label_cluster['class']==0].sample(frac=sub_ratio)

    frac_ratio=sub_ratio*np_ratio/1200
    for i in range(1,1001,1):
        train_part1_uic_label_0_i=df_part1_uic_label_cluster[df_part1_uic_label_cluster['class']==i]
        train_part1_uic_label_0_i=train_part1_uic_label_0_i.sample(frac=frac_ratio)
        train_part1_uic_label=pd.concat([train_part1_uic_label,train_part1_uic_label_0_i])

        train_part2_uic_label_0_i=df_part2_uic_label_cluster[df_part2_uic_label_cluster['class']==i]
        train_part2_uic_label_0_i=train_part2_uic_label_0_i.sample(frac=frac_ratio)
        train_part2_uic_label=pd.concat([train_part2_uic_label,train_part2_uic_label_0_i])
    print('training subset uic_label keys is selected.')

    train_part1_df=pd.merge(train_part1_uic_label,df_part1_u,how='left',on=['user_id'])
    train_part1_df=pd.merge(train_part1_df,df_part1_i,how='left',on=['item_id'])
    train_part1_df=pd.merge(train_part1_df,df_part1_c,how='left',on=['item_category'])
    train_part1_df = pd.merge(train_part1_df, df_part1_ui, how='left',on=['user_id', 'item_id', 'item_category', 'label'])
    train_part1_df = pd.merge(train_part1_df, df_part1_ic, how='left', on=['item_id', 'item_category'])
    train_part1_df = pd.merge(train_part1_df, df_part1_uc, how='left', on=['user_id', 'item_category'])

    train_X_1 = train_part1_df.as_matrix(
                                        ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
                                         'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
                                         'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
                                         'u_b4_rate',
                                         'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
                                         'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
                                         'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
                                         'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
                                         'i_b4_rate',
                                         'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
                                         'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
                                         'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
                                         'c_b4_rate',
                                         'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
                                         'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
                                         'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
                                         'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
                                         'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
                                         'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
                                         'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
                                         'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
                                         'uc_b_count_rank_in_u'])
    train_y_1=train_part1_df['label'].values
    standard_train_X_1=part1_scaler.transform(train_X_1)

    train_part2_df=pd.merge(train_part2_uic_label, df_part2_u, how='left', on=['user_id'])
    train_part2_df=pd.merge(train_part2_df,df_part2_i,how='left',on=['item_id'])
    train_part2_df=pd.merge(train_part2_df,df_part2_c,how='left',on=['item_category'])
    train_part2_df = pd.merge(train_part2_df, df_part2_ui, how='left',on=['user_id', 'item_id', 'item_category', 'label'])
    train_part2_df = pd.merge(train_part2_df, df_part2_ic, how='left', on=['item_id', 'item_category'])
    train_part2_df = pd.merge(train_part2_df, df_part2_uc, how='left', on=['user_id', 'item_category'])

    train_X_2 = train_part2_df.as_matrix(
                                        ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
                                         'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
                                         'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
                                         'u_b4_rate',
                                         'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
                                         'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
                                         'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
                                         'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
                                         'i_b4_rate',
                                         'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
                                         'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
                                         'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
                                         'c_b4_rate',
                                         'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
                                         'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
                                         'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
                                         'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
                                         'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
                                         'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
                                         'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
                                         'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
                                         'uc_b_count_rank_in_u'])
    train_y_2=train_part2_df['label'].values
    standard_train_X_2=part2_scaler.transform(train_X_2)
    train_x=np.concatenate((standard_train_X_1,standard_train_X_2))
    train_y=np.concatenate((train_y_1,train_y_2))
    return train_x,train_y

def valid_set_construct(sub_ratio=0.1):
    train_part1_uic_label = df_part1_uic_label_cluster[df_part1_uic_label_cluster['class'] == 0].sample(frac=sub_ratio)
    train_part2_uic_label = df_part2_uic_label_cluster[df_part2_uic_label_cluster['class'] == 0].sample(frac=sub_ratio)

    for i in range(1, 1001, 1):
        train_part1_uic_label_0_i = df_part1_uic_label_cluster[df_part1_uic_label_cluster['class'] == i]
        train_part1_uic_label_0_i = train_part1_uic_label_0_i.sample(frac=sub_ratio)
        train_part1_uic_label = pd.concat([train_part1_uic_label, train_part1_uic_label_0_i])

        train_part2_uic_label_0_i = df_part2_uic_label_cluster[df_part2_uic_label_cluster['class'] == i]
        train_part2_uic_label_0_i = train_part2_uic_label_0_i.sample(frac=sub_ratio)
        train_part2_uic_label = pd.concat([train_part2_uic_label, train_part2_uic_label_0_i])

    valid_part1_df = pd.merge(train_part1_uic_label, df_part1_u, how='left', on=['user_id'])
    valid_part1_df = pd.merge(valid_part1_df, df_part1_i, how='left', on=['item_id'])
    valid_part1_df = pd.merge(valid_part1_df, df_part1_c, how='left', on=['item_category'])
    valid_part1_df = pd.merge(valid_part1_df, df_part1_ui, how='left',on=['user_id', 'item_id', 'item_category', 'label'])
    valid_part1_df = pd.merge(valid_part1_df, df_part1_ic, how='left', on=['item_id', 'item_category'])
    valid_part1_df = pd.merge(valid_part1_df, df_part1_uc, how='left', on=['user_id', 'item_category'])

    valid_X_1 = valid_part1_df.as_matrix(
                                        ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
                                         'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
                                         'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
                                         'u_b4_rate',
                                         'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
                                         'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
                                         'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
                                         'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
                                         'i_b4_rate',
                                         'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
                                         'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
                                         'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
                                         'c_b4_rate',
                                         'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
                                         'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
                                         'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
                                         'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
                                         'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
                                         'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
                                         'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
                                         'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
                                         'uc_b_count_rank_in_u'])
    valid_y_1 = valid_part1_df['label'].values
    standard_valid_X_1 = part1_scaler.transform(valid_X_1)

    valid_part2_df = pd.merge(train_part2_uic_label, df_part2_u, how='left', on=['user_id'])
    valid_part2_df = pd.merge(valid_part2_df, df_part2_i, how='left', on=['item_id'])
    valid_part2_df = pd.merge(valid_part2_df, df_part2_c, how='left', on=['item_category'])
    valid_part2_df = pd.merge(valid_part2_df, df_part2_ui, how='left',on=['user_id', 'item_id', 'item_category', 'label'])
    valid_part2_df = pd.merge(valid_part2_df, df_part2_ic, how='left', on=['item_id', 'item_category'])
    valid_part2_df = pd.merge(valid_part2_df, df_part2_uc, how='left', on=['user_id', 'item_category'])

    valid_X_2 = valid_part2_df.as_matrix(
        ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
         'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
         'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
         'u_b4_rate',
         'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
         'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
         'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
         'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
         'i_b4_rate',
         'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
         'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
         'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
         'c_b4_rate',
         'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
         'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
         'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
         'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
         'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
         'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
         'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
         'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
         'uc_b_count_rank_in_u'])
    valid_y_2 = valid_part2_df['label'].values
    standard_valid_X_2 = part2_scaler.transform(valid_X_2)
    valid_x = np.concatenate((standard_valid_X_1, standard_valid_X_2))
    valid_y = np.concatenate((valid_y_1, valid_y_2))
    print('train subset is generated.')
    return valid_x,valid_y

def selNP():
    f1_scores=[]
    np_ratios=[]
    valid_X,valid_Y=valid_set_construct(sub_ratio=0.18)
    for np_ratio in range(1,100,2):
        t1=time.time()
        train_X,train_Y=train_set_construct(np_ratio=np_ratio,sub_ratio=0.5)
        LR_clf=LogisticRegression(penalty='l1',verbose=True)
        LR_clf.fit(train_X,train_Y)

        valid_y_pred=LR_clf.predict(valid_X)
        f1_scores.append(metrics.f1_score(valid_Y,valid_y_pred))
        np_ratios.append(np_ratio)

        print('LR_clf [NP ratio=%d] is fitted'%np_ratio)
        t2=time.time()
        print('time used %d'%(t2-t1))
    f1=plt.figure(1)
    plt.plot(np_ratios,f1_scores,label='penalty=l1')
    plt.xlabel('NP ratio')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of NP ratio -LR')
    plt.legend(loc=4)
    plt.grid(True,linewidth=0.3)
    plt.show()
def selC1():
    f1_scores=[]
    cs=[]
    valid_x,valid_y=valid_set_construct(sub_ratio=0.18)
    train_x,train_y=train_set_construct(np_ratio=35,sub_ratio=0.5)
    for c in [0.001,0.01,0.1,1,10,100,1000]:
        t1=time.time()
        LR_clf=LogisticRegression(C=c,penalty='l1',verbose=True)
        LR_clf.fit(train_x,train_y)

        valid_y_pred=LR_clf.predict(valid_x)
        f1_scores.append(metrics.f1_score(valid_y,valid_y_pred))
        cs.append(c)
        print('LR_clf[C=%.3f] is fitted'%c)
        t2=time.time()
        print('time used %d s'%(t2-t1))
    f1=plt.figure(1)
    plt.plot(cs,f1_scores,label="penalty='l1',np_ratio=35")
    plt.xlabel('C')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of C-LR')
    plt.legend(loc=4)
    plt.grid(True,linewidth=0.3)
    plt.show()
def selC2():
    f1_scores = []
    cut_off = []
    valid_x, valid_y = valid_set_construct(sub_ratio=0.18)
    train_x, train_y = train_set_construct(np_ratio=55, sub_ratio=0.5)
    for co in (0.1,1,0.1):
        t1 = time.time()
        LR_clf = LogisticRegression(C=co, penalty='l1', verbose=True)
        LR_clf.fit(train_x, train_y)

        valid_y_pred = LR_clf.predict(valid_x)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        cut_off.append(co)
        print('LR_clf[C=%.3f] is fitted' % co)
        t2 = time.time()
        print('time used %d s' % (t2 - t1))
    f1 = plt.figure(1)
    plt.plot(cut_off, f1_scores, label="penalty='l1', np_ratio=55")
    plt.xlabel('C')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of cut_off-LR')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()

def train_set_part3():
    df_part3_u = df_read(path_3_u)
    df_part3_i = df_read(path_3_i)
    df_part3_c = df_read(path_3_c)
    df_part3_ic = df_read(path_3_ic)
    df_part3_uc = df_read(path_3_uc)
    df_part3_ui = df_read(path_3_ui)
    scaler_3 =preprocessing.StandardScaler()
    batch=0
    for pre_uic in pd.read_csv(open(path_df_part3_uic,'r'),chunksize=100000):
        try:
            pred_df=pd.merge(pre_uic,df_part3_u,how='left',on=['user_id'])
            pred_df=pd.merge(pred_df,df_part3_i,how='left',on=['item_id'])
            pred_df=pd.merge(pred_df,df_part3_c,how='left',on=['item_category'])
            pred_df=pd.merge(pred_df,df_part3_ic,how='left',on=['item_id','item_category'])
            pred_df=pd.merge(pred_df,df_part3_ui,how='left',on=['user_id','item_id','item_category'])
            pred_df=pd.merge(pred_df,df_part3_uc,how='left',on=['user_id','item_category'])

            pred_x=pred_df.as_matrix(['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
                                      'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
                                      'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
                                      'u_b4_rate',
                                      'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
                                      'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
                                      'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
                                      'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
                                      'i_b4_rate',
                                      'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
                                      'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
                                      'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
                                      'c_b4_rate',
                                      'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
                                      'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
                                      'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
                                      'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
                                      'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
                                      'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
                                      'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
                                      'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
                                      'uc_b_count_rank_in_u'])
            scaler_3.partial_fit(pred_x)
            batch+=1
            print('prediction chunk %d done,'%batch)
        except StopIteration:
            print('prediction finish.')
            break

    train_x,train_y=train_set_construct(np_ratio=35,sub_ratio=1)
    LR_clf=LogisticRegression(verbose=True)
    LR_clf.fit(train_x,train_y)

    batch = 0
    for pre_uic in pd.read_csv(open(path_df_part3_uic, 'r'), chunksize=100000):
        try:
            pred_df = pd.merge(pre_uic, df_part3_u, how='left', on=['user_id'])
            pred_df = pd.merge(pred_df, df_part3_i, how='left', on=['item_id'])
            pred_df = pd.merge(pred_df, df_part3_c, how='left', on=['item_category'])
            pred_df = pd.merge(pred_df, df_part3_ic, how='left', on=['item_id', 'item_category'])
            pred_df = pd.merge(pred_df, df_part3_ui, how='left', on=['user_id', 'item_id', 'item_category'])
            pred_df = pd.merge(pred_df, df_part3_uc, how='left', on=['user_id', 'item_category'])
            pred_df.fillna(-1,inplace=True)

            pred_x = pred_df.as_matrix(
                ['u_b1_count_in_6', 'u_b2_count_in_6', 'u_b3_count_in_6', 'u_b4_count_in_6', 'u_b_count_in_6',
                 'u_b1_count_in_3', 'u_b2_count_in_3', 'u_b3_count_in_3', 'u_b4_count_in_3', 'u_b_count_in_3',
                 'u_b1_count_in_1', 'u_b2_count_in_1', 'u_b3_count_in_1', 'u_b4_count_in_1', 'u_b_count_in_1',
                 'u_b4_rate',
                 'i_u_count_in_6', 'i_u_count_in_3', 'i_u_count_in_1',
                 'i_b1_count_in_6', 'i_b2_count_in_6', 'i_b3_count_in_6', 'i_b4_count_in_6', 'i_b_count_in_6',
                 'i_b1_count_in_3', 'i_b2_count_in_3', 'i_b3_count_in_3', 'i_b4_count_in_3', 'i_b_count_in_3',
                 'i_b1_count_in_1', 'i_b2_count_in_1', 'i_b3_count_in_1', 'i_b4_count_in_1', 'i_b_count_in_1',
                 'i_b4_rate',
                 'c_b1_count_in_6', 'c_b2_count_in_6', 'c_b3_count_in_6', 'c_b4_count_in_6', 'c_b_count_in_6',
                 'c_b1_count_in_3', 'c_b2_count_in_3', 'c_b3_count_in_3', 'c_b4_count_in_3', 'c_b_count_in_3',
                 'c_b1_count_in_1', 'c_b2_count_in_1', 'c_b3_count_in_1', 'c_b4_count_in_1', 'c_b_count_in_1',
                 'c_b4_rate',
                 'ic_u_rank_in_c', 'ic_b_rank_in_c', 'ic_b4_rank_in_c',
                 'ui_b1_count_in_6', 'ui_b2_count_in_6', 'ui_b3_count_in_6', 'ui_b4_count_in_6', 'ui_b_count_in_6',
                 'ui_b1_count_in_3', 'ui_b2_count_in_3', 'ui_b3_count_in_3', 'ui_b4_count_in_3', 'ui_b_count_in_3',
                 'ui_b1_count_in_1', 'ui_b2_count_in_1', 'ui_b3_count_in_1', 'ui_b4_count_in_1', 'ui_b_count_in_1',
                 'ui_b_count_rank_in_u', 'ui_b_count_rank_in_uc',
                 'uc_b1_count_in_6', 'uc_b2_count_in_6', 'uc_b3_count_in_6', 'uc_b4_count_in_6', 'uc_b_count_in_6',
                 'uc_b1_count_in_3', 'uc_b2_count_in_3', 'uc_b3_count_in_3', 'uc_b4_count_in_3', 'uc_b_count_in_3',
                 'uc_b1_count_in_1', 'uc_b2_count_in_1', 'uc_b3_count_in_1', 'uc_b4_count_in_1', 'uc_b_count_in_1',
                 'uc_b_count_rank_in_u'])
            standardized_pred_X=scaler_3.transform(pred_x)
            pred_y=(LR_clf.predict_proba(standardized_pred_X)[:,1]>0.5).astype(int)

            pred_df['pred_label']=pred_y
            pred_df[pred_df['pred_label']==1].to_csv(path_df_result_tmp,columns=['user_id','item_id'],index=False,header=False,mode='a')
            batch += 1
            print('prediction chunk %d done,' % batch)
        except StopIteration:
            print('prediction finish.')
            break

if __name__=="__main__":
    df_part1_uic_label_cluster=df_read(path_df_part1_uic_label_cluster)
    df_part2_uic_label_cluster=df_read(path_df_part2_uic_label_cluster)

    df_part1_u=df_read(path_1_u)
    df_part1_i=df_read(path_1_i)
    df_part1_c=df_read(path_1_c)
    df_part1_ic=df_read(path_1_ic)
    df_part1_uc=df_read(path_1_uc)
    df_part1_ui=df_read(path_1_ui)

    df_part2_u=df_read(path_2_u)
    df_part2_i=df_read(path_2_i)
    df_part2_c=df_read(path_2_c)
    df_part2_ic=df_read(path_2_ic)
    df_part2_uc=df_read(path_2_uc)
    df_part2_ui=df_read(path_2_ui)



    part1_scaler=pickle.load(open(path_df_part1_scaler,'rb'))
    part2_scaler=pickle.load(open(path_df_part2_scaler,'rb'))

    # step 1 selection for the best N/P
    selNP()
    # step 2 selection for the best regularization 正则化
    selC1()
    # setp 3 selection for the best cutoff prediction
    selC2()

    train_set_part3()
    df_P = df_read(path_df_p)
    df_P_item = df_P.drop_duplicates(['item_id'])[['item_id']]
    df_pred = pd.read_csv(open(path_df_result_tmp, 'r'), index_col=False, header=None)
    df_pred.columns = ['user_id', 'item_id']

    # output result
    df_pred_P = pd.merge(df_pred, df_P_item, on=['item_id'], how='inner')[['user_id', 'item_id']]
    df_pred_P.to_csv(path_df_result, index=False)





