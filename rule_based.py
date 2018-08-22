# -*- coding:utf-8 -*- 
#Author: shenying
#Date: 18-7-25 下午7:23

import os
import sys
import timeit
import pandas as pd
import matplotlib.pyplot as plt
# 用来测试我不知道的函数
def test():
    data={'state':[1,1,2,2],'pop':['a',None,'c','d']}
    fram=pd.DataFrame(data)
    print(fram)
    # 多重筛选
    fram=fram.drop_duplicates(['state','pop'])
    print(fram)
    fram=fram.dropna()
    print(fram)

def get_act34():
    batch=0
    for df in pd.read_csv(open("/home/shenying/dl/game/recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv","r"),chunksize=1000000):
        try:
            df_act_34=df[df['behavior_type'].isin([3,4])]
            df_act_34.to_csv('../recomm/data/act_34.csv',
                             columns=['time','user_id','item_id','behavior_type'],
                             index=False,header=False,mode='a')
            batch+=1
            print('chunk %d done.'% batch)
        except StopIteration:
            print('finish.')
            break
def pro_act34():
    date_file=open('../recomm/data/act_34.csv','r')
    # 去掉用户,商品,行为完全相同的数列:
    try:
        dateparse=lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d %H')
        df_act_34=pd.read_csv(date_file,parse_dates=[0],date_parser=dateparse,index_col=False)
        df_act_34.columns=['time','user_id','item_id','behavior_type']
        df_act_34=df_act_34.drop_duplicates(['user_id','item_id','behavior_type'])
    finally:
        date_file.close()
    # 找到行为3 和行为 4 的时间  合并 删除空
    df_time_3=df_act_34[df_act_34['behavior_type'].isin(['3'])][['user_id','item_category','time']]
    df_time_4=df_act_34[df_act_34['behavior_type'].isin(['4'])][['user_id','item_id','time']]
    df_time_3.columns=['user_id','item_id','time3']
    df_time_4.columns=['user_id','item_id','time4']
    del df_act_34
    df_time=pd.merge(df_time_3,df_time_4,how='outer',on=['user_id','item_id'])
    df_time_34=df_time.dropna()

    #只保存行为3,丢弃行为4 用来预测
    df_time_3=df_time[df_time['time4'].isnull()].drop(['time4'],axis=1)
    df_time_3=df_time_3.dropna()
    df_time_3.to_csv('../recomm/data/time_3.csv',columns=['user_id','item_id','time3'],index=False)

    # 保存 行为 3 4
    df_time_34.to_csv('../recomm/data/time_34.csv',columns=['user_id','item_id','time3','time4'],index=False)
    # 计算加车时间和购物时间之间差
    date_file=open('../recomm/data/time_34.csv','r')
    try:
         df_time_34=pd.read_csv(date_file,parse_dates=['time3','time4'],index_col=False)
    finally:
        date_file.close()
    delta_time=df_time_34['time4']-df_time_34['time3']
    delta_hour=[]
    for i in range(len(delta_time)):
        d_hour=delta_time[i].days*24+delta_time[i]._h
        if d_hour<0:
            continue
        else:
            delta_hour.append(d_hour)
     # 画出时间间隔分布
    f1=plt.figure(1)
    plt.hist(delta_hour,30)
    plt.xlabel('hours')
    plt.ylabel('count')
    plt.title('time decay for shopping trolley to buy')
    plt.grid(True)
    plt.savefig('delta.jpg')
    plt.show()
def predict():
    date_file=open('../recomm/data/time_3.csv','r')
    try:
        df_time_3=pd.read_csv(date_file,parse_dates=['time3'],index_col=['time3'])
    finally:
        date_file.close()

    ui_pred=df_time_3['2014-12-18']

    date_file=open('../recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv')
    try:
        df_item=pd.read_csv(date_file,index_col=False)
    finally:
        date_file.close()

    ui_pred_in_P=pd.merge(ui_pred,df_item,on=['item_id'])
    ui_pred_in_P.to_csv("../recomm/data/predict.csv",columns=['user_id','item_id'],index=False)
if  __name__=="__main__":
    start_time = timeit.default_timer()
    # test()
    pro_act34()
    predict()
    end_time=timeit.default_timer()
    print(('The code for file ' + os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
