# -*- coding:utf-8 -*- 
#Author: shenying
import os
import sys
import timeit
import pandas as pd
from  pandas import DataFrame
from dict_csv import *
import matplotlib.pyplot as plt

def test():
    dfitem=pd.read_csv(open('/home/shenying/dl/game/recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv','r'))
    dfuser=pd.read_csv(open('/home/shenying/dl/game/recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv','r'))
    # print(dfitem)
    # ef=df.set_index(['item_id'])
    # 筛选出为列为time的所有数据;
    # ef=df.loc[:,'time']
    # 筛选出 某一列中为 某值的数据;
    print(dfitem[dfitem.item_id==10003912])
    print(dfuser[dfuser.time=='2014-11-31 20'])
    # [(indexs,i) for indexs in ef.index for i in range(len(ef.loc[indexs].values))if(ef.loc[indexs].values[i].any()=='2014-12-08 18')]

    # flag=False;
    # for i in range(len(df)):
    #     if(df.loc[:,'time'].values[i=='2014-12-08 18']):
    #         print('ok')
    #         flag=True
    #         break
    # if(flag==False):
    #     print('none')



    # 把 某一列该为 index,并根据index 进行排序;
    # ef=df.set_index(['item_id'])
    # ef=ef.sort_values(axis=0,ascending=True,by='time')
    # print(ef)
    # df_1217=pd.read_csv('/home/shenying/dl/game/recomm/data/count_hour17.csv')
    # df_1218=pd.read_csv('/home/shenying/dl/game/recomm/data/count_hour18.csv')
    # # print(type(df_1217))
    # print(df_1217)
    # # df_1217.set_index('time')
    # print(df_1217.set_index('Unnamed: 0').sort_values(by='Unnamed: 0'))
    # df_1217=df_1217.set_index('Unnamed: 0').sort_values(by='Unnamed: 0')
    # df_1218=df_1218.set_index('Unnamed: 0').sort_values(by='Unnamed: 0')
    #
    # df_1718 = pd.concat([df_1217, df_1218])
    # f1 = plt.figure(1)
    # df_1718.plot(kind='bar')
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.show()
    #
    # f2 = plt.figure(2)
    # df_1718['3'].plot(kind='bar', color='r')
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.show()
# item:
# 计算操作购买转换率,浏览1 收藏2 加车3 购买4
def com_ctr():
    count_all=0
    count_4=0
    for df in pd.read_csv(open('/home/shenying/dl/game/recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv','r'),chunksize=100000):
        try:
            count_user=df['behavior_type'].value_counts()
            count_all+=count_user[1]+count_user[2]+count_user[3]+count_user[4]
            count_4+=count_user[4]
        except StopIteration:
            print("Iteration is stopped.")
            break
    ctr=count_4/count_all
    return ctr
# 可视化用户一个月(11.18-12.18)的购买记录,购买行为是否受时间影响,是否对预测当天成交有影响 from D;
def visualize_d_record():
    count_day={}
    for i in range(31):
        if i<=12:
            data='2014-11-%d'%(i+18)
        elif i < 22:
            data = '2014-12-0%d' % (i - 12)
        else:
            data = '2014-12-%d' % (i - 12)
        count_day[data]=0
    batch=0
    dataparse=lambda datas:pd.datetime.strptime(datas , '%Y-%m-%d %H')
    for df in pd.read_csv(open('/home/shenying/dl/game/recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv','r'),parse_dates=['time'],index_col=['time'],date_parser=dataparse,chunksize=100000):
        try:
            for i in range(31):
                if i<=12:
                    data = '2014-11-%d'%(i + 18)
                elif i<22:
                    data = '2014-12-0%d'%(i - 12)
                else:
                    data='2014-12-%d'%(i-12)
                # print(df[data])
                # print(df[data].shape[0])
                count_day[data]+=df[data].shape[0]
            batch+=1
            print('chunk %d done.' %batch)
        except StopIteration:
            print('finish data process')
            break
    row_dict2csv(count_day,'/home/shenying/dl/game/recomm/data/count_day.csv')
    df_count_day=pd.read_csv(open("/home/shenying/dl/game/recomm/data/count_day.csv",'r'),header=None,names=['time','count'])
    df_count_day=df_count_day.set_index('time').sort_values('time')
    df_count_day['count'].plot(kind='bar')
    plt.title('purchases counts of date')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig("data_purchases.jpg")
    plt.show()
# visiual the P set 11.18-12.18;
def visualize_p_record():
    count_day={}
    for i in range(31):
        if i<=12:
            data='2014-11-%02.d'%(i+18)
        else:
            data='2014-12-%02.d'%(i-12)
        count_day[data]=0
    batch=0
    dataparse=lambda datas:pd.datetime.strptime(datas,'%Y-%m-%d %H')
    df_p=pd.read_csv(open("/home/shenying/dl/game/recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv",'r'),index_col=False)
    for df in pd.read_csv(open("/home/shenying/dl/game/recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv",'r'),parse_dates=['time'],index_col=['time'],date_parser=dataparse,chunksize=1000000):
        try:
            df=pd.merge(df.reset_index(),df_p,on=['item_id']).set_index('time')
            for i in range(31):
                if i<=12:data='2014-11-%02.d'%(i+18)
                else:
                    data='2014-12-%02.d' %(i-12)
                count_day[data]+=df[data].shape[0]
                # print(df[data])
            batch+=1
            print('chunk %d done.' %batch)
        except StopIteration:
            print('finish data process')
            break
    row_dict2csv(count_day,"/home/shenying/dl/game/recomm/data/count_day_of_P.csv")
    df_count_day=pd.read_csv(open("/home/shenying/dl/game/recomm/data/count_day_of_P.csv",'r'),
                             header=None,names=['time','count'])
    df_count_day['time']=pd.to_datetime(df_count_day['time'])
    df_count_day=df_count_day.set_index('time').sort_values('time')

    df_count_day['count'].plot(kind='bar')
    plt.legend(loc='best')
    plt.title("behavior count of P by date")
    plt.grid(True)
    plt.savefig('data_p_purchases.jpg')
    plt.show()
# 统计两个自然日基于小时用户行为操作量
def visualize_hour_record():
    count_hour_1217={}
    count_hour_1218={}
    for i in range(24):
        time_str17='2014-12-17 %02.d'%i
        time_str18='2014-12-18 %02.d'%i
        count_hour_1217[time_str17]=[0,0,0,0]
        count_hour_1218[time_str18]=[0,0,0,0]
    batch=0
    dateparse=lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d %H')
    for df in pd.read_csv(open('/home/shenying/dl/game/recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv','r'),parse_dates=['time'],index_col=['time'],date_parser=dateparse,chunksize=500000):
        try:
            for i in range(24):
                time_str17='2014-12-17 %02.d'%i
                time_str18='2014-12-18 %02.d'%i
                temp_1217=df[time_str17]['behavior_type'].value_counts()
                temp_1218=df[time_str18]['behavior_type'].value_counts()
                for j in range(len(temp_1217)):
                    count_hour_1217[time_str17][temp_1217.index[j]-1]+=temp_1217[temp_1217.index[j]]
                for j in range(len(temp_1218)):
                    count_hour_1218[time_str18][temp_1218.index[j]-1]+=temp_1218[temp_1218.index[j]]
            batch+=1
            print('chunk %d done.' %batch)
        except StopIteration:
            print('finish data process')
            break
    # row_dict2csv(count_hour_1217,"/home/shenying/dl/game/recomm/data/count_hour17.csv")
    # row_dict2csv(count_hour_1218, "/home/shenying/dl/game/recomm/data/count_hour18.csv")
    df_1217=pd.DataFrame.from_dict(count_hour_1217,orient='index')
    df_1218=pd.DataFrame.from_dict(count_hour_1218,orient='index')
    df_1217.to_csv('/home/shenying/dl/game/recomm/data/count_hour17.csv')
    df_1217=pd.read_csv(open("/home/shenying/dl/game/recomm/data/count_hour17.csv",'r'),
                             header=None,names=['time','0-count','1-count','2-count','3-count'])
    df_1218.to_csv('/home/shenying/dl/game/recomm/data/count_hour18.csv')
    df_1218 = pd.read_csv(open("/home/shenying/dl/game/recomm/data/count_hour18.csv", 'r'),
                          header=None, names=['time', '0-count', '1-count', '2-count', '3-count'])
    # df_1217=pd.read_csv('/home/shenying/dl/game/recomm/data/count_hour17.csv')
    # df_1218=pd.read_csv('/home/shenying/dl/game/recomm/data/count_hour18.csv')
    # df_1217 = df_1217.set_index('Unnamed: 0').sort_values(by='Unnamed: 0')
    # df_1218 = df_1218.set_index('Unnamed: 0').sort_values(by='Unnamed: 0')
    df_1217['time']=pd.to_datetime(df_1217['time'])
    df_1217=df_1217.set_index('time').sort_values('time')
    print(df_1217)
    df_1218['time']=pd.to_datetime(df_1218['time'])
    df_1218=df_1218.set_index('time').sort_values('time')
    print(df_1218)

    df_1718=pd.concat([df_1217,df_1218])
    f1=plt.figure(1)
    df_1718.plot(kind='bar')
    plt.title('user behavior of two days')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('user_1718.jpg')
    plt.show()

    f2=plt.figure(2)
    df_1718['3-count'].plot(kind='bar',color='r')
    plt.title('user purchase of two days')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('purchase_1718.jpg')
    plt.show()
#10 用户信息和商品位置信息
def visualize_user_behavior():
    user_list=[10001082,
             10496835,
             107369933,
             108266048,
             10827687,
             108461135,
             110507614,
             110939584,
             111345634,
             111699844]
    user_count={}
    for i in range(10):
        user_count[user_list[i]]=[0,0,0,0]
        batch=0
    for df in pd.read_csv(open('/home/shenying/dl/game/recomm/data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv','r'),chunksize=1000000,index_col=['user_id']):
        try:
            for i in range(10):
                tmp = df[df.index == user_list[i]]['behavior_type'].value_counts()
                for j in range(len(tmp)):
                    user_count[user_list[i]][tmp.index[j]-1] += tmp[tmp.index[j]]
            batch += 1
            print('chunk %d done.' % batch)
        except StopIteration:
            print("Iteration is stopped.")
            break
    user_count = pd.DataFrame.from_dict(user_count, orient='index')
    user_count.to_csv("/home/shenying/dl/game/recomm/data/user_count.csv")
    df_user_count=pd.read_csv(open("/home/shenying/dl/game/recomm/data/user_count.csv",'r'),
                             header=None,names=['user_id','count-1','count-2','count-3','count-4'])
    df_user_count=df_user_count.set_index('user_id')
    print(df_user_count)
    df_user_count.plot(kind='bar')
    plt.legend(loc='best')
    plt.title(' ten custorm behavior ')
    plt.grid(True)
    plt.savefig('user_count.jpg')
    plt.show()

if __name__ == '__main__':
    start_time=timeit.default_timer()
    # test()
    # print(com_ctr())
    # visualize_d_record()
    # visualize_p_record()
    # visualize_hour_record()
    visualize_user_behavior()
    end_time=timeit.default_timer()
    print(('the code for file '+os.path.split(__file__)[1]+' ran %.2fm'%((end_time - start_time)/60.)),file=sys.stderr)
