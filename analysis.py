#!/usr/bin/python
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib
from pylab import mpl
import matplotlib.pyplot as plt
import sys

from matplotlib.font_manager import _rebuild
_rebuild()

#防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

train_data = pd.read_csv('/Users/liucong/Desktop/node/Data-anlysis/Lianjia/data/Lianjia.csv')
#df = train_data.decode('utf-8','ignore')
print(train_data.info())
print('''
''')
print(train_data.describe())
# # #户型统计
# x = np.arange(44)                                                #设置x轴上柱子的个数，此数据集中户型有44种
# y = np.array(list(train_data.Structure.value_counts()))             #设置y轴的数值，需将户型列的数据先转换成数列，再转换成矩阵格式
# train_data.Structure.value_counts().plot(kind='bar')
# plt.xlabel(u"户型")
# plt.ylabel(u"数量")
# plt.title(u"户型统计")
# #设置柱状图上端的数字标签显示
# for a,b in zip(x,y):
#     plt.text(a, b, '%.0f' % b ,ha='center', va = 'bottom', fontsize = 7)
#
# # #区域统计
# x = np.arange(17)
# y = np.array(list(train_data.Station.value_counts()))
# train_data.Station.value_counts().plot(kind = 'bar')
# plt.xlabel(u"区域名称")
# plt.ylabel(u"数量")
# plt.title(u"区域统计")
# for a,b in zip(x,y):
#     plt.text(a, b, '%.0f' % b, ha='center',va='bottom',fontsize = 7)
#
# #房屋朝向
# x = np.arange(21)
# y = np.array(list(train_data.Orientation.value_counts()))
# train_data.Orientation.value_counts().plot(kind='bar')
# plt.title(u"房屋朝向")
# plt.xlabel(u"朝向")
# plt.ylabel(u"数量")
# for a,b in zip(x,y):
#     plt.text(a,b,'%.0f' % b, ha='center', va='bottom', fontsize=8)
#
# #房屋朝向——横向柱状图
# y = np.arange(21)
# x = list(train_data.Orientation.value_counts())
# train_data.Orientation.value_counts().plot(kind='barh')
# plt.xlabel(u"数量")
# plt.ylabel(u"朝向")
# for b,a in zip(y,x):
#
#     plt.text( a, b,'%.0f'%a, fontsize=10)

#房源面积

# bins = [0,50,100,150,200,250,300,350,5000]
# group_mianji = ['小于50','50-100','100-150','150-200','200-250','250-300','300-350','大于350']
# train_data.Area = pd.cut(train_data.Area,bins,labels=group_mianji)
# #按照房源面积分组对房源数量进行汇总
# group_mianji = train_data.groupby('Area')['Area'].agg(len)
# plt.figure(figsize=(12,8))
# plt.rc('font',size = 15)
# a = np.array([1,2,3,4,5,6,7,8])
# plt.barh([1,2,3,4,5,6,7,8],group_mianji,color = '#99CC01',alpha=0.8, align='center', edgecolor = 'white')
# plt.ylabel(u"面积分组")
# plt.xlabel(u"数量")
# plt.title(u"房源面积分布")
# plt.legend(['数量'],loc='upper right')
# plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.4)
# plt.yticks(a,('小于50','50-100','100-150','150-200','200-250','250-300','300-350','大于350'))
# y = np.arange(8)+1
# x = list(train_data.groupby('Area')['Area'].value_counts())
# for b,a in zip(y,x):
#     plt.text(a,b,'%.0f'%a,ha='center',va='bottom',fontsize=8)

#价格的散点图
# fig = plt.figure()
# fig.set(alpha=0.65)
# plt.title(u"面积与价格的散点图")
# x = list(train_data['Area'])
# y = list(train_data['Price'])
# # size = np.random.randint(0,100,1000)
# plt.scatter(x,y)
# plt.xlabel(u"面积")
# plt.ylabel(u"价格")

#上海各区域的二手房的价格对比
#所有区域安单价排名
dist = train_data.groupby('Station').mean()[['Price_sqm','Price','Area']]
dist = dist.sort_values(by='Price_sqm',ascending=False)                      #ascending=False表示降序排序
print(dist)

#所有小区按单价排序
rsd = train_data.groupby(['Name']).mean()[['Price_sqm','Price','Area']]
rsd_top10 = rsd.sort_values(by='Price_sqm',ascending=False).head(10)
rsd_tail10 = rsd.sort_values(by='Price_sqm',ascending=False).tail(10)
print("单价最高的前10个小区")
print(rsd_top10)
print("单价最低的10个小区")
print(rsd_tail10)

#不同年份与平均单价的联系
#year = train_data.pivot_table(index='建筑时间',values='单价（平方米）').sort_values(by='单价（平方米）',ascending=False)       #pivot_table为透视表
year = train_data.groupby(['Build_time']).mean()[['Price_sqm','Area']]
year_top10 = year.sort_values(by='Price_sqm',ascending=False).head(10)
year_tail10 = year.sort_values(by='Price_sqm',ascending=False).tail(10)
print("单价最贵的房子修建时间")
print(year_top10)
print("单价最低的房子的修建时间")
print(year_tail10)

#对房源的修建时间进行处理,去掉其中的中文字符
buildtime = train_data.drop('Build_time',axis=1).join(train_data['Build_time'].astype(str).replace("年建","").rename('Build_time'))
buildtime.to_csv('/Users/liucong/Desktop/node/Data-anlysis/Lianjia/data/Lianjia.csv',index=False)
buildtime['Build_time'] = buildtime['Build_time'].apply(lambda x:x.strip())
#print(buildtime['Build_time'])

# #上海二手房平均年份和面积的散点图
# station_price = train_data.groupby('Name').mean()[['Price_sqm','Price','Area','Build_time']]
# station_sct = station_price.plot.scatter(x='Build_time',y='Area')
# plt.title(u"上海二手房平均年份和面积的散点图")

#根据所属地区计算小于50平米的二手房的比例
ttl = train_data.Station.value_counts().rename('ttl')
lt = train_data[train_data.Area <= 50].Station.value_counts().rename('area<50')
new = pd.concat([ttl,lt],axis=1,sort=False)
new['prop'] = new['area<50']/new['ttl']
print(new.sort_values(by='area<50',ascending=False))

#房源单价与时间的散点图
# fig = plt.figure()
# fig.set(alpha = 0.65)
# plt.title(u"房源单价与时间的散点图")
# x = list(train_data['Build_time'])
# y = list(train_data['Price_sqm'])
# size = np.random.randint(0,100,1000)
# plt.scatter(x,y)
# plt.xlabel(u"修建时间（年）")
# plt.ylabel(u"单价(万元/平米)")

# #填补空缺值
# from sklearn.ensemble import RandomForestRegressor
# def set_missing_buildtime1(df):
#     bulidtime1 = df[['Area','Price','Price_sqm','Build_time']]
#     #房源的分成已知、未知修建时间两种
#     known_buildtime = bulidtime1[bulidtime1.Build_time.notnull()].as_matrix()
#     unknown_buildtime = bulidtime1[bulidtime1.Build_time.isnull()].as_matrix()
#
#     y = known_buildtime[:,0]
#     X = known_buildtime[:,1:]
#
#     rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
#     rfr.fit(X,y)
#
#     predictedBuildtimes = rfr.predict(unknown_buildtime[:,1::])
#     train_data.loc[(train_data.Build_time.isnull()),'Build_time'] = predictedBuildtimes
#     return df,rfr
# train_data,rfr = set_missing_buildtime1(train_data)

#房源单价与时间的散点图
fig = plt.figure()
fig.set(alpha = 0.65)
plt.title(u"房源房价与面积的分布图")
y = list(train_data['Price'])
x = list(train_data['Area'])
size = np.random.randint(0,100,1000)
plt.scatter(x,y)
plt.xlabel(u"面积（平米）")
plt.ylabel(u"价格（万元）")





plt.show()