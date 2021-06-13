import pandas as pd
import numpy as np
import missingno as mg
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
air_data = pd.read_csv('data/air_data.csv',header=0,index_col=0)
air_data.head(10)

# 缺失值可视化
mg.matrix(air_data)

air_data.isnull().sum().sort_values(ascending=False)

def missing_percentage(df):
    dtypes = df.dtypes[df.isnull().sum() != 0]
    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum() != 0]
    percentage = total*100 / len(df)
    return pd.concat([total,percentage,dtypes],axis=1,keys=['Total','Percentage','Dtypes'])

missing_percentage(air_data)

# 数据类别统计分析
air_data.dtypes.value_counts()

# 连续型变量
num_columns = air_data.loc[:,air_data.dtypes != object].columns
for var in num_columns:
    fig,ax = plt.subplots(figsize=(5,5))
    sns.boxplot(air_data[var],orient='v')
    ax.set_xlabel(var)

explore = air_data.describe().T
# count代表非空值个数
explore['null'] = len(air_data) - explore['count']
# 构建缺失值个数，最大值，最小值的表格
explore = explore[['null','min','max']]
explore.columns = ['空值个数','最小值','最大值']
# 按空值个数进行排序
explore.sort_values(by='空值个数',ascending= False,inplace=True)

from datetime import datetime

# 将时间字符串转换为日期
ffp = air_data['FFP_DATE'].apply(lambda x : datetime.strptime(x,'%Y/%m/%d'))
# 提取入会年份
ffp_year = ffp.map(lambda x : x.year)
# 统计人数
ffp_year_count = ffp_year.value_counts()

# 绘制各年份入会人数趋势图
plt.rcParams['font.sans-serif'] = [u'simHei']   # 显示中文
plt.rcParams['axes.unicode_minus'] = False      # 解决负号问题

plt.figure(figsize=(10,10))
plt.hist(ffp_year,bins='auto',color='r')
plt.xlabel('年份')
plt.ylabel('入会人数')
plt.title('各年份入会人数变化趋势图')
plt.show()

gender =  air_data['GENDER']
# 统计男女比例
gender_count = gender.value_counts()
gender_count = gender_count/gender_count.sum()

plt.figure(figsize=(8,8))
color = ['yellowgreen', 'gold']
plt.pie(gender_count,labels=['男','女'],colors=color,autopct='%1.1f%%')
plt.title('入会性别比例图',fontsize=20)
plt.show()

ffp_tier = air_data['FFP_TIER']
# 会员卡级别统计
ffp_tier_count = ffp_tier.value_counts()
ffp_tier_count

plt.figure(figsize=(8,8))
plt.hist(ffp_tier,bins='auto',color='b',alpha=0.8)
plt.xlabel('会员卡级别',fontsize=15)
plt.ylabel('人数',fontsize=15)
plt.title('会员卡级别统计图',fontsize=20)
plt.show()

age = air_data['AGE'].dropna()
# 绘制箱型图
plt.figure(figsize=(10,10))
sns.boxplot(age,orient='v')
plt.xlabel('会员年龄',fontsize=15)
plt.title('会员年龄分布箱型图',fontsize=20)

flight_count = air_data['FLIGHT_COUNT']
# 统计飞行次数
flight_count.value_counts()

seg_km_sum = air_data['SEG_KM_SUM']
# 统计飞行里程数
seg_km_sum.value_counts()

fig,ax = plt.subplots(1,2,figsize=(16,8))
sns.boxplot(flight_count,orient='v',ax=ax[0])
ax[0].set_xlabel('飞行次数',fontsize=15)
ax[0].set_title('会员飞行次数分布箱型图',fontsize=20)

sns.boxplot(seg_km_sum,orient='v',ax=ax[1])
ax[1].set_xlabel('飞行里程数',fontsize=15)
ax[1].set_title('会员飞行里程数分布箱型图',fontsize=20)

sum_yr = air_data['SUM_YR_1'] + air_data['SUM_YR_2']
sum_yr.value_counts()

plt.figure(figsize=(10,10))
sns.boxplot(sum_yr,orient='v')
plt.xlabel('票价收入',fontsize=15)
plt.title('会员票价收入分布箱型图',fontsize=20)

avg_interval = air_data['AVG_INTERVAL']
avg_interval.value_counts()

plt.figure(figsize=(10,10))
sns.boxplot(avg_interval,orient='v')
plt.xlabel('平均乘机时间间隔',fontsize=15)
plt.title('会员平均乘机时间间隔分布箱型图',fontsize=20)

last_to_end = air_data['LAST_TO_END']
plt.figure(figsize=(8,8))
sns.boxplot(last_to_end,orient='v')
plt.xlabel('最后一次乘机时间至观测窗口时长',fontsize=15)
plt.title('客户最后一次乘机时间至观测窗口时长箱型图分布',fontsize=20)

exchange_count = air_data['EXCHANGE_COUNT']
# 统计积分兑换次数
exchange_count.value_counts()

plt.figure(figsize=(8,8))
plt.hist(exchange_count,bins='auto',color='b',alpha=0.8)
plt.xlabel('积分兑换次数',fontsize=15)
plt.ylabel('会员人数',fontsize=15)
plt.title('会员卡积分兑换次数分布直方图',fontsize=20)
plt.show()

point_sum= air_data['Points_Sum']
# 统计总累计积分
point_sum.value_counts()

plt.figure(figsize=(10,10))
sns.boxplot(point_sum,orient='v')
plt.xlabel('总累计积分',fontsize=15)
plt.title('会员总累计积分分布箱型图',fontsize=20)

corr = air_data.corr()
# 画热力图
fig,ax = plt.subplots(figsize=(16,16))
sns.heatmap(corr,
           annot=True,
           square=True,
           center=0,
           ax=ax)
plt.title("Heatmap of all the Features", fontsize = 30)

data_corr = air_data[['FFP_TIER','FLIGHT_COUNT','LAST_TO_END','SEG_KM_SUM','AVG_INTERVAL','EXCHANGE_COUNT','Points_Sum']]
age1 = air_data['AGE'].fillna(0)
data_corr['AGE'] = age1.astype('int64')
data_corr['ffp_year'] = ffp_year
data_corr['sum_yr'] = sum_yr.fillna(0)
data_corr

# 计算相关性矩阵
dt_corr = data_corr.corr(method='pearson')
dt_corr

fig,ax = plt.subplots(figsize=(16,16))
# 绘制热力图
sns.heatmap(dt_corr,
           annot=True,
           square= True,
            center=0,
           ax=ax)
plt.title("Heatmap of some Features", fontsize = 30)

# 删除年龄中的异常值
data = pd.read_csv('data/air_data.csv',header=0,index_col=0)
print(data.shape)
# 去除票价为空的记录
data_notnull = data.loc[data['SUM_YR_1'].notnull() & data['SUM_YR_2'].notnull(),:]

# 只保留票价非零的，或者平均折扣率不为0且总飞行公里数大于0的记录,去除年龄大于100的记录
index1 = data_notnull['SUM_YR_1'] !=0
index2 = data_notnull['SUM_YR_2'] !=0
index3 = (data_notnull['avg_discount'] !=0) & (data_notnull['SEG_KM_SUM'] >0)
index4 = data_notnull['AGE'] >100

airline = data_notnull[(index1|index2 ) & index3 & ~index4]   # 按位取反
airline.to_csv('data_cleaned.csv')

data_clean['AGE'].fillna(data_clean['AGE'].mean(),inplace = True)

data_select = data_clean[['FFP_DATE','LOAD_TIME','LAST_TO_END', 'FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]

# 构造L 单位为天数
L = pd.to_datetime(data_select['LOAD_TIME']) - pd.to_datetime(data_select['FFP_DATE'])
L = L.astype('str').str.split().str[0]   # 去掉单位

# 构造LRFMC指标
data_change = pd.concat([L,data_select.iloc[:,2:]],axis=1)
data_change.columns = ['L','R','F','M','C']

# 标准化—聚类模型基于距离
from sklearn.preprocessing import StandardScaler
data_scale = StandardScaler().fit_transform(data_change)
# 保存数据
np.savez('data_scale.npz',data_scale)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5,n_jobs=-1,random_state=1234)
# 模型训练
kmeans_fit = kmeans.fit(data_scale)
# 聚类中心
kmeans_cluster = kmeans_fit.cluster_centers_
print('聚类中心为\n',kmeans_fit.cluster_centers_)
# 聚类后样本的类别标签
kmeans_label = kmeans_fit.labels_
print('聚类后样本标签为\n',kmeans_fit.labels_)
# 聚类后各个类别数目
r1 = pd.Series(kmeans_label).value_counts()
print('聚类后各个类别数目\n',r1)
# 输出聚类分群结果
cluster_center = pd.DataFrame(kmeans_cluster,columns=['ZL','ZR','ZF','ZM','ZC'])
cluster_center.index = pd.DataFrame(kmeans_label).drop_duplicates().iloc[:,0]
cluster = pd.concat([r1,cluster_center],axis=1)
# 修改第一列列名
list_column = list(cluster.columns)
list_column[0] = '类别数目'
cluster.columns = list_column

import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 

# 客户分群雷达图
labels = ['ZL','ZR','ZF','ZM','ZC']
legen = ['客户群' + str(i + 1) for i in cluster_center.index]  # 客户群命名，作为雷达图的图例
lstype = ['-','--',(0, (3, 5, 1, 5, 1, 5)),':','-.']
kinds = list(cluster_center.index)
# 由于雷达图要保证数据闭合，因此再添加L列，并转换为 np.ndarray
cluster_center = pd.concat([cluster_center, cluster_center[['ZL']]], axis=1)
centers = np.array(cluster_center)

# 分割圆周长，并让其闭合
n = len(labels)
# endpoint=False表示一定没有stop
angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
angle = np.concatenate((angle, [angle[0]]))
# 绘图
fig = plt.figure(figsize = (8,6))
# 以极坐标的形式绘制图形
ax = fig.add_subplot(111, polar=True)  
# 画线
for i in range(len(kinds)):
    ax.plot(angle, centers[i], linestyle=lstype[i], linewidth=2,label=kinds[i])
# 添加属性标签
ax.set_thetagrids(angle * 180 / np.pi, labels)
plt.title('客户特征分析雷达图')
plt.legend(legen)
plt.show()

