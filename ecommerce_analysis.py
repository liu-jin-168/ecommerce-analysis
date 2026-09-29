import pandas as pd
import sqlite3

df= pd.read_csv('UserBehavior.csv')
print(df.head())
print(df.shape)
print(df.isnull().sum())

conn = sqlite3.connect('ecommerce.db')
df.to_sql('user_behavior',conn,index=False,if_exists='replace')
conn.close()
print("导入完成")

conn = sqlite3.connect('ecommerce.db')
sql = '''
SELECT
   user_id,
   COUNT(*) AS total_actions,
   SUM(CASE WHEN behavior_type='buy' THEN 1 ELSE 0 END ) AS buy_count,
   SUM(CASE WHEN behavior_type='cart' THEN 1 ELSE 0 END ) AS cart_count,
   SUM(CASE WHEN behavior_type='fav' THEN 1 ELSE 0 END ) AS fav_count,
   COUNT(DISTINCT item_category ) AS total_items,
   MIN(timestamp) AS min_timestamp,
   MAX(timestamp) AS max_timestamp
FROM user_behavior
GROUP BY user_id
'''
user_summary = pd.read_sql(sql,conn)
conn.close()

print(user_summary.head())
print(user_summary.shape)

user_summary['first_time'] = pd.to_datetime(user_summary['min_timestamp'], unit='s')
user_summary['last_time'] = pd.to_datetime(user_summary['max_timestamp'], unit='s')

print(user_summary[['user_id', 'first_time', 'last_time']].head())
user_summary['active_days'] = (user_summary['last_time']-user_summary['first_time']).dt.days
user_summary = user_summary[user_summary['active_days']>7]
print(user_summary.shape)

user_summary = user_summary[user_summary['last_time']<pd.Timestamp('2017-12-04')]
print(user_summary.head())
end_time = user_summary['last_time'].max()
end_data = pd.to_datetime(end_time,unit='s')
print('截止时间：',end_data)
print('删除之后：',user_summary.shape)

user_summary['day_since_last'] = (pd.Timestamp('2017-12-03')-user_summary['last_time']).dt.days
user_summary['churn'] = (user_summary['day_since_last']>5).astype(int)
print(user_summary['churn'].value_counts())
print(user_summary['churn'].value_counts(normalize=True))

def user_segment(row):
    if row['buy_count'] > 0:
        return '购买用户'
    elif row['cart_count'] > 0 or row['fav_count'] > 0:
        return '意向用户'
    else:
        return '仅浏览用户'

user_summary['segment'] = user_summary.apply(user_segment, axis=1)
print(user_summary['segment'].value_counts())
print(user_summary['segment'].value_counts(normalize=True))

print(user_summary.groupby('segment')[['total_actions','total_items']].mean())

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10,5))
user_summary['segment'].value_counts().plot(kind='bar', color=['green', 'orange', 'red'])
plt.title('用户分层分布')
plt.xlabel('用户类型')
plt.ylabel('人数')
plt.xticks(rotation=45)
plt.show()

import pandas as pd
import sqlite3
conn = sqlite3.connect('ecommerce.db')
sql = '''
SELECT
  item_category,
  SUM(CASE WHEN behavior_type='buy' THEN 1 ELSE 0 END ) AS buy_count,
  SUM(CASE WHEN behavior_type='pv' THEN 1 ELSE 0 END) AS pv_count
FROM user_behavior
GROUP BY item_category
ORDER BY pv_count DESC
'''

category_stats =pd.read_sql(sql,conn)
conn.close()

print(category_stats.head(10))
print(category_stats.shape)

category_stats = category_stats[category_stats['pv_count']>0]
category_stats['conversion_rate'] = category_stats['buy_count']/category_stats['pv_count']
print(category_stats.sort_values('conversion_rate').head(10))
print(category_stats.sort_values('conversion_rate',ascending=False).head(10))

import pandas as pd
import sqlite3
conn = sqlite3.connect('ecommerce.db')
sql='''
SELECT item_id, item_category
FROM user_behavior
WHERE item_category IN (1712001,1404020,3203956)
LIMIT 20
'''
print(pd.read_sql(sql,conn))

conn=sqlite3.connect('ecommerce.db')
sql = '''
SELECT 
  item_id,
  SUM(CASE WHEN behavior_type='buy' THEN 1 ELSE 0 END) AS buy_count,
  SUM(CASE WHEN behavior_type='pv' THEN 1 ELSE 0 END) AS pv_count
FROM user_behavior
WHERE item_category IN (1712001,1404020,3203956)
GROUP BY item_id
HAVING pv_count > 0
ORDER BY buy_count DESC 
'''
item_stats = pd.read_sql(sql,conn)
conn.close()
item_stats['conversion_rate'] = item_stats['buy_count']/item_stats['pv_count']
print(item_stats.sort_values('conversion_rate',ascending=False).head(10))

def user_value(row):
    if row['buy_count'] >= 5:
        return '高价值用户'
    elif row['buy_count'] >= 2:
        return '中价值用户'
    elif row['buy_count'] == 1:
        return '低价值用户'
    else:
        return '未购买用户'

user_summary['value_level'] = user_summary.apply(user_value, axis=1)

print(user_summary['value_level'].value_counts())
print(user_summary['value_level'].value_counts(normalize=True))

import numpy as np
import pandas as pd
from scipy import stats

# 1. 模拟数据：A 组（旧版）1000 人，B 组（新版）1000 人
np.random.seed(42)

# A 组转化率 10%，B 组转化率 13%
a_converted = np.random.binomial(1, 0.10, 1000)
b_converted = np.random.binomial(1, 0.13, 1000)

# 2. 算转化率
a_rate = a_converted.mean()
b_rate = b_converted.mean()

print(f"A 组转化率: {a_rate:.2%}")
print(f"B 组转化率: {b_rate:.2%}")

# 3. 卡方检验
contingency = [[a_converted.sum(), 1000 - a_converted.sum()],
               [b_converted.sum(), 1000 - b_converted.sum()]]

chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"p 值: {p_value:.4f}")

# 4. 判断
if p_value < 0.05:
    print("结论：B 组显著优于 A 组，建议上线新版。")
else:
    print("结论：两组差异不显著，不能确定新版有效。")
