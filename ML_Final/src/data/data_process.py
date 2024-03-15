import pandas as pd
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns

font_prop = FontProperties(size=12)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['figure.dpi'] = 300

# """
# (1) 初步特征选择
# """
# df = pd.read_excel('超市.xls')
# columns_to_drop = ['订单 Id', '发货日期', '客户名称', '细分', '地区', '城市', '省/自治区', '订单日期', '记录数', '国家']
# df = df.drop(columns=columns_to_drop, axis=1)
# print('初步特征选择后的数据:')
# print(df.head(5))
# df.to_excel('processed.xlsx', index=False)
#
# """
# (2) 处理数据缺失
# """
# # 找到包含空白值的列
# df = pd.read_excel('processed.xlsx')
# print('找到包含空白值的列:')
# print(df.columns[df.isnull().any()])
# # 填充空值
# df['利润'] = df['利润'].replace('￥', '', regex=True).replace(',', '.', regex=True).astype(float)
# df['销售额'] = df['销售额'].replace('￥', '', regex=True).replace(',', '.', regex=True).astype(float)
# df['利润'].fillna(df['销售额'] * df['利润率'], inplace=True)
# print('处理数据缺失后的数据:')
# print(df.columns[df.isnull().any()])
# print(df.head(5))
#
# """
# (3) 特征拆分
# """
# print('特征拆分前:')
# print(df['产品名称'].head(5))
# df[['产品', '标签']] = df['产品名称'].str.split(',', expand=True)
# df['产品'] = df['产品'].str.split(' ').str[-1]
# df = df.drop(['产品名称'], axis=1)
# print('特征拆分后:')
# print(df.head(5))
#
# """
# (4) 特征转换
# """
# # 绘制销售额,利润,数量热图
# selected_columns = ['销售额', '利润', '数量']
# df_selected = df[selected_columns]
# correlation_matrix = df_selected.corr()
# plt.figure(figsize=(10, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, square=True)
# plt.title('销售额,利润,数量热图')
# plt.show()
# # 转换销售额，利润
# print('特征转换前:')
# print(df.head(5))
# df = df.rename(columns={'销售额': '单个售价', '利润': '单个利润'})
# df['单个售价'] = df['单个售价'] / df['数量']
# df['单个利润'] = df['单个利润'] / df['数量']
# print('特征转换后:')
# print(df.head(5))
# # 绘制销售额,利润,数量热图
# selected_columns = ['单个售价', '单个利润', '数量']
# df_selected = df[selected_columns]
# correlation_matrix = df_selected.corr()
# plt.figure(figsize=(10, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, square=True)
# plt.title('单个售价,单个利润,数量热图')
# plt.show()
# # 转换数量
# print('特征转换前:')
# print(df.head(5))
# df['销量'] = \
#     df.groupby(['利润率', '制造商', '产品', '标签', '单个利润', '子类别', '折扣', '类别', '邮寄方式', '单个售价'])[
#         '数量'].transform('sum')
# df = df.drop('数量', axis=1)
# print('特征转换后:')
# print(len(df))
# df = df.drop_duplicates()
# print(len(df))
# print(df.head(5))
# df.to_excel('processed.xlsx', index=False)
# """
# (5) 处理噪点
# """
# # 绘制数值变量的箱型图
# df = pd.read_excel('processed.xlsx')
# columns = ['利润率', '单个利润', '折扣', '单个售价', '销量']
# plt.figure(figsize=(15, 10))
# for i, column in enumerate(columns, 1):
#     plt.subplot(1, len(columns), i)
#     plt.boxplot(df[column])
#     plt.title(column, fontproperties=font_prop)
# plt.tight_layout()
# plt.show()
# # 画饼状图，找出折扣比例。
# value_counts = df['折扣'].value_counts()
# labels = value_counts.index
# sizes = value_counts.values
# plt.figure(figsize=(12, 12))
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
# plt.title('折扣饼状图')
# plt.show()
# # 绘制销量柱状图
# df = pd.read_excel('processed.xlsx')
# value_counts = df['销量'].value_counts()
# plt.bar(value_counts.index, value_counts.values, color='skyblue')
# plt.xlabel('销量')
# plt.ylabel('频率')
# plt.title('销量频率柱状图')
# plt.show()
# # 清除噪点
# print('清除噪点前:')
# print(len(df))
# df = pd.read_excel('processed.xlsx')
# df.drop(df[df['折扣'] == 0.8].index, inplace=True)
# df.drop(df[df['销量'] > 14].index, inplace=True)
# print('清除噪点后:')
# print(len(df))
# df.to_excel('processed.xlsx', index=False)
# # 画箱型图
# columns = ['利润率', '单个利润', '折扣', '单个售价', '销量']
# plt.figure(figsize=(15, 8))
# for i, column in enumerate(columns, 1):
#     plt.subplot(1, len(columns), i)
#     plt.boxplot(df[column])
#     plt.title(column, fontproperties=font_prop)
# plt.tight_layout()
# plt.show()
# # 绘制销量柱状图
# df = pd.read_excel('processed.xlsx')
# value_counts = df['销量'].value_counts()
# plt.bar(value_counts.index, value_counts.values, color='skyblue')
# plt.xlabel('销量')
# plt.ylabel('频率')
# plt.title('销量频率柱状图')
# plt.show()
#
# """
# (6) 数值化
# """
# # 观察字符串种类数
# df = pd.read_excel('processed.xlsx')
# print('制造商种类：' + str(df['制造商'].nunique()))
# print('子类别种类：' + str(df['子类别'].nunique()))
# print('类别种类：' + str(df['类别'].nunique()))
# print('邮寄方式种类：' + str(df['邮寄方式'].nunique()))
# print('产品种类：' + str(df['产品'].nunique()))
# print('标签种类：' + str(df['标签'].nunique()))
#
# # 独热编码
# print('数值化前:')
# print(df.head(5))
# df = pd.get_dummies(df, columns=['类别', '邮寄方式'])
# # 目标编码
# encoder = TargetEncoder()
# columns_to_encode = ['制造商', '子类别', '产品', '标签']
# for column in columns_to_encode:
#     df[column] = encoder.fit_transform(df[column], df['销量'])
# print('数值化后:')
# print(df.head(5))
# df.to_excel('processed_encode.xlsx', index=False)
# df.to_excel('feature_all.xlsx', index=False)

"""
(7) 进一步特征探索
"""
df = pd.read_excel('feature_all.xlsx')
# 绘制 all 热图
correlation_matrix = df.corr()
plt.figure(figsize=(15, 15))  # 设置图的大小
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('总体热图')
plt.show()
# 绘制 销量 热图
correlation_matrix = df.drop('销量', axis=1).corrwith(df['销量'])
correlation_df = pd.DataFrame(correlation_matrix, index=df.columns, columns=['销量']).drop(index=['销量'])
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('销量热图')
plt.show()

# 特征选择并产生新数据
columns_to_drop = ['利润率', '子类别', '折扣', '标签', '类别_办公用品', '类别_家具', '类别_技术', '邮寄方式_一级',
                   '邮寄方式_二级',
                   '邮寄方式_当日']
df = df.drop(columns=columns_to_drop, axis=1)
df['销量'] = \
    df.groupby(['制造商', '产品', '单个利润', '单个售价', '邮寄方式_标准级'])[
        '销量'].transform('sum')
df = df.drop_duplicates()
# 绘制销量柱状图
value_counts = df['销量'].value_counts()
plt.bar(value_counts.index, value_counts.values, color='skyblue')
plt.xlabel('销量')
plt.ylabel('频率')
plt.title('销量频率柱状图')
plt.show()
df.drop(df[df['销量'] > 14].index, inplace=True)
print('经过筛选的特征:')
print(df.head(5))
df.to_excel('feature_selected.xlsx', index=False)
