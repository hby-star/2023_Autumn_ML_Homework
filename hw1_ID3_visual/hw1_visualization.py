# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 14:22:04 2023

@author: hby
"""

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import warnings; warnings.filterwarnings(action='once')



large = 22; med = 16; small = 12

params = {'axes.titlesize': large,

          'legend.fontsize': med,

          'figure.figsize': (16, 10),

          'axes.labelsize': med,

          'axes.titlesize': med,

          'xtick.labelsize': med,

          'ytick.labelsize': med,

          'figure.titlesize': large}

plt.rcParams.update(params)

plt.style.use('seaborn-whitegrid')

sns.set_style("white")


# Version

print(mpl.__version__)  #> 3.0.0

print(sns.__version__)  #> 0.9.0

sns.set(font="simhei")

"""
相关图
"""
"""
# Import Dataset
df = pd.read_csv("src1_num.csv")
# Plot
plt.figure(figsize=(12,10), dpi= 720)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)

# Decorations

plt.title('相关图', fontsize=22)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()

"""
"""
相关图
"""

"""
类型变量直方图
"""
"""
# Import Data
df = pd.read_csv("src1.csv")

# Prepare data
x_var = '文化程度'

groupby_var = '犯罪程度'

df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)

vals = [df[x_var].values.tolist() for i, df in df_agg]

# Draw
plt.figure(figsize=(16,9), dpi= 80)

colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]

n, bins, patches = plt.hist(vals, df[x_var].unique().__len__(), stacked=True, density=False, color=colors[:len(vals)])

# Decoration
plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})

plt.title(f"Stacked Histogram of 文化程度 colored by 犯罪程度", fontsize=22)

plt.xlabel(x_var)

plt.ylabel("Frequency")

plt.ylim(0, 10)

plt.xticks(ticks=bins, labels=np.unique(df[x_var]).tolist(), rotation=90, horizontalalignment='left')

plt.show()
"""
"""
类型变量直方图
"""

"""
joy plot
"""
"""
import joypy

# Import Data
mpg = pd.read_csv("src1.csv")

# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

fig, axes = joypy.joyplot(mpg, column=['犯罪记录次数','犯罪程度'], by="文化程度", ylim='own', figsize=(14,10))

# Decoration

plt.title('Joy Plot', fontsize=22)

plt.show()
"""
"""
joy plot
"""


"""
饼图
"""
"""
# Import

df_raw = pd.read_csv("src1.csv")

# Prepare Data

df = df_raw.groupby('犯罪程度').size()

# Make the plot with pandas

df.plot(kind='pie', subplots=True, figsize=(8, 8))

plt.title("犯罪程度饼状图")

plt.ylabel
"""
"""
饼图
"""


"""
树形图
"""
# pip install squarify
import squarify

# Import Data

df_raw = pd.read_csv("src1.csv")

# Prepare Data

df = df_raw.groupby('文化程度').size().reset_index(name='counts')

labels = df.apply(lambda x: str(x[0]) + "(" + str(x[1]) + ")", axis=1)

sizes = df['counts'].values.tolist()

colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

# Draw Plot

plt.figure(figsize=(12,8), dpi= 80)

squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)

# Decorate

plt.title('文化程度树形图')

plt.axis('off')

plt.show()
"""
树形图
"""


















#数据预处理
"""
df=[]
for x in data:
    match x:
        case '无':
            temp=0
        case '有':
            temp=100
        case '差':
            temp=30
        case '中':
            temp=60
        case '好':
            temp=90
        case '<20':
            temp=10
        case '20-30':
            temp=25
        case '30-40':
            temp=35
        case '>40':
            temp=50
        case '小学':
            temp=20
        case '初中':
            temp=40
        case '中专':
            temp=60
        case '高中':
            temp=80
        case '大专':
            temp=100
        case '否':
            temp=0
        case '有':
            temp=100
        case '严重':
            temp=100
        case '较轻':
            temp=0
    df.append(x)
"""



