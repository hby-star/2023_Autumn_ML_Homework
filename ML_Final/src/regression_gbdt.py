# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:49:01 2023

@author: 21714
"""
import sys
from datetime import datetime

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 01:30:18 2023

@author: 21714
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

font_prop = FontProperties(size=12)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['figure.dpi'] = 300

# 定义超参数的搜索范围
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 4, 5, 6, 7, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20, 30, 40],
}

# 重定向
original_stdout = sys.stdout
new_stdout = open('result.txt', 'a')
sys.stdout = new_stdout
print("==============================GBDT Regression=================================")
current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("Time:", formatted_time)

df_all = pd.read_excel('feature_all.xlsx')
X_all = df_all.drop(['销量'], axis=1)
y_all = df_all['销量']
X_all, y_all = np.array(X_all), np.array(y_all)

df_selected = pd.read_excel('feature_selected.xlsx')
X_selected = df_selected.drop(['销量'], axis=1)
y_selected = df_selected['销量']
X_selected, y_selected = np.array(X_selected), np.array(y_selected)

# all
print()
print("===================ALL====================")
# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
# 创建 GBDT 回归模型
model = GradientBoostingRegressor()
# 创建网格搜索对象
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

# 在训练集上进行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳超参数和对应的模型性能
print("最佳超参数:", grid_search.best_params_)
print("最佳模型性能 (负均方误差):", grid_search.best_score_)

# 在测试集上评估最佳模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("在测试集上的均方误差:", mse)
r2 = r2_score(y_test, y_pred)
print("在测试集上的R²值:", r2)
print("===================ALL====================")

# selected
print()
print("================SELECTED==================")
# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)
# 创建 GBDT 回归模型
model = GradientBoostingRegressor()
# 创建网格搜索对象
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

# 在训练集上进行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳超参数和对应的模型性能
print("最佳超参数:", grid_search.best_params_)
print("最佳模型性能 (负均方误差):", grid_search.best_score_)

# 在测试集上评估最佳模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("在测试集上的均方误差:", mse)
r2 = r2_score(y_test, y_pred)
print("在测试集上的R²值:", r2)
print("================SELECTED==================")

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("Time:", formatted_time)

print("==============================GBDT Regression=================================")
print()
print()
print()
# 恢复重定向
sys.stdout = original_stdout
new_stdout.close()
