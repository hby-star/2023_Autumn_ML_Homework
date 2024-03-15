# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:06:16 2023

@author: 21714
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from scipy import interp
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt


# 设置随机数种子
np.random.seed(0)

# 读取数据并存入dataframe表结构
dataSet = csv.reader(open('Churn-Modelling-new.csv'))
data= pd.DataFrame(dataSet,columns=dataSet.__next__())

# 将字符串列转换为数字
data['Surname'] = pd.factorize(data['Surname'])[0]
data['Geography'] = pd.factorize(data['Geography'])[0]
data['Gender'] = pd.factorize(data['Gender'])[0]

# 随机选取70%的数据作为训练集，剩余30%作为测试集
data['is_train'] = np.random.uniform(0, 1, len(data)) <= .70
train, test = data[data['is_train']==True], data[data['is_train']==False]

# 显示训练集和测试集的大小
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# 特征索引
features = data.columns[:13].tolist()
features.append('EB')

# 以features为特征，创建随机森林并拟合
clf= RandomForestClassifier(n_jobs=4,random_state=0)
clf.fit(train[features], train['Exited'])

# 获取测试数据，以及预测数据
X_test=test[features].values.tolist()
y_test= [int(x) for x in test['Exited']]
y_pred=[int(x) for x in clf.predict(test[features])]

# 显示预测表
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))

# 显示准确率，召回率，f1分数
print('AccuracyScore: ' + str(accuracy_score(y_test, y_pred)))
print('RecallScore:   ' + str(recall_score(y_test, y_pred)))
print('F1_Score:      ' + str(f1_score(y_test, y_pred)))

# 画出ROC曲线
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 9),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

skf = StratifiedKFold(n_splits=5)
linetypes = ['--',':','-.','-','-','O']

i = 0
for train, test in skf.split(X_test, y_test):
    probas_ = clf.predict_proba(np.array(X_test)[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(np.array(y_test)[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1.5,linestyle = linetypes[i], alpha=0.8,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
         label='Chance', alpha=.6)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('FPR',fontsize=20)
plt.ylabel('TPR',fontsize=20)
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()





























