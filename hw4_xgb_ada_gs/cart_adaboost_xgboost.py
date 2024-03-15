# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 01:58:05 2023

@author: 21714
"""

import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import csv


from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()

warnings.filterwarnings ("ignore")
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

#规范标签
for feature in features:
    train[feature] = lbl.fit_transform(train[feature].astype(str))
    test[feature] = lbl.fit_transform(test[feature].astype(str))
train['Exited'] = lbl.fit_transform(train['Exited'].astype(str))
test['Exited'] = lbl.fit_transform(test['Exited'].astype(str))

# 训练数据
X=train[features]
Y=train['Exited']
# 测试数据
X_test=test[features].values.tolist()
y_test= [int(x) for x in test['Exited']]


print("")
seed = 42
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
dtree = DecisionTreeClassifier(criterion='gini',max_depth=3)
dtree = dtree.fit(X, Y)

result = cross_val_score(dtree, X, Y, cv=kfold)
print("CART决策树结果：")
#print('CrossValScore: ',result.mean())
y_pred=[int(x) for x in dtree.predict(test[features])]
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
y_score = dtree.predict_proba(X_test)
fpr,tpr,threshold = roc_curve(y_test, y_score[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr)
 
lw = 2
plt.plot(fpr, tpr, color='blue', alpha=0.8,
         lw=lw, label='CART ROC (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线



print("")
print("使用adaboost:")
model = AdaBoostClassifier(base_estimator=dtree, n_estimators=20,random_state=seed)
model = model.fit(X, Y)
result = cross_val_score(model, X, Y, cv=kfold)
#print("CrossValScore: ",result.mean())
y_pred=[int(x) for x in model.predict(test[features])]
# 显示准确率，召回率，f1分数
print('AccuracyScore: ' + str(accuracy_score(y_test, y_pred)))
print('RecallScore:   ' + str(recall_score(y_test, y_pred)))
print('F1_Score:      ' + str(f1_score(y_test, y_pred)))
y_score = model.predict_proba(X_test)
fpr,tpr,threshold = roc_curve(y_test, y_score[:,1])
roc_auc = auc(fpr,tpr) 
lw = 2
plt.plot(fpr, tpr, color='darkorange', alpha=0.8,
         lw=lw, label='Adaboost ROC (area = %0.2f)' % roc_auc)


print("")
print("使用xgboost:")
# 训练模型
xgb_clt = xgb.XGBClassifier(max_depth=5,learning_rate=0.1,n_estimators=100,num_class= 2,silent=True,objective='multi:softmax')

xgb_clt = xgb_clt.fit(X,Y)
y_pred = xgb_clt.predict(X_test)

# 预测结果处理
i = 0
for y in y_pred:
    if y < 0.50:
        y_pred[i]=0
    else:
        y_pred[i]=1
    i=i+1
# 显示准确率，召回率，f1分数
print('AccuracyScore: ' + str(accuracy_score(y_test, y_pred)))
print('RecallScore:   ' + str(recall_score(y_test, y_pred)))
print('F1_Score:      ' + str(f1_score(y_test, y_pred)))
y_score = xgb_clt.predict_proba(X_test)
fpr,tpr,threshold = roc_curve(y_test, y_score[:,1])
roc_auc = auc(fpr,tpr)
lw = 2
plt.plot(fpr, tpr, color='green', alpha=0.8,
         lw=lw, label='XGBoost ROC (area = %0.2f)' % roc_auc)


plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
         label='Chance', alpha=.6)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()




