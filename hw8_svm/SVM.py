from sklearn.svm import SVC
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt



plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置

print("读取训练集与测试集文件.........")
df = pd.read_csv("./data/select-data.csv")
df_test = pd.read_csv("./data/scalar-test.csv")

print("构建特征向量...........")
# 构建向量
train = []
target = []
for i in range(0, len(df["EstimatedSalary"])):
    mid = []
    mid.append(df["Geography"][i])
    mid.append(df["Gender"][i])
    mid.append(df["EB"][i])
    mid.append(df["Age"][i])
    mid.append(df["EstimatedSalary"][i])
    mid.append(df["NumOfProducts"][i])
    mid.append(df["CreditScore"][i])
    mid.append(df["Tenure"][i])
    mid.append(df["HasCrCard"][i])
    mid.append(df["IsActiveMember"][i])
    target.append(df["Exited"][i])
    train.append(mid)
train = np.array(train)
target = np.array(target)

test = []
test_target = []

for i in range(0, len(df_test["EstimatedSalary"])):
    mid = []
    mid.append(df_test["Geography"][i])
    mid.append(df_test["Gender"][i])
    mid.append(df_test["EB"][i])
    mid.append(df_test["Age"][i])
    mid.append(df_test["EstimatedSalary"][i])
    mid.append(df_test["NumOfProducts"][i])
    mid.append(df_test["CreditScore"][i])
    mid.append(df_test["Tenure"][i])
    mid.append(df_test["HasCrCard"][i])
    mid.append(df_test["IsActiveMember"][i])
    test_target.append(df_test["Exited"][i])
    test.append(mid)
test = np.array(test)

print("构建完成..............................")
train, target = shuffle(train, target)
print("构建K折交叉验证............................")
index = []
value = []
i = 1
kf_size = 10
kf = KFold(n_splits=kf_size)
print("开始SVM训练...........")
import time
dt_start = time.time()

for train_index, test_index in kf.split(train):
    # print('train_index', train_index, 'test_index', test_index)
    trainx = train[train_index]
    trainy = target[train_index]
    testx = train[test_index]
    testy = target[test_index]    
    svc = SVC(kernel='linear', C=0.1)
    clf = svc.fit(trainx, trainy)
    sc = svc.score(test, test_target)   
    print('%.7f' % sc)
    index.append(i)
    i = i + 1
    value.append(sc)
print('time1: ', time.time() - dt_start)

avg = round(sum(value) / kf_size, 4) * 100
maxV = round(max(value), 4) * 100
minV = round(min(value), 4) * 100

plt.title('K折交叉验证(k=' + str(kf_size) + ') avg=' + str(avg) + "% max:" + str(maxV) + "% min:" + str(minV) + "%")
plt.xlabel('次数')
plt.ylabel('正确率acc')
plt.ylim([0.6, 0.9])
plt.xlim([1, kf_size])
plt.plot(index, value)
plt.show()
# train = np.trunc(train * 100)

