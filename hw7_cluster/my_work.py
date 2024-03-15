import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.model_selection import ParameterGrid

# Silhouette Score： 轮廓系数是一种衡量聚类效果的指标，其取值范围在[-1, 1]之间。轮廓系数越接近1表示样本越适合于自己的簇，越接近-1表示样本更适合于与相邻簇。
from sklearn.metrics import silhouette_score
# Calinski-Harabasz指数： 这个指数考虑了簇内的稠密程度和簇间的分离度。指数值越大越好，表示簇内方差小，簇间方差大。
from sklearn.metrics import calinski_harabasz_score
# Davies-Bouldin指数： 衡量簇的紧密度和分离度，数值越低表示簇内越紧密、簇间越分离。
from sklearn.metrics import davies_bouldin_score

# 读取数据并存入dataframe表结构
dataSet = csv.reader(open('data.csv'))
data= pd.DataFrame(dataSet,columns=dataSet.__next__())
data= pd.DataFrame(data).drop(columns=['RowNumber','Geography'])

# 数据标准化
scaler = StandardScaler()
data = scaler.fit_transform(data)

"""
DBSCAN start
"""
# 训练模型
print("------------------------------------------------")
print("DBSCAN:")
param_grid = {
    'eps': [1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3],
    'min_samples': [2, 3, 4, 5, 6, 10, 15, 20]
}
grid = ParameterGrid(param_grid)

best_score = -1
best_params = None
best_label = None

for params in grid:
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'], n_jobs=-1)
    labels = dbscan.fit_predict(data)
    
    # 由于可能返回标签中的噪声点（-1），在计算轮廓系数时要排除这些点
    if len(set(labels)) > 1:
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_params = params
            best_label = labels
            
print("Best Parameters: (based on silhouette score)", best_params)
# 评价模型
s_score = silhouette_score(data, best_label)
c_score = calinski_harabasz_score(data, best_label)
d_score = davies_bouldin_score(data, best_label)
print("All score:")
print('silhouette_score:         '+str(s_score))
print('calinski_harabasz_score:  '+str(c_score))
print('davies_bouldin_score:     '+str(d_score))
print("------------------------------------------------")
"""
DBSCAN end
"""

"""
K-means start
"""
print("------------------------------------------------")
print("k-means:")
param_grid = {
    'n_clusters': range(2, 12),
}
grid = ParameterGrid(param_grid)

distortions = []
best_score = -1
best_params = None
best_label = None

for params in grid:
    kmeanModel = KMeans(n_clusters=params['n_clusters']).fit(data)
    labels = kmeanModel.fit_predict(data)
    distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
    
    # 由于可能返回标签中的噪声点（-1），在计算轮廓系数时要排除这些点
    if len(set(labels)) > 1:
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_params = params
            best_label = labels
            
print("Best Parameters: (based on silhouette score)", best_params)
# 评价模型
s_score = silhouette_score(data, best_label)
c_score = calinski_harabasz_score(data, best_label)
d_score = davies_bouldin_score(data, best_label)
print("All score:")
print('silhouette_score:         '+str(s_score))
print('calinski_harabasz_score:  '+str(c_score))
print('davies_bouldin_score:     '+str(d_score))
print("------------------------------------------------")

# sse
plt.plot(range(2,12), distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
"""
K-means end
"""