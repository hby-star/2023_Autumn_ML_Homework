import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

font_prop = FontProperties(size=12)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['figure.dpi'] = 300

labels = ['ridge', 'elasticnet', 'decision tree', 'random forest', 'xgboost', 'gbdt', 'lightgbm']

mse_all_train = [5.332307979513909, 5.332313601393271, 5.049975963124824, 4.814321968859235, 4.7580901491048575, 4.73779156173512, 4.750771687261698]

mse_all_test = [5.9785274330344835, 5.978555438754639, 5.690257670730443, 5.4084415858253445, 5.335830892275311, 5.304333683406015, 5.362673731712184]

rsquare_all_test = [0.17170812574020, 0.171704245702737, 0.21164630523101025, 0.2506903634505646, 0.2607501730149635, 0.265113936911763, 0.25703124623296525]

mse_selected_train = [5.995641142024539, 5.9956419194053465, 5.608856121820231, 5.363082118353599, 5.355241783672942, 5.308665210217837, 5.303406700294638]

mse_selected_test = [5.989906625003307, 5.989988428636153, 5.5513152742186715, 5.3976326179569085, 5.457553013064786, 5.395155612609416, 5.406304908421738]

rsquare_selected_test = [0.14259512363079085, 0.1425834141271045, 0.20537579559091468, 0.22737417839410834, 0.218797076583371, 0.22772874092683093, 0.22613281277702235]
# mse
x = np.arange(len(labels))
plt.figure(figsize=(20, 10))
fig, ax = plt.subplots()
bar_width = 0.20
bar_all_train = ax.bar(x - bar_width*1.5, mse_all_train, bar_width, label='mse_all_train')
bar_all_test = ax.bar(x - bar_width / 2, mse_all_test, bar_width, label='mse_all_test')
bar_selected_train = ax.bar(x + bar_width / 2, mse_selected_train, bar_width, label='mse_selected_train')
bar_selected_test = ax.bar(x + bar_width*1.5, mse_selected_test, bar_width, label='mse_selected_test')
ax.set_xlabel('Labels')
ax.set_ylabel('MSE')
ax.set_title('mse_all_train & mse_all_test & mse_selected_train & mse_selected_test')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.show()

# rsquare
x = np.arange(len(labels))
plt.figure(figsize=(20, 10))
fig, ax = plt.subplots()
bar_width = 0.35
bar_selected = ax.bar(x + bar_width / 2, rsquare_selected_test, bar_width, label='R2_selected_test')
bar_all = ax.bar(x - bar_width / 2, rsquare_all_test, bar_width, label='R2_all_test')
ax.set_xlabel('Labels')
ax.set_ylabel('Rsquare')
ax.set_title('R2_all & R2_selected')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.show()
