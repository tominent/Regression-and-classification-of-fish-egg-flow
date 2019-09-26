import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn import tree
from sklearn import ensemble
def load_data(execl_path):
    return pd.read_excel(execl_path)

# data_path = r"C:\Users\TW\shuiwen\spawn.xlsx"
#
# data_two1 = load_data(data_path)
# spawn = pd.DataFrame(data_two1)
#
# spawn.plot(x='shuiwen', y='liuliang')
#
# no_spawn = pd.DataFrame(load_data(r"C:\Users\TW\shuiwen\10.1to11.30.xlsx"))
#
# no_spawn.plot(x='shuiwen', y='liuliang')


####找到数据
data_all = pd.DataFrame(load_data(r"C:\Users\TW\shuiwen\all.xlsx"))

x = data_all.iloc[:, 0:-1]
y = data_all.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=3)
scaler = StandardScaler().fit(x_train)  # fit生成规则
x_trainScaler = scaler.transform(x_train)  # 将规则应用于训练集
x_testScaler = scaler.transform(x_test)  # 将规则应用于测试集

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# classifier = SVC(kernel = "linear", probability = True)
classifier = RandomForestClassifier(n_estimators=8)
classifier.fit(x_train, y_train)
# predict the result
y_pred = classifier.predict(x_test)
print(y_pred)
# 查看每种类别的概率
y_pred_proba = classifier.predict_proba(x_test)
print(y_pred_proba)


score = classifier.score(x_test, y_test)

print(score)

