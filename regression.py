#!/usr/bin/env python
# coding: utf-8


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






DATA_PATH = r"C:\Users\TW\shuiwen\shuiwenyuleifanzhi.xlsx"

def load_data(execl_path=DATA_PATH):
    return pd.read_excel(execl_path)

def try_different_method(model):
    model.fit(x_train, y_train)
    #     score = model.score(x_test, y_test)
    result = model.predict(x_test)
    score = mean_absolute_error(y_test, result)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test, 'go-', label='真实值')
    plt.plot(np.arange(len(result)), result, 'ro-', label='预测值')
    plt.title('最优结果: %f' % score)
    plt.legend()
    plt.show()
    d1 = pd.DataFrame({'真实值': y_test, '预测值': result})
    print(d1)



data = load_data(DATA_PATH)
df = pd.DataFrame(data)

#####数据处理
df = df.drop(index=[14])

df_data = df.iloc[2:-1, 0:-2]
df_target = df.iloc[2:-1, -1]

x_train, x_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.5, random_state=5)
print(x_train, x_test)

######预处理
scaler = StandardScaler().fit(x_train)  # fit生成规则
x_trainScaler = scaler.transform(x_train)  # 将规则应用于训练集
x_testScaler = scaler.transform(x_test)  # 将规则应用于测试集


######决策树
model_DecisionTreeRegressor = tree.DecisionTreeRegressor(max_depth=3)
test = []
for i in range(10):
    dtr = tree.DecisionTreeRegressor(random_state=3, max_depth=i + 1)
    dtr.fit(x_train, y_train)
    result = dtr.predict(x_test)
    score = mean_absolute_error(y_test, result)
    test.append(score);

plt.plot(range(1, 11), test, 'ro-', label='决策树的绝对误差')
plt.legend()
plt.show()

try_different_method(model_DecisionTreeRegressor)


######随机森林
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)  # 这里使用20个决策树
test = []
for i in range(30):
    dtr = ensemble.RandomForestRegressor(random_state=3, n_estimators=i + 1)
    dtr.fit(x_train, y_train)
    result = dtr.predict(x_test)
    score = mean_absolute_error(y_test, result)
    test.append(score);

import matplotlib.pyplot as plt

plt.plot(range(1, 31), test, 'ro-', label='随机森林的绝对误差')
plt.legend()
plt.show()

try_different_method(model_RandomForestRegressor)


########AdaBoost
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(random_state=3,n_estimators=48)  # 这里使用50个决策树
test = []
for i in range(41, 61):
    dtr = ensemble.RandomForestRegressor(random_state=3, n_estimators=i + 1)
    dtr.fit(x_train, y_train)
    result = dtr.predict(x_test)
    score = mean_absolute_error(y_test, result)
    test.append(score);

import matplotlib.pyplot as plt

plt.plot(range(41, 61), test, 'ro-', label='AdaBoost的绝对误差')
plt.legend()
plt.show()

try_different_method(model_AdaBoostRegressor)





