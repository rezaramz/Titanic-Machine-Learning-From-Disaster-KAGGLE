import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from functions import cost, sigmoid

data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
# data2 = pd.DataFrame([['mohammadreza', 1138531], ['john', 145679], ['paul', 873219]],
#                      index=[0, 1, 2], columns=['name', 'SID'])
# print(data.head())
# print(data.tail())
# print("Column title are as follows:")
# print(data.columns)
# print('indexing numbers:')
# print(data.index)
# print('Size of the data is', data.shape)

y = data['Survived']  # y is the label
print('Columns with NaN values:', data.columns[data.isnull().any()].tolist())  # find which columns have NaN value
X = data.loc[:, ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']]  # More automatic way???
X_test = data_test.loc[:, ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']]
gender = {'male': 1, 'female': 0}
X['Sex'] = X['Sex'].map(gender)
X_test['Sex'] = X_test['Sex'].map(gender)
m, n = X.shape
# bias = np.ones([m, 1])
features = X.columns.values
mean_std = {}
for f in features:
    mn = np.mean(X[f])
    st = np.std(X[f])
    X[f] = (X[f] - mn) / st  # feature scaling
    X_test[f] = (X_test[f] - mn) / st
    mean_std[f] = [mn, st]   # dictionary with a key values of features, values of list of mean and std

y = y.values  # converting y labels into np array
x0 = np.zeros([n+1, 1])
x0 = np.reshape(x0, (n+1,))
# print(cost(x0, X, y))
res = opt.minimize(cost, x0, args=(X, y), method='BFGS')
theta = np.transpose(res.x)
# print(res)
print('**************************')
print('theta coefficients are:', theta)
# *********** algorithm learned ************

theta = np.reshape(theta, (-1, 1))
bias = np.ones((np.shape(X_test)[0], 1))
X_test = np.concatenate((bias, X_test), axis=1)

prediction = sigmoid(np.matmul(X_test, theta))
prediction = np.ravel(prediction, 1)
for it in range(len(prediction)):
    if prediction[it] >= 0.5:
        prediction[it] = 1
    else:
        prediction[it] = 0
# print(prediction)
passengerId = np.zeros(np.shape(X_test)[0])
for it in range(np.shape(X_test)[0]):
    passengerId[it] = 892 + it

res = {'PassengerId': passengerId, 'Survived': prediction}
df = pd.DataFrame(res)
df.to_csv('Result.csv', index=False)
