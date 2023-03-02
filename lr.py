import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_diabetes
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


class LrModel:
    def __init__(self):
        pass

    def prepareData(self):
        pass

    def initializeParams(self, dims):
        w = np.zeros((dims, 1))
        b = 0
        return w, b

    def linearLoss(self, X, y, w, b):
        num_train = X.shape[0]
        num_feature = X.shape[1]

        # model formula
        y_hat = np.dot(X, w) + b
        # loss function
        loss = np.sum((y_hat - y) ** 2) / num_train
        # 
        dw = np.dot(X.T, (y_hat -y )) / num_train
        db = np.sum((y_hat - y)) / num_train
        return y_hat, loss, dw, db

    def linearTrain(self, X, y, learning_rate, epochs):
        w, b = self.initializeParams(X.shape[1])
        losses = []
        for i in range(1, epochs+1):
            y_hat, loss, dw, db = self.linearLoss(X, y, w, b)
            losses.append(loss)
            # 梯度
            w += -learning_rate * dw
            b += -learning_rate * db

            # 打印迭代次数和损失
            if i % 10000 == 0:
                print(f'epoch:{i} loss: {loss}')

            # 保存参数
            params = {
                'w': w,
                'b': b
            }

            # 保存梯度
            grads = {
                'dw': dw,
                'db': db
            }

        return loss, params, grads
	
    # 根据梯度下降法更新的参数，对模型进行预测，查看在测试集上的表现
    def predict(self, X, params):
        w, b = params['w'], params['b']
        y_pred = np.dot(X, w) + b
        return y_pred

    def linearCrossValidation(self, data, k, randomize=True):
        if randomize:
            data = list(data)
            shuffle(data)
        slices = [data[i::k] for i in range(k)]
        for i in range(k):
            validation = slices[i]
            train = [
                data
                for s in slices
                if s is not validation
                for data in s
            ]
            train = np.array(train)
            validation = np.array(validation)
            yield train, validation


if __name__ == '__main__':
    lr = LrModel()
    data = pd.read_csv('C:/Users/xuziq/Desktop/correlation/exp.csv',encoding='utf-8')
    R = pd.DataFrame(data['R'])
    G = pd.DataFrame(data['G'])
    B = pd.DataFrame(data['B'])
    Output = pd.DataFrame(data['Output'])   

    i = 1
    # training sets and testing sets
    for train, validation in lr.linearCrossValidation(data, 5):
        X_train, y_train = train[:, :10], train[:, -1].reshape((-1, 1))
        X_valid, y_valid = validation[:, :10], validation[:, -1].reshape((-1, 1))

        losses_5 = []
        loss, params, grads = lr.linearTrain(X_train, y_train, 0.001, 100000)
        losses_5.append(loss)
        score = np.mean(losses_5)
        print(f'{i} of 5 kold cross validation score is {score}')
        y_pred = lr.predict(X_valid, params)
        valid_score = np.sum(((y_pred - y_valid) ** 2)) / len(X_valid)
        print('valid score is', valid_score)
        i += 1