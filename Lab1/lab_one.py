import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, X, y, learningrate, tolerance, maxIteration=50000, error='rmse', gd=True):
        self.X = X
        self.y = y
        self.learningrate = learningrate
        self.tolerance = tolerance
        self.maxIteration = maxIteration
        self.error = error
        self.gd = gd

    def splitToTrainTest(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state = 0)
        return X_train, X_test, y_train, y_test

    def add_x0(self, X):
        return np.column_stack([np.ones([X.shape[0], 1]), X])

    def normalizeTrain(self, X):
        mean = np.mean(X, 0) # get mean of each column
        std = np.std(X, 0) # get std of each column
        X_norm = (X - mean) / std
        X_norm = self.add_x0(X_norm)
        return X_norm, mean, std

    def normalizeTest(self, X, train_mean, train_std):
        X_norm = (X - train_mean) / train_std
        X_norm = self.add_x0(X_norm)
        return X_norm

    def normal_equation(self, X, y):
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return w

    def gradient_descent(self, X, y):
        iter_count = 0
        loss_history = []
        loss = float('inf') # first loss
        while iter_count < self.maxIteration:
            y_hat = self.predict(X) # y prediction
            error = y_hat - y # deviation between predicted and actual value
            # grad = (1.0/y.size) * l(fw(xi),y) # gradient
            grad = error.dot(X)
            self.w = self.w - self.learningrate*grad # update weights
            loss_new = self.rmse(X,y) # new loss
            loss_history.append(loss_new)
            if loss-loss_new < self.tolerance:
                print("The model stopped - no further improvment")
                break
            loss = loss_new
            iter_count += 1
        print('Iteration times:', iter_count)
        return self.w, loss_history, iter_count, grad

    def predict(self, X):
        return X.dot(self.w)

    def sse(self, X, y):
        y_hat = self.predict(X)
        return ((y_hat - y) ** 2).sum() # ||X^TQ-Y||2

    def rmse(self, X, y):
        return math.sqrt(self.sse(X, y) / y.size)

    def run_model(self, colour='white'):

        #Split training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = self.splitToTrainTest()

        #Normalization
        self.X_train, self.mean, self.std = self.normalizeTrain(self.X_train)
        self.X_test = self.normalizeTest(self.X_test, self.mean, self.std)
        grad = 0

        if self.gd == False:
            print('-----------Solved using Normal equation-------------')
            self.w = self.normal_equation(self.X_train, self.y_train)

        else:
            print('-----------Solved using gradient descent------------')
            #Initiate the w all as zeros
            self.w = np.zeros(self.X_train.shape[1])
            print('Initiated w')
            print(self.w)

            #Improvement plot for gradient descent
            w,loss_hist,iter_count,grad = self.gradient_descent(self.X_train, self.y_train)
            plt.plot(range(iter_count+1), loss_hist, color = colour)
            plt.rcParams["figure.figsize"] = (10,6)
            plt.grid()
            plt.xlabel("Number of iterations")
            plt.ylabel("cost ")
            plt.title("Convergence of gradient descent")
            # plt.show()

        print('Weight Vector')
        print(self.w)

        error_train = self.rmse(self.X_train, self.y_train)
        error_test = self.rmse(self.X_test, self.y_test)

        print('Gradients for training data:')
        print(grad)

        print('{} for training data:'.format(self.error))
        print(error_train)

        print('{} for testing data:'.format(self.error))
        print(error_test)
