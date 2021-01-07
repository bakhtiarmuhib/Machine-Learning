import numpy as np
import pandas as pd


def predict(x, w):
    x_shape = x.shape
    features = np.hstack(((np.ones((x_shape[0], 1))),x))
    return features.dot(w)


def cost(x, y, w):
    predicted = predict(x, w)
    squar_error = (predicted - y)**2
    r_m_s_e = (squar_error.sum())/(len(x)*2)
    return r_m_s_e


def gradient(x, y, w, learning_rate):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    w0_derivative = (predict(x, w) - y).sum()/len(x)
    w1_derivative = ((predict(x, w) - y) * x1).sum()/len(x)
    w2_derivative = ((predict(x, w) - y) * x2).sum()/len(x)
    w3_derivative = ((predict(x, w) - y) * x3).sum()/len(x)
    w[0] -= learning_rate * w0_derivative
    w[1] -= learning_rate * w1_derivative
    w[2] -= learning_rate * w2_derivative
    w[3] -= learning_rate * w3_derivative
    return w


def train(x, y, learning_rate=0.00000000001, iteration=10000):
    x_shape = x.shape
    w = np.ones((x_shape[1]+1, 1))
    for i in range(iteration):
        w = gradient(x, y, w, learning_rate)
        cost1 = cost(x, y, w)
        if i % 10 == 0:
            print("weight {} cost {}".format(w, cost1))
    return w




df = pd.read_csv("homeprices.csv")
x1 = np.array(df.drop(['price'], axis=1))
y1 = np.array(df.price)





train(x1,y1)