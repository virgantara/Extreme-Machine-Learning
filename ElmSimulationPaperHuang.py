import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

X_train = np.arange(-10.,10., 0.004) # generate 5000 data

y_temp = np.sin(X_train) / (X_train)
y_train = np.random.uniform(y_temp - (-0.2), y_temp + (-0.2))

X_test = np.arange(-10,10, 0.004) # generate 5000 data
y_test = np.sin(X_test) / (X_test)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feedforward(X, W, B):
    H = np.dot(X, W) + B
    return sigmoid(H)

def prediksi(X, W, B, beta):
    o = feedforward(X, W, B)
    return np.dot(o, beta)

def mse(yT, yP):
    d = yT - yP
    return np.mean(d**2)

num_inputs = X_train.shape[1]
num_hidden = 20
num_outputs = len(y_train)
W = np.random.uniform(-1.,1.,size=(num_inputs, num_hidden))
B = np.random.uniform(size=[num_hidden])

# H = np.zeros(shape=(num_inputs, num_hidden))
zH = feedforward(X_train, W, B)
zH_inv = np.linalg.pinv(zH)
T = y_train
beta = np.dot(zH_inv, T)

hasil_prediksi = prediksi(X_test, W, B, beta)

print('Nilai MSE: ', mse(y_test, hasil_prediksi))
plt.figure(figsize=(15, 10), dpi=120)
plt.plot(X_train,y_train,'c.')
plt.plot(X_test,y_test,'r.')
plt.plot(X_test,hasil_prediksi,'g.')
plt.show()
