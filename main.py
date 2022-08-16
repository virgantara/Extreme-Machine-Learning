import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

# from ELM import ELM

from ELMOtenim import ELMOtenim
#
iris = load_iris()

X = iris.data
y = iris.target.reshape(-1,1)

n_classes = 3
normalized_X = preprocessing.normalize(X)

x_train, x_test, t_train, t_test = train_test_split(normalized_X, y)
y_encoded = pd.DataFrame(enc.fit_transform(y).toarray())

t_train = enc.fit_transform(t_train).toarray().astype(np.float32)
t_test = enc.fit_transform(t_test).toarray().astype(np.float32)

# jumlah_input_node = normalized_X.shape[1]
# jumlah_hidden_node = 5
# jumlah_output_node = n_classes

model = ELMOtenim(
    n_input_nodes=4,
    n_hidden_nodes=10,
    n_output_nodes=n_classes,
    loss='mean_squared_error',
    activation='sigmoid',
    name='elm',
)
# model = ELMOtenim(jumlah_input_node,50, jumlah_output_node)

model.fit(x_train, t_train)
# hasil = model.evaluate(X_train, y_train)

train_loss, train_acc = model.evaluate(x_train, t_train, metrics=['loss', 'accuracy'])
print('train_loss: %f' % train_loss)
print('train_acc: %f' % train_acc)
# print(hasil)
# X_train = np.arange(-10.,10., 0.004) # generate 5000 data
#
# y_temp = np.sin(X_train) / (X_train)
# y_train = np.random.uniform(y_temp - (-0.2), y_temp + (-0.2))
#
# X_test = np.arange(-10,10, 0.004) # generate 5000 data
# y_test = np.sin(X_test) / (X_test)
#
# X_train = X_train.reshape(-1, 1)
# X_test = X_test.reshape(-1, 1)
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def feedforward(X, W, B):
#     H = np.dot(X, W) + B
#     return sigmoid(H)
#
# def prediksi(X, W, B, beta):
#     o = feedforward(X, W, B)
#     return np.dot(o, beta)
#
# def mse(yT, yP):
#     d = yT - yP
#     return np.mean(d**2)
#
# num_inputs = X_train.shape[1]
# num_hidden = 20
# num_outputs = len(y_train)
# W = np.random.uniform(-1.,1.,size=(num_inputs, num_hidden))
# B = np.random.uniform(size=[num_hidden])
#
# # H = np.zeros(shape=(num_inputs, num_hidden))
# zH = feedforward(X_train, W, B)
# zH_inv = np.linalg.pinv(zH)
# T = y_train
# beta = np.dot(zH_inv, T)
#
# hasil_prediksi = prediksi(X_test, W, B, beta)
#
# print('Nilai MSE: ', mse(y_test, hasil_prediksi))
# plt.figure(figsize=(15, 10), dpi=120)
# plt.plot(X_train,y_train,'c.')
# plt.plot(X_test,y_test,'r.')
# plt.plot(X_test,hasil_prediksi,'g.')
# plt.show()
