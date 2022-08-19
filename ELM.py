import numpy as np
from sklearn.metrics import r2_score,mean_squared_error

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ELM:

    def __init__(self, jumlah_input_nodes, jumlah_hidden_node, jumlah_output):
        self.jumlah_input_node = jumlah_input_nodes
        self.jumlah_hidden_node = jumlah_hidden_node
        self.jumlah_output = jumlah_output

        self.weights = np.random.uniform(-1.,1.,size=(self.jumlah_input_node, self.jumlah_hidden_node))
        self.biases = np.random.uniform(size=[self.jumlah_hidden_node])


    def fit(self, x, t):
        nilai_x = x.dot(self.weights) + self.biases
        H = sigmoid(nilai_x)

        H_inv = np.linalg.pinv(H)

        self.beta = H_inv.dot(t)

    def feedforward(self, x):
        h = sigmoid(x.dot(self.weights) + self.biases)
        return h.dot(self.beta)

    def evaluate(self, x, t):
        yp = self.feedforward(x)
        yt = t

        if t.shape[1] == 1: #regression
            r2 = r2_score(yt, yp)
            mse = mean_squared_error(yt, yp)
            return r2, mse
        else:
            y_pred = np.argmax(yp, axis=-1)
            y_true = np.argmax(yt, axis=-1)

            loss = mean_squared_error(y_true, y_pred)
            acc = np.sum(y_pred == y_true) / len(t)
            return loss, acc

