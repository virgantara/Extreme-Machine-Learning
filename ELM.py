import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(y_true, y_pred):
    return 0.5 * np.mean((y_true - y_pred)**2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

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
        yp = list(self.feedforward(x))
        yt = t
        y_pred = np.argmax(yp, axis=-1)
        y_true = np.argmax(yt, axis=-1)

        loss = mse(y_true, y_pred)
        acc = np.sum(y_pred == y_true) / len(t)
        return loss, acc
