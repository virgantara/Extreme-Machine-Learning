import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

enc = preprocessing.OneHotEncoder(handle_unknown='ignore')

from ELM import ELM
iris = load_iris()

X = iris.data
y = iris.target.reshape(-1, 1)

n_classes = 3
normalized_X = preprocessing.normalize(X)

x_train, x_test, t_train, t_test = train_test_split(normalized_X, y)
y_encoded = pd.DataFrame(enc.fit_transform(y).toarray())

t_train = enc.fit_transform(t_train).toarray().astype(np.float32)
t_test = enc.fit_transform(t_test).toarray().astype(np.float32)


model = ELM(
    jumlah_input_nodes=4,
    jumlah_hidden_node=10,
    jumlah_output=n_classes,
)

model.fit(x_train, t_train)
loss, acc = model.evaluate(x_train, t_train)

print("Loss : ",loss)
print("Accuracy : ",acc)