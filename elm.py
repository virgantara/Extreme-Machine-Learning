import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

enc = preprocessing.OneHotEncoder(handle_unknown='ignore')

from ELM import ELM

data = pd.read_csv("dataset/iris.csv")

y = data['variety'].astype("category")


X = data.drop('variety',axis=1)
y = np.array(y).reshape(-1,1)
y_encoded = enc.fit_transform(y).toarray().astype(np.float32)
n_classes = y_encoded.shape[1]

normalized_X = preprocessing.normalize(X)

x_train, x_test, t_train, t_test = train_test_split(normalized_X, y_encoded, test_size=0.25, random_state=42,
                                                    shuffle=True)

model = ELM(
    jumlah_input_nodes=X.shape[1],
    jumlah_hidden_node=5,
    jumlah_output=n_classes,
)

model.fit(x_train, t_train)
loss, acc = model.evaluate(x_test, t_test)

print("Loss : ", loss)
print("Accuracy : ", acc)
