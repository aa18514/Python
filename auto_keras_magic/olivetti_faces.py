import autokeras as ak
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report

if __name__ == "__main__":
    olivetti_faces = fetch_olivetti_faces()
    x, y = olivetti_faces['data'], olivetti_faces['target']
    x = x.reshape(len(y), 64, 64)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))
    clf = ak.ImageClassifier(verbose=True)
    clf.fit(X_train, y_train, time_limit=10)
    y_pred = clf.predict(X_test)
    test_accuracy = np.sum(y_pred == y_test)
    test_accuracy = 100 * test_accuracy/len(y_pred)
    print("test accuracy %f: " % test_accuracy)
    print(classification_report(y_test, y_pred))
