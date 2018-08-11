import autokeras as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report


def classification_report_csv(report: classification_report)->classification_report:
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ') 
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', index=False)

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
    report = classification_report(y_test, y_pred)
    classification_report_csv(report)
