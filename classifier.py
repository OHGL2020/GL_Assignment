from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot

# Get Iris data set

iris = load_iris()

data = iris.data
labels = iris.target
target_names = iris.target_names
y = []

for i, j in enumerate(labels):
    if j == 0:
        y.append(target_names[0])
    if j == 1:
        y.append(target_names[1])
    if j == 2:
        y.append(target_names[2])

y = np.array(y)

np.random.seed(117)
shuffle_data = np.random.permutation(len(data))
X = data[shuffle_data]
y = y[shuffle_data]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=17)


# Visualization

def visualization():

    setosa = [[], [], [], []]
    versicolor = [[], [], [], []]
    virginica = [[], [], [], []]

    for i2, [j0, j1, j2, j3] in enumerate(X):
        if y[i2] == "setosa":
            setosa[0].append(j0)
            setosa[1].append(j1)
            setosa[2].append(j2)
            setosa[3].append(j3)
        if y[i2] == "versicolor":
            versicolor[0].append(j0)
            versicolor[1].append(j1)
            versicolor[2].append(j2)
            versicolor[3].append(j3)
        if y[i2] == "virginica":
            virginica[0].append(j0)
            virginica[1].append(j1)
            virginica[2].append(j2)
            virginica[3].append(j3)

    plt.scatter(setosa[0], setosa[1])
    plt.scatter(versicolor[0], versicolor[1])
    plt.scatter(virginica[0], virginica[1])
    plt.legend(['setosa', 'versicolor', 'virginica'])
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('Iris dataset')

    plt.show()

    plt.scatter(setosa[2], setosa[3])
    plt.scatter(versicolor[2], versicolor[3])
    plt.scatter(virginica[2], virginica[3])
    plt.legend(['setosa', 'versicolor', 'virginica'])
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.title('Iris dataset')

    plt.show()


# visualization()


# Model

clf = svm.SVC(kernel='linear', C=1, degree=3, probability=True)
score = clf.fit(x_train, y_train)
prediction = clf.predict(x_test)


# Estimate accuracy of data classification

print("Accuracy of data classification =", clf.score(x_test, y_test))
print(classification_report(y_test, prediction))


# To CSV

def toCVV(x_test):
    x_test = np.transpose(x_test)
    d = {'sepal_length': x_test[0], 'sepal_width': x_test[1],
         'petal_length': x_test[2], 'petal_width': x_test[3],
         'Y True': y_test, 'Prediction': prediction}
    df = pd.DataFrame(d)
    df.to_csv('result.csv')


toCVV(x_test)


def roc_auc_curve(y_test, prob_pred, name):
    fig, ax = plt.subplots(figsize=(15, 7))
    scikitplot.metrics.plot_roc_curve(y_test, prob_pred, ax=ax)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(name, bbox_inches=extent.expanded(1.1, 1.2))


prob_pred = clf.predict_proba(x_test)
roc_auc_curve(y_test, prob_pred, "ROC_AUC_Curve")
