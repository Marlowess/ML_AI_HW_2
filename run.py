import numpy as np
from sklearn.svm import SVC
from sklearn import datasets, svm
import matplotlib.pyplot as plt

# This function load the iris dataset in memory and gets just the first two dimensions
def loadData():
    iris = datasets.load_iris()
    X = iris.data[:,0:2]
    y = iris.target
    X = X[y != 0, :2]
    y = y[y != 0]
    return X, y

def splitting_function(X, y):
    np.random.seed(0)
    num_sample = len(X)
    order = np.random.permutation(num_sample)
    X = X[order]
    y = y[order].astype(np.float)
    X_train = X[:int(.5 * num_sample)]
    y_train = y[:int(.5 * num_sample)]
    X_validation = X[int(.5 * num_sample):int(.7 * num_sample)]
    y_validation = y[int(.5 * num_sample):int(.7 * num_sample)]
    X_test = X[int(.7 * num_sample):]
    y_test = y[int(.7 * num_sample):]
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def svc_function(X, y, kernel, gamma, X_train, y_train, X_validation, y_validation):
    for fig_num, cost in enumerate((0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        clf = svm.SVC(kernel=kernel, C=cost, gamma=gamma)
        clf.fit(X_train, y_train)

        plt.figure(fig_num)
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                    edgecolor='k', s=20)

        # Circle out the test data
        plt.scatter(X_validation[:, 0], X_validation[:, 1], s=80, facecolors='none',
                    zorder=10, edgecolor='k')

        plt.axis('tight')
        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
        y_min = X[:, 1].min()
        y_max = X[:, 1].max()

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                    linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

        plt.title(str.capitalize(kernel) + ' SVM with C = ' + str(cost) + '\nMean accuracy = ' + str(clf.score(X_validation, y_validation)))
    plt.show()

#def svc_evaluation(clf, X_test, y_test):


X, y = loadData() # gets input and labels arrays
X_train, X_validation, X_test, y_train, y_validation, y_test = splitting_function(X, y)
svc_function(X, y, 'linear', 10,  X_train, y_train, X_validation, y_validation)
