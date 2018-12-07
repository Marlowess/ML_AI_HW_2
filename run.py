import numpy as np
from sklearn.svm import SVC
from sklearn import datasets, svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from tabulate import tabulate


# This function load the iris dataset in memory and gets just the first two dimensions
def loadData():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    return X, y

def splitting_function(X, y):
    y = y[:].astype(np.float)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=12)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.375, random_state=7)
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out


def svm_with_kernel(X, y, X_train, y_train, X_validation, y_validation, kernel_type, gamma_value):
    accuracy = []
    color_map = {-1: (2, 2, 2), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}
    best_clf = []
    best_score = 0
    for i, cost in enumerate((0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        model = SVC(C=cost, kernel=kernel_type, gamma=gamma_value)
        clf = model.fit(X_train, y_train)
        # title for the plots
        plt.figure(i)
        title = ('Decision surface using SVM with ' + kernel_type + ' Kernel' )
        # Set-up grid for plotting.
        X0, X1 = X_train[:, 0], X_train[:, 1]
        xx, yy = make_meshgrid(X0, X1)

        plot_contours(clf, xx, yy, cmap="brg", alpha=0.8)
        plt.xlabel('Sepal Length')
        plt.ylabel('Sepal Width')
        colors = [color_map[y] for y in y_train]
        plt.scatter(X0, X1, c=y_train, cmap="brg", s=20, edgecolors='black')
        score = clf.score(X_validation, y_validation)
        if(score > best_score):
            best_score = score
            best_clf = clf
        accuracy.append((cost,score))
        plt.suptitle('SVM with C = ' + str(cost) + '\nAccuracy on validation set: ' + str(score))
    plt.show()
    return accuracy, best_clf

def plot_boundaries(X_test, y_test, clf):
    #plt.figure()
    color_map = {-1: (2, 2, 2), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}
    title = ('Decision boundaries on test set')
    # Set-up grid for plotting.
    X0, X1 = X_test[:, 0], X_test[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(clf, xx, yy, cmap="brg", alpha=0.8)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.scatter(X0, X1, c=y_test, cmap="brg", s=20, edgecolors='black')
    plt.title(title)
    plt.show()

def plotSplineFunction(array, kernel_type):
    x = array[:,0]
    y = array[:,1]
    # print(x)
    # print(y)
    xi = [i for i in range(0, len(x))]
    plt.ylim(0.2,0.9)

    plt.plot(xi, y, marker='o', linestyle='-', color='blue', label='Square')
    plt.xlabel('C parameter')
    plt.ylabel('Score')
    plt.xticks(xi, x)

    #plt.plot (x_new, y)
    plt.grid(True)
    plt.title('SVM, {}: accuracy on validation'.format(kernel_type))
    plt.show()
    #plt.savefig(rootFolder + "variance_plot" + '.jpg')

def gridsearch_no_cv(X_train, y_train, X_validation, y_validation):
    best_score = 0
    best_clf = None
    gamma_vals = [1e-5, 1e-4, 1e-3, 2e-3, 1e-2, 2e-1, 1e-1]
    c_vals = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    scores = np.empty([len(gamma_vals) * len(c_vals), 3])
    index = 0
    for i, gamma in enumerate(gamma_vals):
        for j, C in enumerate(c_vals):
            model = SVC(C=C, kernel='rbf', gamma=gamma)
            clf = model.fit(X_train, y_train)
            score = clf.score(X_validation, y_validation)
            scores[index, 0:3] = gamma_vals[i], c_vals[j], score
            index += 1
            if(score > best_score):
                best_score = score
                best_clf = clf
    headers = ["Gamma", "C", "Accuracy on Validation"]
    table = tabulate(scores, headers, tablefmt="fancy_grid", numalign="right")
    print(table)
    return best_clf
#def svc_evaluation(clf, X_test, y_test):

scaler = StandardScaler()
X, y = loadData() # gets input and labels arrays
X = scaler.fit_transform(X)
X_train, X_validation, X_test, y_train, y_validation, y_test = splitting_function(X, y)

scores, clf = svm_with_kernel(X, y, X_train, y_train, X_validation, y_validation, 'linear', 'auto')
print(scores)
print(clf.score(X_test, y_test))
plotSplineFunction(np.asarray(scores), 'Linear Kernel')

scores, clf = svm_with_kernel(X, y, X_train, y_train, X_validation, y_validation, 'rbf', 'auto')
print(scores)
print(clf.score(X_test, y_test))
plotSplineFunction(np.asarray(scores), 'RBF Kernel')

# Gridsearch without CV step
clf = gridsearch_no_cv(X_train, y_train, X_validation, y_validation)
print(clf.score(X_test, y_test))
plot_boundaries(X_test, y_test, clf)

# Gridsearch with CV
X_train = np.concatenate((X_train, X_validation), axis=0)
y_train = np.concatenate((y_train, y_validation), axis=0)

# At this points the dataset proportions are: Training 70% (105 elements) and Testing 30% (45 elements)
