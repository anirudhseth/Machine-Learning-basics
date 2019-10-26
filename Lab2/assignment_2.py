print(__doc__)


# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets as dt
from sklearn.datasets import make_classification
def plot_classes(classA,classB):
    plt.plot([p[0] for p in classA],[p[1] for p in classA],'g+ ',label = 'Class A +1')
    plt.plot([p[0] for p in classB],[p[1] for p in classB],'b+ ',label = 'Class B -1')
    plt.legend(loc = 'lower right')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Polynomial Kernel degree=3')
    plt.axis('equal')
def input():
    X = np.c_[(.4, -.7),
          (-1.5, -1),
          (-1.4, -.9),
          (-1.3, -1.2),
          (-1.1, -.2),
          (-1.2, -.4),
          (-.5, 1.2),
          (-1.5, 2.1),
          (1, 1),
          # --
          (1.3, .8),
          (1.2, .5),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
    Y = [0] * 8 + [1] * 8
    return X,Y
def input_sample6():
    np.random.seed(100)
    # X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=1,
    #                          n_clusters_per_class=1)
    # plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
    #         s=25, edgecolor='k')

    
    X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
                s=25, edgecolor='k')
    plt.title("Two informative features, two clusters per class",
          fontsize='small')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Polynomial Kernel degree:4')
    return X1,Y1        
def input_sample5():
    # Make a large circle containing a smaller circle in 2d

    X, Y = dt.make_circles(100, factor=0.2, noise=0.1)
    classC = []
    classC = []
    listA = []
    listB = []
    for i in range(len(Y)):
        if(Y[i]==1):
            listA.append(X[i])
        else:
             listB.append(X[i])
    classA=np.asarray(listA)
    classB=np.asarray(listB)
    print(type(listA))
    print(type(classA))
    plot_classes(classA,classB)
    inputs=np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
    
    return inputs, targets
def input_sample1():
    classA = np.concatenate((np.random.randn(10,2)*0.2+[1.5,0.5], np.random.randn(10,2)*0.2+[-1.5,0.5]))
    classB = np.random.randn(20,2)*0.2+[0.0,-0.5]
    inputs=np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
    inputs=np.concatenate((classA,classB))
    plot_classes(classA,classB)
    return inputs,targets
def input_sample4():

    X, Y = dt.make_moons(100)
    
    classC = []
    listA = []
    listB = []
    for i in range(len(Y)):
        if(Y[i]==1):
            listA.append(X[i])
        else:
             listB.append(X[i])
    classA=np.asarray(listA)
    classB=np.asarray(listB)
    print(type(listA))
    print(type(classA))
    plot_classes(classA,classB)
    inputs=np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
    
    return inputs, targets
# for kernel in ('linear', 'poly', 'rbf'):
np.random.seed(100)
X,Y=input_sample1()
C=float('inf')
print(type(C))
clf = svm.SVC(kernel='poly', degree=4)
clf.fit(X, Y)

# plot the line, the points, and the nearest vectors to the plane


# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],marker = '^', color = 'r')
# plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,edgecolors='k')

plt.axis('tight')
x_min = -3
x_max = 3
y_min = -3
y_max = 3

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
# plt.figure(fignum, figsize=(4, 3))
# plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['black', 'black', 'black'], linestyles=['--', '-', '--'],
            levels=[-.5, 0, .5])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(())
plt.yticks(())
    
plt.show()