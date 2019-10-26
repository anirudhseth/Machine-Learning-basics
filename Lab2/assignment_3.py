import numpy as np
from sklearn import svm,datasets as dt
import matplotlib.pyplot as plt
import random

def plot_classes(classA,classB):
    plt.plot([p[0] for p in classA],[p[1] for p in classA],'g+ ',label = 'Positive +1')
    plt.plot([p[0] for p in classB],[p[1] for p in classB],'b+ ',label = 'Negative -1')
    plt.legend(loc = 'lower right')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Linearly separable, binary classification')
    plt.axis('equal')
def input_sample1():
    classA = np.concatenate((np.random.randn(10,2)*0.2+[1.5,0.5], np.random.randn(10,2)*0.2+[-1.5,0.5]))
    classB = np.random.randn(20,2)*0.2+[0.0,-0.5]
    inputs=np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
    inputs=np.concatenate((classA,classB))
    plot_classes(classA,classB)
    return inputs,targets

def input_sample2():
    classA = np.concatenate((np.random.randn(10,2)*0.2+[2,0.5], np.random.randn(10,2)*0.2+[3,0.5]))
    classB = np.random.randn(20,2)*0.2+[4,0.5]
    inputs=np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
    inputs=np.concatenate((classA,classB))
    plot_classes(classA,classB)
    return inputs,targets

def input_sample3():
    listA = [(random.normalvariate(-2, 1), random.normalvariate(1.5,1)) for i in range(10)] +\
             [(random.normalvariate(2, 1), random.normalvariate(0.5,1)) for i in range(10)]
    listB = [(random.normalvariate(2, 0.5), random.normalvariate(-2, 0.5)) for i in range(10)]
    classA=np.asarray(listA)
    classB=np.asarray(listB)
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

def input_samples5():
    
    classA = np.concatenate((np.random.randn(10 , 2)*0.2 + [0.5, 0.5], np.random.randn(10, 2) * 0.2 + [-0.5,0.5]));
    classB = np.random.randn(20,2) * 0.2 + [0.0, -0.5] 
    inputs=np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
    inputs=np.concatenate((classA,classB))
    plot_classes(classA,classB)
    return inputs,targets   

inputs,targets=input_samples5();
clf = svm.SVC(kernel='linear')
clf.fit(inputs,targets)
# print(clf)
# xrange = np.line(-4, 4, 1)
# yrange = np.arange(-4, 4, 1)

# xx, yy = np.meshgrid(np.arange(-2, 2, 1),np.arange(-2, 2, 1))
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = clf.decision_function(xy).reshape(XX.shape)

# ax = plt.gca()
# ax.contour(XX, YY, Z, levels=[-1, 0, 1], alpha=0.5, colors=('blue', 'black', 'green'),linewidths=(1, 1, 1))

xgrid = np.linspace( -2, 2);
ygrid = np.linspace( -3, 3);
# print(clf.predict([[1,2]])[0])
# print(type(clf.predict([[1,2]])[0]))
grid = np.array([[clf.predict([[x, y]])[0]
                   for x in xgrid]
                   for y in ygrid])
print(grid.shape)
plt.contour(xgrid, ygrid, grid,
           (-1, 0, 1),
           colors=('red', 'black', 'blue'),
           linewidths=(1,2,1));
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Linearly separable, low cost')
plt.axis('equal')
plt.legend(loc = 'lower right')
plt.show()