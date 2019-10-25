

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