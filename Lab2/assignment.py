import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def linear_kernel(x_vector,y_vector):
    return numpy.dot(numpy.array(x_vector).transpose(),numpy.array(y_vector))
def polynomial_kernel(x_vector,y_vector,p):
    return numpy.power((numpy.dot(numpy.array(x_vector).transpose(),numpy.array(y_vector))+1),int(p))
def rbf_kernel(x_vector,y_vector,sigma):
    x=numpy.array(x_vector)
    y=numpy.array(y_vector)
    

classA = numpy.concatenate((numpy.random.randn(10,2)*0.2+[1.5,0.5], numpy.random.randn(10,2)*0.2+[-1.5,0.5]))
classB = numpy.random.randn(20,2)*0.2+[0.0,-0.5]
inputs=numpy.concatenate((classA,classB))
targets= numpy.concatenate((numpy.ones(classA.shape[0]),-numpy.ones(classB.shape[0])))
print(targets)
N=inputs.shape[0]

permute=list(range(N))
random.shuffle(permute)
inputs=inputs[permute,:]
targets=targets[permute]

start=numpy.zeros(N)
bounds=[(0, None) for b in range(N)]
# bounds=[(0, C) for b in range(N)]

# print(inputs)
# print(targets)
print(polynomial_kernel([1,2],[3,4],2))
# plt.plot([p[0] for p in classA],[p[1] for p in classA], 'b. ')
# plt.plot([p[0] for p in classB],[p[1] for p in classB], 'r. ')
# plt.axis('equal')
# plt.show()