import numpy as np
import math, random
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def objective(alpha):
    temp = np.dot(alpha, np.dot(np.transpose(alpha), P))/2;
    print('P: ',P.shape);
    print('alpha', type(alpha));
    print('transpose', np.transpose(alpha).shape);
    return temp - np.sum(alpha);
    
def zerofun(alpha):
    return np.dot(alpha,targets);

def kernel(x1, x2):
#    return pow(np.dot(x1, np.transpose(x2))+1, poly);  ## polynomial kernel
#    return np.dot(x1, np.transpose(x2));
    return math.exp(-(np.linalg.norm(x1-x2))/(2*pow(sigma,2)));     ## RBF Kernel

  
def indicator(x,y):
    s = np.array([[x,y]])
    temp_k = 0;
    for i in range(len(vectors)):
#        print(classes[i]);
        temp_k += vectors[i] * classes[i] * kernel(s, points[i]);
#    print(temp_k)
    return temp_k - b;       ## only for RBF
#    return temp_k[0] - b;
np.random.seed(100);

#simple
#classA = np.concatenate((np.random.randn(50, 2) * 0.5 + [1.5, 0.5], np.random.randn(50, 2) * 0.2 + [-1.5, -0.5]));
#classB = np.random.randn(50, 2) * 0.2 + [0.0, -0.5]

#medium
classA = np.concatenate((np.random.randn(10 , 2)*0.2 + [0.5, 0.5], np.random.randn(10, 2) * 0.2 + [-0.5,0.5]));
classB = np.random.randn(20,2) * 0.2 + [0.0, -0.5]

#difficult
#classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]));
#classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5];

inputs = np.concatenate((classA, classB));
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])));
poly = 3;
sigma = 1;
N = inputs.shape[0];

permute = list(range(N));
random.shuffle(permute);

inputs = inputs[permute, :];
targets = targets[permute];

P = np.zeros((N,N));
alpha = np.zeros(N);
C = None;
bound = [(0, C) for t in range(N)];
#print(bound)
constraint = {'type': 'eq', 'fun': zerofun};

for i in range(N):
    for j in range(N):
        P[i][j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j]);


ret = minimize(objective, alpha, bounds = bound, constraints = constraint);

if(ret['success'] == False):
    print("no solution found");
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.');
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.');
    plt.axis('equal');
    plt.show();
    exit(0);

a = ret['x'];

vectors = [];
classes = [];
points = [];
for i in range(len(a)):
#    vectors.append([]);
    if (a[i] > 0.00001):
#        print(i)
        points.append(inputs[i]);
        vectors.append(a[i]);
        classes.append(targets[i]);

vectors = np.asarray(vectors);
points = np.asarray(points);
classes = np.asarray(classes);

temp_k = 0;
for i in range(len(vectors)):
    temp_k += vectors[i] * classes[i] * kernel(points[0], points[i]);

b = temp_k - classes[0];

plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.');
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.');
plt.axis('equal');
#plt.show();

xgrid = np.linspace( -2, 2);
ygrid = np.linspace( -3, 3);

grid = np.array([[indicator(x, y)
                    for x in xgrid]
                    for y in ygrid])
print(grid.shape)
plt.contour(xgrid, ygrid, grid,
            (-1, 0, 1),
            colors=('red', 'black', 'blue'),
            linewidths=(1,2,1));
plt.show();
