import matplotlib.pyplot as plt
import monkdata as m
import dtree
import drawtree_qt5
import statistics as s

import random
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

fraction = [0.3,0.4,0.5,0.6,0.7,0.8]
monk1check = [[],[],[],[],[],[],[]]


monk1sample=[]
monk3sample=[]

for par in range(len(fraction)):
    iter=0
    monk1check = []
    while(iter<5000):
        iter=iter+1
        monk1train, monk1val= partition(m.monk1,fraction[par])
        tree=dtree.buildTree(monk1train,m.attributes)
        apt= dtree.allPruned(tree)
        temp=dtree.check(apt[0],monk1val)
        for x in range(len(apt)) :
            if(temp<dtree.check(apt[x],monk1val)):
                temp=dtree.check(apt[x],monk1val)
        monk1check.append(temp)
    monk1sample.append(monk1check)   
plot1data=[]
for x in range(len(monk1sample)):
    temp=s.mean(monk1sample[x])
    plot1data.append(temp)

for par in range(len(fraction)):
    iter=0
    monk3check = []
    while(iter<5000):
        iter=iter+1
        monk3train, monk3val= partition(m.monk3,fraction[par])
        tree3=dtree.buildTree(monk3train,m.attributes)
        apt3= dtree.allPruned(tree3)
        temp3=dtree.check(apt3[0],monk3val)
        for x in range(len(apt3)) :
            if(temp3<dtree.check(apt3[x],monk3val)):
                temp3=dtree.check(apt3[x],monk3val)
        monk3check.append(temp3)
    monk3sample.append(monk3check)   
plot3data=[]
for x in range(len(monk3sample)):
    temp3=s.mean(monk3sample[x])
    plot3data.append(temp3)
plot1data[:]=[1-y for y in plot1data]
plot3data[:]=[1-y for y in plot3data]
plt.plot(fraction,plot1data, label='Monk1', marker='o')
plt.plot(fraction,plot3data, label='Monk3', marker='o')
plt.xlabel('Parameter fraction')
plt.ylabel('Classification error')
plt.legend(loc='best')
plt.title('Classification error as a function of the Parameter fraction') 
plt.show()
    


# plt.plot(partition2,monk1check)
# plt.show()
