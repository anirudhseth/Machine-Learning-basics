import matplotlib.pyplot as plt

import monkdata as m
import dtree
import drawtree_qt5
print('Entropy for monk1:',dtree.entropy(m.monk1))
print('Entropy for monk2:',dtree.entropy(m.monk2))
print('Entropy for monk3:',dtree.entropy(m.monk3))

##for x in range(0, 6):
##    print('Info Gain for monk1 with arrtribute A',x+1,':',dtree.averageGain(m.monk1,m.attributes[x]))
##for x in range(0, 6):
##    print('Info Gain for monk2 with arrtribute A',x+1,':',dtree.averageGain(m.monk2,m.attributes[x]))
##for x in range(0, 6):
##    print('Info Gain for monk3 with arrtribute A',x+1,':',dtree.averageGain(m.monk3,m.attributes[x]))
##
##t1=dtree.buildTree(m.monk1,m.attributes)
##print('E Train for Monk1',dtree.check(t1,m.monk1))
##print('E Test for Monk1',dtree.check(t1,m.monk1test))
##
##t2=dtree.buildTree(m.monk2,m.attributes)
##print('E Train for Monk2',dtree.check(t2,m.monk2))
##print('E Test for Monk2',dtree.check(t2,m.monk2test))
##
##t3=dtree.buildTree(m.monk3,m.attributes)
##print('E Train for Monk3',dtree.check(t3,m.monk3))
##print('E Test for Monk3',dtree.check(t3,m.monk3test))

##subset = dtree.select(m.monk1, m.attributes[4], m.attributes[4].values)
##print('subset',m.attributes[4].values)

##drawtree_qt5.drawTree(t1)
##drawtree_qt5.drawTree(t2)
##drawtree_qt5.drawTree(t3)


import random
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

##monk1train, monk1val= partition(m.monk1, 0.3)
##monk2train, monk2val= partition(m.monk1,0.4)
partition2 = [0.3,0.4,0.5,0.6,0.7,0.8]
monk1check = []
for par in range(len(partition2)):
    temp2=0.0
    flag= True
    while (flag):
        y2=0
        monk1train, monk1val= partition(m.monk1,partition2[par])
        tree=dtree.buildTree(monk1train,m.attributes)
        apt= dtree.allPruned(tree)
        temp=dtree.check(apt[0],monk1val)
        y=0;
        for x in range(len(apt)) :
##            print('length',len(apt))
            if(temp<dtree.check(apt[x],monk1val)):
                temp=dtree.check(apt[x],monk1val)
                y=x;
        if temp2<temp:
            temp2=temp
            y2=y
            
        else:
    ##          drawtree_qt5.drawTree(apt[y2])
##            print(y2)
            monk1check.append(dtree.check(apt[y2],monk1val))
            flag=False


##plt.plot(partition2,monk1check)
##plt.show()

##monk1train2, monk1val2= partition(m.monk1, 0.4)
##monk1train3, monk1val3= partition(m.monk1, 0.5)
##monk1train4, monk1val4= partition(m.monk1, 0.6)
##monk1train5, monk1val5= partition(m.monk1, 0.7)
##monk1train6, monk1val6= partition(m.monk1, 0.8)
##
##monk3train1, monk3val1= partition(m.monk3, 0.3)
##monk3train2, monk3val2= partition(m.monk3, 0.4)
##monk3train3, monk3val3= partition(m.monk3, 0.5)
##monk3train4, monk3val4= partition(m.monk3, 0.6)
##monk3train5, monk3val5= partition(m.monk3, 0.7)
##monk3train6, monk3val6= partition(m.monk3, 0.8)
##
##t1=dtree.buildTree(monk1train1,m.attributes)
##apt1= dtree.allPruned(t1)
##drawtree_qt5.drawTree(apt1[4])
##
##
##train, val= partition(m.monk1, 0.3)
##tree=dtree.buildTree(train,m.attributes)
##allprunedtrees=dtree.allPruned(tree)
##        
##    
##
