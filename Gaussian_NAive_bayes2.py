import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from math import pi


iris = load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,train_size=120,random_state=1000)
#print(x_train)

clc= np.zeros((3,1))
lst1,lst2,lst3=[],[],[]
for i in range(120):
    if y_train[i] ==0:
        clc[0][0] = clc[0][0] + 1
        lst1.append(x_train[i])
    elif y_train[i] ==1:
        clc[1][0] = clc[1][0] + 1
        lst2.append(x_train[i])
    else:
        clc[2][0] = clc[2][0] + 1
        lst3.append(x_train[i])
y_0 = np.array(lst1)
y_1 = np.array(lst2)
y_2 = np.array(lst3)
clc =clc/120
def mean(inp,j):
    array = inp[:,j]
    return np.mean(array)
def std_deviation(inp , c):
    array = inp[:,c]
    return np.var(array)

def prob(inp,c):
    p=1
    if c == 0:
        yd = y_0
    elif c == 1:
        yd = y_1
    else:
        yd = y_2
    for i in range(4):
        p=p*np.exp(-((inp[i]-mean(yd,i))**2)/(2*std_deviation(yd,i)))/np.sqrt(2*pi*std_deviation(yd,i))
        #p = p* (np.exp(-1*((inp[i] - mean(yd,i))**2)/(2*std_deviation(yd,i))))/(np.sqrt(2*pi*std_deviation(yd,i)))
    if c==0:
        return p*(clc[0][0])
    elif c==1:
        return p*(clc[1][0])
    else:
        return p*(clc[2][0])


def naive(inp):
    pro = {}
    for i in range(3):
        pro[i] = prob(inp,i)
    ke = max(pro, key=pro.get)
    return ke



prediction=np.zeros(len(y_test))
for i in range(len(y_test)):
    prediction[i]=naive(x_test[i])
print("ACCURACY=",np.mean(y_test==prediction))


