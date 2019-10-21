import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
iris.data = iris.data[:100,:2]
iris.data=np.insert(iris.data,0,1,axis=1)
iris.target = iris.target[:100]
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,train_size=70,random_state=1000)

alpha =0.09
theta = np.random.random((3,1))
#print(iris)
def sig_moid(inp):

    value = 0
    for j in range(3):
        value = value + (inp[j]*theta[j][0])
    rvalue = 1/(1+np.exp(-1*value))
    return rvalue

def optim_grad_descent():
    for k in range(100):
        for i in range(70):
            for j in range(3):
                theta[j][0] = theta[j][0] + alpha*(y_train[i] - sig_moid(x_train[i]))*x_train[i][j]
    return theta


def optim_newton_method(inp,test):
    theta = np.dot((np.linalg.inv(np.dot(np.transpose(inp),inp))),(np.dot((np.transpose(inp),test))))
    return theta

def loss_fun(theta):
    loss = 0
    for i in range(70):
        loss=loss+(y_train[i]*(np.log(sig_moid(x_train[i])) + (1 - y_train[i])*(np.log(1 - sig_moid(x_train[i])))))
        #print(loss)
    return loss

def output(inp):
    value=0
    for j in range(3):
        value=value+(inp[j]*optim_grad_descent()[j])
    if(value>=0):
        return(1)
    else:
        return(0)


prediction=np.zeros(len(y_test))
for i in range(len(y_test)):
    prediction[i]=output(x_test[i])
print("ACCURACY=",np.mean(y_test==prediction))

