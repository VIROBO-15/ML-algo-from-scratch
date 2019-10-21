import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,train_size=120,random_state=1000)

theta = np.random.rand(3,4)
alpha = 0.02
theta[2,:]=0
def softmax_pre_model(inp,c):
    pre_model = 0
    for i in range(3):
        pre_model = pre_model + np.exp(np.dot(theta[i,:],inp))
    n = np.exp(np.dot(theta[c,:],inp))
    s = n/pre_model
    return s
def st(y,z):
    if y==z:
        return 1
    else :
        return 0
def loss(inp,out):
    s1=0
    for i in range(len(inp)):
        s2=0
        for j in range(3):
            s2 = s2 + np.log(softmax_pre_model(inp[i],j))*st(out[i],j)
        s1 = s1+s2
        #print(s1)
    return s1
def gradient_descent():
    s=0
    for m in range(500):
        for i in range(len(x_train)):
            for j in range(3):
                theta[j,:] = theta[j,:] + alpha *(x_train[i] * (st(y_train[i],j) - softmax_pre_model(x_train[i],j)))
    return theta
theta = gradient_descent()

def output(inp):
    out=np.zeros(3)
    for j in range(3):
        out[j]=softmax_pre_model(inp,j)
    c=max(out)
    for j in range(3):
        if(out[j]==c):
            c=j
            break
    return c


prediction=np.zeros(len(y_test))
for i in range(len(y_test)):
    prediction[i]=output(x_test[i])
print("ACCURACY=",np.mean(y_test==prediction))





            
