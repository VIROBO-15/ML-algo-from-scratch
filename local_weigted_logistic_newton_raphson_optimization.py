from numpy import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
iris.data = iris.data[:100,:2]
iris.data = insert(iris.data,0,1,axis=1)
iris.target = iris.target[:100]

x_train,x_cv,y_train,y_cv = train_test_split(iris.data,iris.target,train_size=70,random_state=1000)
lamda = 0.0001
theta = random.rand(3,1)
tou=0.03
def sig_moid(inp,theta):

    value = 0
    for j in range(3):
        value = value + (inp[j]*theta[j][0])
    rvalue = 1/(1+exp(-1*value))
    return rvalue

def weight(data1,data2):
    length = len(data2)
    d=0
    for i in range(length):
        d = d + power((data1[i]-data2[i]),2)
    w=exp((-1*d)/(2*tou*tou))
    return w
def dia_D(data1,data2):
    lst=[]
    lengt=len(data1)
    for i in range(lengt):
        d=-1*weight(data1[i],data2)*dot(sig_moid(data1[i],theta),(1-sig_moid(data1[i],theta)))
        lst.append(d)
    d=diag(lst)
    return d
def Hessian(data1,data2):
    H = dot(dot(transpose(data1),dia_D(data1,data2)),data1) - lamda*identity(3)
    #print('hessian',H.shape)
    return H
def z(data1,data2,test):
    length = len(data1)
    z = zeros((length, 1))
    for i in range(length):
       p=weight(data1[i],data2)*(test[i]-sig_moid(data1[i],theta))
       z[i][0]=p
    return z
def grad(data1,data2,test):
    g = dot(transpose(data1),z(data1,data2,test)) - lamda*theta
    #print('g',g.shape)
    return g

def newton_Rapson_optim(data1,data2,test):
    theta = random.rand(3,1)
    g=grad(data1,data2,test)
    inv_hes=linalg.inv(Hessian(data1,data2))
    theta = theta - dot(inv_hes,g)
    return theta
def loss_fn(data1,data2,test):
    leng=len(data1)
    theta = newton_Rapson_optim(data1,data2,test)
    loss=0
    regu = -0.5*lamda*dot(transpose(theta),theta)
    for i in range(leng):
        loss=loss + weight(data1[i],data2)*(test[i]*log(sig_moid(data1[i],theta)) + (1 - test[i])*(log(1 - sig_moid(data1[i],theta))))
    loss = loss+regu
    return loss
def main(data1,data2,test1):
    leng= len(data2)
    for i in range(leng):
        loss = loss_fn(data1,data2[i],test1)
    print(loss)
main(x_train,x_cv,y_train)
#s = newton_Rapson_optim(x_train[1:10],x_cv[1],y_train[1:10])
#print(s)

def output(data1,data2,test):
    value = 0
    lst=[]
    s=[]
    for k in range(len(data1)):
        data3 = data1[k]
        for i in range(len(data2)):
            value = 0
            for j in range(3):
                value = value + data3[j]*newton_Rapson_optim(data1,data2[i],test)[j]
            lst.append(value)
        for i in range(len(data2)):
            if lst[i]>0.5:
                s.append(1)
            else:
                s.append(0)
        return s
def accuracy(data1,data2,test1,test2):
    cnt = 0
    p=output(data1,data2,test1)
    for i in range(len(data2)):
        if test2[i]==p[i]:
            cnt = cnt+1
    acc=(cnt*100)/len(data2)
    print(acc)
accuracy(x_train,x_cv,y_train,y_cv)

