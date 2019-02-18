import numpy as np
import random
import scipy.io as io
import matplotlib.pyplot as plt


twitter=io.loadmat('twitter.mat')
Xtr=twitter['Xtr']
Xte=twitter['Xte']
YteMat=twitter['YteMat']
YtrMat=twitter['YtrMat']
print(Xtr.shape)



#update B
def update_B(X,B,theta):
    d1=np.matmul(B,theta)
    d1=d1+np.finfo(np.float64).eps
    a=np.divide(X,d1)
    s2=np.matmul(a,theta.T)
    num=np.multiply(B,s2)
    o=np.ones((X.shape[0],X.shape[0]))
    denom=np.matmul(o,num)
    denom=denom+np.finfo(np.float64).eps
    B_final=np.divide(num,denom)
    return B_final


def update_theta(X,B_new,theta):
    d1=np.matmul(B_new,theta)
    d1=d1+np.finfo(np.float64).eps
    a=np.divide(X,d1)
    s2=np.matmul(B_new.T,a)
    num=np.multiply(theta,s2)
    o=np.ones((50,50))
    denom=np.matmul(o,num)
    denom=denom+np.finfo(np.float64).eps
    theta_final=np.divide(num,denom)
    return theta_final


#Learn the B and theta using PLSI
def train_PLSI(X,B,theta):
    cnt=0
    Xhat=np.matmul(B,theta)
    #add small value to the Xhat to avoid NaN
    Xhat=Xhat+np.finfo(np.float64).eps
    div=np.divide(X,Xhat)
    div=div+np.finfo(np.float64).eps
    ldiv=np.log(div)
    l1=np.multiply(X,ldiv)
    lfinal=l1-X+Xhat
    while True:
        old_sum=np.sum(np.sum(lfinal,1))
        
        #update the B and theta
        B_new=update_B(X,B,theta)
        theta_new=update_theta(X,B_new,theta)
        B=B_new
        theta=theta_new
        
        #calculate the error
        Xhat=np.matmul(B,theta)
        #add small value to the Xhat to avoid NaN
        Xhat=Xhat+np.finfo(np.float64).eps
        div=np.divide(X,Xhat)
        div=div+np.finfo(np.float64).eps
        ldiv=np.log(div)
        l1=np.multiply(X,ldiv)
        lfinal=l1-X+Xhat
        
        print(np.sum(np.sum(lfinal,1)))
        
        if (old_sum-(np.sum(np.sum(lfinal,1))))<1:
            break
        cnt=cnt+1
        #print(cnt)
    return B,theta

    
#topic matrix
#random initialization of B and theta matrix
B=np.random.rand(Xtr.shape[0],50)
theta=np.random.rand(50,Xtr.shape[1])
print(B.shape)
print(theta.shape)

B_train,theta_train=train_PLSI(Xtr,B,theta)
print('Train Topic Matrix',B_train)
print('Train Weights Matrix',theta_train)



#Get the weight matrix by reusing train topic matrix for Test PLSI
def test_update_theta(Xte,B_train,theta_test):
    d1=np.matmul(B_train,theta_test)
    d1=d1+np.finfo(np.float64).eps
    a=np.divide(Xte,d1)
    s2=np.matmul(B_train.T,a)
    num=np.multiply(theta_test,s2)
    o=np.ones((50,50))
    denom=np.matmul(o,num)
    denom=denom+np.finfo(np.float64).eps
    theta_test_final=np.divide(num,denom)
    return theta_test_final


def test_PLSI(Xte,B_train,theta_test):
    cnt=0
    Xhat=np.matmul(B_train,theta_test)
    #add small value to the Xhat to avoid NaN
    Xhat=Xhat+np.finfo(np.float64).eps
    div=np.divide(Xte,Xhat)
    div=div+np.finfo(np.float64).eps
    ldiv=np.log(div)
    l1=np.multiply(Xte,ldiv)
    lfinal=l1-Xte+Xhat
    while True:
        old_sum=np.sum(np.sum(lfinal,1))
        
        #update the B and theta
        theta_new=test_update_theta(Xte,B_train,theta_test)
        theta_test=theta_new
        
        #calculate the error
        Xhat=np.matmul(B_train,theta_test)
        
        #add small value to the Xhat to avoid NaN
        Xhat=Xhat+np.finfo(np.float64).eps
        div=np.divide(Xte,Xhat)
        
        #add small value to the div to avoid NaN after log
        div=div+np.finfo(np.float64).eps
        ldiv=np.log(div)
        l1=np.multiply(Xte,ldiv)
        lfinal=l1-Xte+Xhat
        
        print(np.sum(np.sum(lfinal,1)))
        
        if (old_sum-(np.sum(np.sum(lfinal,1))))<1:
            break
        cnt=cnt+1
    return theta_test



# get the test weights 

#first randomly initialize the weights
theta_test=np.random.rand(50,Xte.shape[1])
print(theta_test.shape)
theta_test=test_PLSI(Xte,B_train,theta_test)
print('Test Weight Matrix',theta_test)


# #### Perceptron Learning

def softmax(z):
    return (np.exp(z)/np.sum(np.exp(z),axis=0))

def perceptron_lrn(Zd,wts,labels,epochs,threshold,lrn_rate):
    err_conv=[]
    
    a=np.dot(wts,Zd)
    y_hat=softmax(a)
    log_likelihood=np.log(y_hat)
    error=-np.sum(labels*log_likelihood,axis=1)/labels.shape[1]
    error=np.mean(error)
    print(error)

    err_conv.append(error)
    i=1 
    
    while True:
        
        delta=(labels-y_hat)
        wts=wts+(np.dot(delta,Zd.T)*lrn_rate)
        old_error=error
        
        #cross entropy as error
        #added small number to avoid NaN
        
        a=np.dot(wts,Zd)
        y_hat=softmax(a)
        
        
        log_likelihood=np.log(y_hat)
        error=-np.sum(labels*log_likelihood,axis=1)/labels.shape[1]
        error=np.mean(error)
        err_conv.append(error)
        print(error)
        
        if (old_error-error)<0.000000001:
            break
        print('The error for epoch',i,'=',error)
                    
        i=i+1
        
    return y_hat,wts,err_conv
    

def find_acc(preds,labels):
    ypreds=np.zeros((preds.shape[0],preds.shape[1]))
    for i in range(preds.shape[1]):
        row=np.argmax(preds[:,i])
        ypreds[row,i]=1
    count=0
    for i in range(ypreds.shape[1]):
        bool1=np.all(labels[:,i]==ypreds[:,i])
        if bool1==True:
            count=count+1
    acc=count/ypreds.shape[1]
    return acc


#init wts 
Zd=theta_train
#Z = Zd - np.mean(Zd) / np.std(Zd)
#Zd=np.concatenate((theta_train,np.ones(shape=(1,theta_train.shape[1]))),axis=0)
print(Zd.shape)
#3 output units
wts=np.random.rand(3,Zd.shape[0])
print(wts.shape)
preds,wts,err_conv=perceptron_lrn(Zd,wts,YtrMat,30000,0.1,0.0005)

#Train Classification Accuracy
acc=find_acc(preds,YtrMat)
print('The accuracy is (%): ',acc*100)


def test_pl(theta_test,wts):
    a=np.dot(wts,theta_test)
    y_hat=softmax(a)
    return y_hat

#Test Classification Accuracy
preds=test_pl(theta_test,wts)
acc=find_acc(preds,YteMat)
print('The accuracy is (%): ',acc*100)

