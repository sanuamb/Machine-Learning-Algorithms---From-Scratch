import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt


#load the train and test matrix created in matlab 
example=io.loadmat('InputTrain.mat')
Xtrain=example['Xtrain']
print(Xtrain.shape)
example=io.loadmat('TrainTarget.mat')
Mtrain=example['Mtrain']
print(Mtrain.shape)
example=io.loadmat('InputTest.mat')
Xtest=example['Xtest']
print(Xtest.shape)


#Number of units to be used
inp_units=513
hidden_units=50
output_units=513


def sigmoid(z):
    denom=1+np.exp(np.negative(z))
    return (1/denom)

def ftanh(z):
    #return (np.exp(z)-np.exp(np.negative(z)))/(np.exp(z)+np.exp(np.negative(z)))
    return np.tanh(z)


def derivative_ftanh(z):
    return (1-np.power(z,2))


def derivative_sigmoid(z):
    return (z*(1-z))


def init(inp_units,hidden_units,output_units):
    #np.random.seed(43)
    #wt1=np.random.normal(0,0.5,(hidden_units,inp_units))
    wt1=np.random.randn(hidden_units,inp_units)
    #b1=np.random.normal(0,0.5,(hidden_units,1))
    #b1=np.ones((hidden_units,1))
    b1=np.random.randn(hidden_units,1)
    #np.random.seed(33)
    #wt2=np.random.normal(0,0.5,(output_units,hidden_units))
    wt2=np.random.randn(output_units,hidden_units)
    b2=np.random.randn(output_units,1)
    #b2=np.random.rand(output_units,1)
    #b2=np.ones((output_units,1))
    return wt1,b1,wt2,b2


def backprop(X,wt1,b1,wt2,b2,target,epochs,lrn_rate):
   
    err_conv=[]
    
    for i in range(0,epochs):
    
        z1=np.dot(wt1,X)+b1
        a1=ftanh(z1)
    
        z2=np.dot(wt2,a1)+b2
        a2=sigmoid(z2)
    
        #backprop
        oerr=(a2-target)
        delta_o=oerr*derivative_sigmoid(a2)
    
        herr=np.dot(wt2.T,delta_o)
        delta_h=herr*derivative_ftanh(a1)
    
        #update weights
        dw2=np.dot(delta_o,a1.T)
        dw1=np.dot(delta_h,X.T)
    
        wt1=wt1-(lrn_rate*dw1)
        wt2=wt2-(lrn_rate*dw2)
    
        b1=b1-(np.dot(delta_h,np.ones((786,1)))*lrn_rate)
        b2=b2-(np.dot(delta_o,np.ones((786,1)))*lrn_rate)
        
        error=np.sum(0.5 * ((target-a2)**2))/target.shape[1]
        print(error)
        #if (old_error-error)<0:
            #break
        #old_error=error
    
    return wt1,b1,wt2,b2
    

wt1,b1,wt2,b2=init(inp_units,hidden_units,output_units)
wt1,b1,wt2,b2=backprop(Xtrain,wt1,b1,wt2,b2,Mtrain,2000,0.005)


def forwardprop(X,wt1,b1,wt2,b2):
    h_pre_act=np.dot(wt1,X)+b1
    h_act=ftanh(h_pre_act)
    
    o_pre_act=np.dot(wt2,h_act)+b2
    o_act=sigmoid(o_pre_act)
    
    return o_act


#Xtest to get final Mtest
Mtest=forwardprop(Xtest,wt1,b1,wt2,b2)
print(Mtest)


test_target={}
test_target['Mtest']=Mtest
io.savemat('test_target.mat',test_target)


#snr
import librosa as l

s,fs=l.load('tes.wav',sr=None)
shat,fshat=l.load('reconNN.wav',sr=None)

num=np.dot(s.T,s)
denom=np.dot((s-shat).T,(s-shat))
snr=10*np.log10(num/denom)
print(snr)

