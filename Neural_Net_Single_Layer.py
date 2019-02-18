
import numpy as np
import random
import scipy.io as io
import matplotlib.pyplot as plt


exm=io.loadmat('concentric.mat')
X=exm['X']
#Normalizing the input
X = X - np.mean(X) / np.std(X)
X.shape


#Given in the PCA problem generate labels since chosing losgistic function
labels=np.zeros((X.shape[1],1))
labels[0:51,0]=0
labels[51:X.shape[1],0]=1
labels.shape



#initialize units for each layer
inp_units=2
hidden_units=3
output_units=1



#initializing the wts and biases randomly
def initialize(inp_units,hidden_units,output_units,labels):
    wt1=np.random.uniform(-1.0,1.0,size=(inp_units,hidden_units))
    b1=np.random.uniform(-1.0,1.0,size=(1,hidden_units))
    wt2=np.random.uniform(-1.0,1.0,size=(hidden_units,output_units))
    b2=np.random.uniform(-1.0,1.0,size=(1,output_units))
    y=labels
    return wt1,b1,wt2,b2,y


def sigmoid(z):
    denom=1.0+np.exp(np.negative(z))
    return (1.0/denom)



def derivative_sigmoid(z):
    return (z*(1-z))


#backprop routine
def backprop(X,inp_units,hidden_units,output_units,labels,epochs,lrn_rate):
    
    wt1,b1,wt2,b2,y=initialize(inp_units,hidden_units,output_units,labels)
    err_conv=[]
    for i in range(0,epochs):
        #Hidden layer calculating pre-acts and acts (forward pass)
        h_pre_act=np.dot(X.T,wt1)+b1
        h_act=sigmoid(h_pre_act)
    
        #Output layer calculating pre-acts and acts (forward pass)
        o_pre_act=np.dot(h_act,wt2)+b2
        o_act=sigmoid(o_pre_act)
    
        #backprop
        o_err=y-o_act
        dO=o_err*derivative_sigmoid(o_act)
    
        h_err=np.dot(dO,wt2.T)
        dh=h_err * derivative_sigmoid(h_act)
        
        #update wts and biases
        wt2=wt2+(np.dot(h_act.T,dO)*lrn_rate)
        b2=b2+(np.sum(dO,axis=0,keepdims=True)*lrn_rate)
        wt1=wt1+(np.dot(X,dh)*lrn_rate)
        b1=b1+(np.sum(dh,axis=0,keepdims=True)*lrn_rate)
    
        #calculate error
        err=np.sum(0.5 * ((o_act-y)**2))
        err_conv.append(err)
        #print(err)
    
    return wt1,wt2,o_act,err_conv


#get the predicted vals from probs and accuracy
def find_acc(preds,labels):
    ypreds=np.zeros((labels.shape[0],1))
    for i in range(labels.shape[0]):
        if preds[i,0]>0.5:
            ypreds[i,0]=1
        else:
            ypreds[i,0]=0
    count=0
    for i in range(labels.shape[0]):
        if ypreds[i,0]==labels[i,0]:
            count=count+1
    acc=count/labels.shape[0]
    return acc
    

wt1,wt2,preds,err_conv=backprop(X,inp_units,hidden_units,output_units,labels,10000,0.03)
acc=find_acc(preds,labels)
print('The accuracy is for 1000 epochs and 0.3 learning rate:',acc)
print('Final Layer 1 wts',wt1)
print('Final Layer 2 wts',wt2)




wt1,wt2,preds,err_conv=backprop(X,inp_units,hidden_units,output_units,labels,100000,0.02)
acc=find_acc(preds,labels)
print('The accuracy is for 100000 epochs and 0.02 learning rate:',acc)
print('Final Layer 1 wts',wt1)
print('Final Layer 2 wts',wt2)




#Convergence graph
plt.plot(err_conv)
plt.title('Convergence Graph')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

