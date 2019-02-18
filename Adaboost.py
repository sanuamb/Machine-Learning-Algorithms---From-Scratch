
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


#Import the dataset and check the shape
trn_X=io.loadmat('trX.mat')['trX']
trn_Y=io.loadmat('trY.mat')['trY']
print(trn_X.shape)
print(trn_Y.shape)


# #### Adaboost

#Perceptron learning
#concatenate bias to the data matrix
Zd=np.concatenate((trn_X,np.ones((1,trn_X.shape[1]))),axis=0)
print(Zd.shape)


def ftanh(z):
    #return (np.exp(z)-np.exp(np.negative(z)))/(np.exp(z)+np.exp(np.negative(z)))
    return np.tanh(z)


def derivative_ftanh(z):
    return (1-np.power(z,2))


def perceptron_lrn(Zd,wts,labels,epochs,lrn_rate,wa):
    err_conv=[]
    
    for i in range(1,epochs+1):
        a=np.dot(wts,Zd)
        y_hat=ftanh(a)
        
        error= np.sum(((labels-y_hat)**2)*wa)
        err_conv.append(error)
        
        #print('The error for epoch',i,'=',error)
        delta=(-2.0*(labels-y_hat)*derivative_ftanh(y_hat)*wa)
        wts=wts-(np.dot(delta,Zd.T)*lrn_rate)
        
    return y_hat,wts


def find_acc(preds,labels):
    ypreds=np.zeros((1,labels.shape[1]))
    for i in range(labels.shape[1]):
            ypreds[0,i]=np.sign(preds[0,i])
    count=0
    for i in range(labels.shape[1]):
        if ypreds[0,i]==labels[0,i]:
            count=count+1
    acc1=count/labels.shape[1]
    return ypreds,acc1


#initialize number of learners and wts matrix
m=100
wa=np.random.rand(trn_Y.shape[0],trn_Y.shape[1])
print(wa.shape)


def find_missclass(ypreds,labels):
    I=np.zeros((labels.shape[0],labels.shape[1]))
    for i in range(labels.shape[1]):
        if ypreds[0,i]!=labels[0,i]:
            I[0,i]=1
        else:
            I[0,i]=0
    return I

def find_class(ypreds,labels):
    I=np.zeros((labels.shape[0],labels.shape[1]))
    for i in range(labels.shape[1]):
        if ypreds[0,i]==labels[0,i]:
            I[0,i]=1
        else:
            I[0,i]=0
    return I


def update_beta(wa,Im,Ic):
    #beta=(np.log((1.0-err)/err))
    beta=0.5*np.log(np.sum((wa*Ic)/np.sum((wa*Im))))
    #print(np.sum((wa*Ic)))
    #print(np.sum((wa*Im)))
    return beta


def get_mat(ypreds,labels,beta):
    I=np.zeros((labels.shape[0],labels.shape[1]))
    for i in range(labels.shape[1]):
        if ypreds[0,i]==labels[0,i]:
            I[0,i]=-1
        else:
            I[0,i]=1
    return I


def find_ada_acc(final_preds,labels):
    return np.sum(final_preds[0,:]==labels[0,:])/160


#adaboost algorithm

#1. Randomly initialize the wts
wa=np.ones((trn_Y.shape[0],trn_Y.shape[1]))/160
final_preds=np.zeros((trn_Y.shape[0],trn_Y.shape[1]))

perceptron_wts=[]
store_beta=[]

for i in range(80):
    
    #2. call the mth perceptron learner
    wts=np.random.normal(0,0.5,(trn_X.shape[0]+1))
    preds,wts=perceptron_lrn(Zd,wts,trn_Y,4000,0.001,wa)
    ypreds,acc1=find_acc(preds,trn_Y)
    print('The accuracy is (%): ',acc1*100)
    
    perceptron_wts.append(wts)
    
    #3. Calulate the adaboost error
    
    #find misclassified rate
    Im=find_missclass(ypreds,trn_Y)
    
    #find the classified rate
    Ic=find_class(ypreds,trn_Y)
    
    merr=np.sum(wa*Im)/np.sum(wa)
    #print(merr)
    
    
    
    
    
    #update the beta
    beta=update_beta(wa,Im,Ic)
    store_beta.append(beta)
   
    
    final_preds=np.sign((final_preds+(beta*ypreds)))
    #print(final_preds+(beta*ypreds))
    #print(final_preds)
    
    
    
    #update the weights
    h=get_mat(ypreds,trn_Y,beta)
    wa=wa*np.exp(-beta*trn_Y*ypreds)
    
    
    #print(wa)
    #wa=wa/np.sum(wa)
    
    i=i+1
    

acc=find_ada_acc(final_preds,trn_Y)
print('The total accuracy is',acc*100)
    




x0_metal = trn_X[0, (trn_Y == 1).ravel()]
x1_metal = trn_X[1, (trn_Y == 1).ravel()]
x0_rock = trn_X[0, (trn_Y == -1).ravel()]
x1_rock = trn_X[1, (trn_Y == -1).ravel()]


# #### Construction of Contour Plot


def forward_pass(mesh_X,w,b):
    a=np.dot(w,mesh_X)+b
    y_hat=ftanh(a)
    return y_hat



def predict(mesh_X,w,sample_weights,b):
    yhat = forward_pass(mesh_X,w,b)
    proba = sample_weights * yhat
    Z = np.sum(proba, axis=0, keepdims=True)
    Z /= sample_weights.shape[0]
    return Z



def plot_contour(Zd,wts,sample_weights):
    f1_min, f1_max = Zd[0, :].min() - 0.01, Zd[0, :].max() + 0.02
    f2_min, f2_max = Zd[1, :].min() - 0.01, Zd[1, :].max() + 0.02
    f1, f2 = np.meshgrid(np.arange(f1_min, f1_max, 0.01),
                         np.arange(f2_min, f2_max, 0.01))
    
    mesh_X = np.c_[f1.ravel(), f2.ravel()].T
    print(mesh_X.shape)
    
    #predict values from learned wts and adaboost weights
    w=wts
    b=wts[:,2]
    b=np.expand_dims(b, 1)
    w=np.delete(w,1,1)
    Z = predict(mesh_X,w,sample_weights,b)
    #print(Z.shape)
    
    fig, ax = plt.subplots()
    cs = ax.contourf(f1, f2, Z.reshape(f1.shape),100)
    ax.scatter(x0_metal, x1_metal, c="y", marker="o")
    ax.scatter(x0_rock, x1_rock, c="b", marker="x")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(cs)
    plt.show()


sample_weights=np.array(store_beta)
sample_weights=np.expand_dims(store_beta, 1)
print(sample_weights.shape)


W = np.array(perceptron_wts)
W = np.squeeze(perceptron_wts, 1)
plot_contour(Zd,W,sample_weights)

