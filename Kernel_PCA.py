import numpy as np
import random
import scipy.io as io
import matplotlib.pyplot as plt


exm=io.loadmat('concentric.mat')
X=exm['X']
X.shape


#getting kernel matrix by applying kernel function (RBF) on data 
def kernel_func(D,sigma):
    k_mat=np.zeros((D.shape[0],D.shape[0]))
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            k_mat[i,j]=np.exp((-1*(np.linalg.norm(D[i]-D[j])**2))/(sigma**2))
    return k_mat


#generate labels
labels=np.zeros((X.shape[1],1))
labels[0:51,0]=0
labels[51:X.shape[1],0]=1
labels.shape


#Displaying the conecentric circles
plt.scatter(X[0],X[1],c=[{0:'r',1:'b'}[i[0]] for i in labels])
plt.show()


#Perfrom eigendecomposition on the kernel matrix
k_mat=kernel_func(X.T,-0.9)
print(k_mat)
evals,evecs=np.linalg.eig(k_mat)
Z=evecs.real[:,0:3]
Z=Z.T
Z = Z - np.mean(Z) / np.std(Z)
print(Z.shape)


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Z[0,:],Z[1,:],Z[2,:],c=[{0.0:'r',1.0:'b'}[i[0]] for i in labels])
plt.show()


# # Perceptron Learning


#initialize weights randomly and concatenate bias to the input matrix
wts=np.random.rand(1,4)
print(wts)
Zd=np.concatenate((Z,np.ones(shape=(1,Z.shape[1]))),axis=0)
print(Zd)


# In[149]:


def sigmoid(z):
    denom=1.0+np.exp(np.negative(z))
    return (1.0/denom)


# In[150]:


def derivative_sigmoid(z):
    return (z*(1-z))


# In[151]:


def perceptron_lrn(Zd,wts,labels,epochs,threshold,lrn_rate):
    err_conv=[]
    for i in range(1,epochs+1):
        a=np.dot(wts,Zd)
        y_hat=sigmoid(a)
        
        error= np.sum(0.5 * ((y_hat-labels)**2))
        err_conv.append(error)
        if error<threshold:
            break
        
        if i%500==0:
            print('The error for epoch',i,'=',error)
        delta=(y_hat-labels)*derivative_sigmoid(y_hat)
        wts=wts-(np.dot(delta,Zd.T)*lrn_rate)
        
    return y_hat,err_conv


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

preds,err_conv=perceptron_lrn(Zd,wts,labels,100000,0.01,0.1)
acc=find_acc(preds,labels)
print('The accuracy is: ',acc)

#Convergence graph
plt.plot(err_conv)
plt.title('Convergence Graph')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

