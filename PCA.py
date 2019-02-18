

#PCA Implementation
import numpy as np
import librosa
import matplotlib.pyplot as plt


#audio read
y,fs=librosa.load('s.wav',sr=None)


rand_samples=np.zeros((8,10),dtype=float)
#pick random 8 consecutive samples
samples=np.random.randint(low=0,high=5000000-8,size=10)
#rand_samples
for i in range(10):
    cols = y[samples[i]:(samples[i]+8)]
    rand_samples[:,i] = cols

print(rand_samples.shape)


#Compute Covariance
def cov(Z):
    
    #Calculating the row means
    m=Z.mean(axis=1)
    #Getting the mean matrix 
    
    Mean_mat=m.reshape(len(Z),1)
    D=Z-Mean_mat
    
    
    #Calulating the covariance
    C=(D.dot(np.transpose(D)))/(D.shape[1]-1)
    return C


Acov=cov(rand_samples)
print(Acov.shape)


#power iteration for eigendecomposition 
def power_iter(M):
    
    #intialize a v0 as column vector
    v0=np.random.rand(len(M),1)
    
    #Finding the First eigenvalue and eigenvector
    Mv=M.dot(v0)
    li=np.linalg.norm(Mv)
    vi=Mv/np.linalg.norm(Mv)
    
    #Iterating till the convergence criteria
    while True:
        
        #Dot product
        Mv=M.dot(vi)
        lnew=np.linalg.norm(Mv)
        vnew=Mv/np.linalg.norm(Mv)
        
        #Setting the convergence criteria as difference between new and old eigen value less than 0.001
        if abs(lnew-li)<0.001:
            break
        li=lnew
        vi=vnew
        
    return vnew,lnew
        


#Method of deflation of second vectors
def deflation(evec,evals):
    u=evec
    uT=np.transpose(u)
    temp=evals*u*uT
    return temp


# In[114]:


#Calculating eigenvectors

def perform_eigendecomposition(Acov):
#1st
    evec1,evals1=power_iter(Acov)
    print('First Eigenvalue:',evals1)

#2nd
    B=Acov-deflation(evec1,evals1)
    evec2,evals2=power_iter(B)
    print('Second Eigenvalue:',evals2)

#3rd
    B=Acov-deflation(evec1,evals1)-deflation(evec2,evals2)
    evec3,evals3=power_iter(B)
    print('Third Eigenvalue',evals3)


#4th
    B=Acov-deflation(evec1,evals1)-deflation(evec2,evals2)-deflation(evec3,evals3)
    evec4,evals4=power_iter(B)
    print('Fourth Eigenvalue',evals4)



#5th

    B=Acov-deflation(evec1,evals1)-deflation(evec2,evals2)-deflation(evec3,evals3)-deflation(evec4,evals4)
    evec5,evals5=power_iter(B)
    print('Fifth Eigenvalue',evals5)


#6th

    B=Acov-deflation(evec1,evals1)-deflation(evec2,evals2)-deflation(evec3,evals3)-deflation(evec4,evals4)-deflation(evec5,evals5)
    evec6,evals6=power_iter(B)
    print('Sixth Eigenvalue',evals6)


#7th

    B=Acov-deflation(evec1,evals1)-deflation(evec2,evals2)-deflation(evec3,evals3)-deflation(evec4,evals4)-deflation(evec5,evals5)-deflation(evec6,evals6)
    evec7,evals7=power_iter(B)
    print('Seventh Eigenvalue',evals7)


#8th
    B=Acov-deflation(evec1,evals1)-deflation(evec2,evals2)-deflation(evec3,evals3)-deflation(evec4,evals4)-deflation(evec5,evals5)-deflation(evec6,evals6)-deflation(evec7,evals7)
    evec8,evals8=power_iter(B)
    print('Eight Eigenvalue',evals8)

    W = np.concatenate((evec1, evec2, evec3, evec4, evec5, evec6, evec7, evec8),axis=1)
    WT = W.T
    
    return WT


WT1=perform_eigendecomposition(Acov)

plt.imshow(WT1)
plt.show()



#take 100 random consecutive samples and perform PCA
rand_samples1=np.zeros((8,100),dtype=float)
#pick random 8 consecutive samples
samples=np.random.randint(low=0,high=5000000-8,size=100)
#rand_samples
for i in range(100):
    cols = y[samples[i]:(samples[i]+8)]
    rand_samples1[:,i] = cols

print(rand_samples1.shape)
Xcov=cov(rand_samples1)
WT2=perform_eigendecomposition(Xcov)
plt.imshow(WT2)
plt.show()


#take 1000 random consecutive samples and perform PCA
rand_samples2=np.zeros((8,1000),dtype=float)
#pick random 8 consecutive samples
samples=np.random.randint(low=0,high=5000000-8,size=1000)
#rand_samples
for i in range(100):
    cols = y[samples[i]:(samples[i]+8)]
    rand_samples2[:,i] = cols
print(rand_samples2.shape)


Xcov1=cov(rand_samples2)
WT3=perform_eigendecomposition(Xcov1)
plt.imshow(WT3)
plt.show()

