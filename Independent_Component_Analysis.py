
import numpy as np
import matplotlib.pyplot as plt
import librosa
import math

# Input the audios as source 
str1="x_ica_"
str2=".wav"
lst=[]
for i in range(1,21):
    filename=str1+str(i)+str2
    v,fs=librosa.load(filename,sr=None)
    v=v.tolist()
    lst.append(v)
#source matrix
x=np.array(lst)
print(x.shape)    
    

#Eigen decomposition of the coviariance matrix
Xcov=np.cov(x)
evals,evecs=np.linalg.eig(Xcov)


#Dimension Reduction
revals=evals[0:4]
revals
revecs=evecs[:,0:4]
revecs


#Whitening the and getting the whitened reduced matrix 
r=revecs.T
mrevecs=np.zeros((r.shape[0],r.shape[1]))
for i in range(0,len(revals)):
    mrevecs[i]=r[i]/math.sqrt(revals[i])
mrevecs

Z=np.matmul(mrevecs,x)
Z

#Apply ICA
#initialize W random first
W=np.random.rand(4,4)
W

#K*N source matrix we are estimating 
Y=np.matmul(W,Z)
Y.shape
Y.shape[1]

#Updating W
lrn_rate=10**(-7)
fx=np.power(Y,3)
gx=np.tanh(Y)
delta_W=np.matmul(((Y.shape[1]*np.identity(4))-np.matmul(gx,fx.T)),W)
W=W+(lrn_rate*delta_W)
old_sum=delta_W.sum()


#Store the error for convergence graph in list

err_lst=[]
iter=0
while True:
    #update Y according to new W
    Y=np.matmul(W,Z)
    fx=np.power(Y,3)
    gx=np.tanh(Y)
    delta_W=np.matmul(((Y.shape[1]*np.identity(4))-np.matmul(gx,fx.T)),W)
    
    #update W
    W=W+(lrn_rate*delta_W)
    
    err_lst.append(old_sum-delta_W.sum())
    if (np.abs(old_sum)-np.abs(delta_W.sum()))<0.00001:
        break
    old_sum=delta_W.sum()
    print(iter)
    iter=iter+1


#Write the output wave file 
librosa.output.write_wav('ox1.wav',Y[0],fs)
librosa.output.write_wav('ox2.wav',Y[1],fs)
librosa.output.write_wav('ox3.wav',Y[2],fs)
librosa.output.write_wav('ox4.wav',Y[3],fs)
err_lst



#graph of convergence 
x=range(0,len(err_lst))
y=np.abs(err_lst)
plt.plot(x,y)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show

