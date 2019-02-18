import numpy as np
import librosa
from scipy import signal
import scipy.io as io
from scipy.spatial import distance
import scipy.stats
import math


#Read the audio file
audiofile,fs=librosa.load('Piano_Clap.wav',sr=None)
audiofile=audiofile.reshape(len(audiofile),1)
audiofile.shape


#Load MFCC since already given
exm=io.loadmat('mfcc.mat')
mfcc_mat=exm['X']
mfcc_mat.shape


#Load mean vectors and covariance matrix
exm=io.loadmat('MuSigma.mat')
sigma=exm['Sigma']
mean_vec=exm['mX']


# Calculate PDF

#Multivariate Function
def mv_pdf(x,sigma,mvec):
        size=len(x)
        norm_const = 1.0/(math.pow((2*math.pi),size/2) * math.sqrt(np.linalg.det(sigma)))
        xmu= np.matrix(x - mvec)
        inv= np.linalg.inv(sigma)        
        result= math.pow(math.e, -0.5 * (xmu * inv * xmu.T))
        return norm_const * result


pdf=np.zeros((2,mfcc_mat.shape[1]))
for i in [0,1]:
    for j in range(962):
        pdf[i][j]=mv_pdf(mfcc_mat[:,j],sigma[:,:,i],mean_vec[:,i])
pdf.shape


#Normalizing the pdf to recover post probs belongs to two classes
for i in range(2):
        npdf[i,:] = pdf[i,:]/np.sum(pdf,axis=0)
npdf.shape
print(npdf)


#plotting the normalized pdf 
plt.imshow(npdf,aspect='auto')
plt.title('Posterior Probs')
plt.show()


# Since wrong claps in the middle come up with the transition matrix to smoothen them out

# ### Naive Smoothing 



T=np.array([[0.9,0.1],[0,1]])
print(T)



#Naive smoothing
#1. Make a copy of npdf
spdf=npdf.copy()
for j in range(962):
    b=np.argmax(npdf[:,j-1])
    trans=T[b,:]
    spdf[:,j]=trans * npdf[:,j]
print(spdf.shape)

#2. Normalize the smoothed pdf 
for i in range(2):
    spdf[i,:] = spdf[i,:]/np.sum(spdf,axis=0)


#Plotting the naive smoothing post probs
plt.imshow(spdf,aspect='auto')
plt.title('Naive Smoothing')


# The results from naive smoothing is not good since it is still sensitive to claps which can be observed from the above plot. Hence we will execute viterbi algorithm for better results.

# ### Viterbi Algorithm 


T=np.array([[0.9,0.1],[0,1]])
#To keep record of best choices
B=np.zeros(npdf.shape,dtype=int)
pp_viterbi=np.zeros(npdf.shape)

#first post prob remains same
pp_viterbi[:,0]=npdf[:,0]


for i in range(1,962):
    for j in [0,1]:
        #store the best path
        B[j,i] = np.argmax(T[:,j]*npdf[:,i-1])
        #use the best transition prob to cal post prob
        pp_viterbi[j,i] = T[int(B[j,i]),j]*pp_viterbi[int(B[j,i]),i-1]*npdf[j,i]
    #Normalize    
    pp_viterbi[:,i] = pp_viterbi[:,i]/np.sum(pp_viterbi[:,i], axis = 0)
    


# Backtracking


#get hmm sequence 
bseq = np.zeros((2,962))
states=[]
#choose the larger
ind = np.argmax(pp_viterbi[:,961])
#get the state
b = int(B[ind,961])
states.append(b)
bseq[ind,961] = 1
for i in range(1,962):
    bseq[b,961-i] = 1
    b = int(B[b,961-i])
    ind = int(B[b, 961-i])
    states.append(b)


plt.imshow(sequence,aspect='auto')
plt.title('Viterbi Algorithm')


print(pp_viterbi)


print(sequence)


plt.plot(states)
plt.title('Hidden states')
plt.show()

