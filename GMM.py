

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.stats import norm




contents=loadmat('june.mat')
june=contents['june']
june.shape




content=loadmat('december.mat')
dec=content['december']
dec.shape


#disparity matrix
disparity=dec-june
disparity.shape
#disparity=np.delete(disparity,1,1)
#converting matrix to pandas dataframe
disparity=pd.DataFrame(disparity)
disparity.drop([1],axis=1,inplace=True)
disparity.rename(index=str,columns={0:'X'},inplace=True)
#disparity= disparity.astype(np.int16)
#print(disparity)



#Calculate responsibility/ likelihood of each cluster
def cal_resp(wt,prior,k):
    z=np.zeros(shape=(wt.shape[0],k))
    resp=pd.DataFrame(z)
    for j in range(wt.shape[0]):
        for i in range(k):
        #print(wt.iloc[j][i])
            val=prior[i]*wt.iloc[j][i]
        #print(val)
            resp.iloc[j][i]=val
            
    row_sums = resp.sum(axis=1).tolist()
    rs=pd.DataFrame(row_sums)
    
    for j in range(disparity.shape[0]):
        for i in range(k):
            val=resp.iloc[j][i]/rs.iloc[j]
            resp.iloc[j][i]=val
    
    #resp.div(rs.iloc[0], axis='columns')
    #print(rs)
    #print(resp)
    return resp




#M-step
#1. Compute soft counts
def comp_soft_counts(resp,k):
    Nk=[]
    for i in range(k):
        Nk.append(resp[i].sum())
    return Nk


#2. Update mean (taking weighted average)
def update_mean(disparity,resp,Nk,k):
    new_mean=[]
    for i in range(k):
        lst=[]
        for j in range(disparity.shape[0]):
            val=resp.iloc[j][i]*disparity.iloc[j]['X']
            lst.append(val)
        new_mean.append(sum(lst)/Nk[i])
    return new_mean



#3.update variance
def update_variance(means,disparity,resp,Nk,k):
    new_variance=[]
    for i in range(k):
        lst=[]
        for j in range(disparity.shape[0]):
            lst.append(resp.iloc[j][i]*((disparity.iloc[j]['X']-means[i])**2))
        new_variance.append(sum(lst)/Nk[i])
    return new_variance


#4.update prior
def update_prior(disparity,Nk,k):
    prior=[]
    for i in range(k):
        prior.append(Nk[i]/disparity.shape[0])
    return prior


#Convergence loglikelihood
#def check_convergence(disparity,means,variance,k):
    #Z=[]
    #for i in range(k):
        #lst=[]
        #for j in range(disparity.shape[0]):
            #lst.append(((disparity.iloc[j]['X']-means[i])**2)/(2*variance[i]))
        #val=sum(lst)
        #v1=(-0.5)*disparity.shape[0]*math.log((2*math.pi*variance[i]))
        #vt=v1-val
        #Z.append(vt)
    #print(Z)
    #v2=np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))
    #return(v2)


def check_convergence(disparity,new_mean,old_mean,k):
    sum_of_cluster=0
    for i in range(k):
        
        d=new_mean[i]-old_mean[i]
        d=d**2
        #sum_c<-sum_c+d
        sum_of_cluster=sum_of_cluster+d
  
    return(sum_of_cluster)


def compute_gaussian(disparity,means,variance,k):
    z=np.zeros(shape=(disparity.shape[0],k))
    wt=pd.DataFrame(z)
    for i in range(k):
        for j in range(disparity.shape[0]):
            #p=norm(means[i],math.sqrt(variance[i])).pdf(disparity.iloc[j]['X'])
            #norm.pdf(disparity,means[i],math.sqrt(variance[i]))
            p=(1.0/math.sqrt((2.0*math.pi*variance[i])))*math.exp((-1.0*(disparity.iloc[j]['X']-means[i])**2)/(2.0*variance[i]))
            wt.iloc[j][i]=p
        
    return wt



#Initializing the parameters
k=2
threshold=0.0001

#initialize the means, variance and prior
rand_samples=disparity.sample(n=k,replace=True)
means=rand_samples['X'].tolist()
#means=[41,38]
print('Initial Means:',means)

#Computing variance for given means 
#variance=[4,6]
#variance=[]
#for i in range(k):
    #lst=[]
    #for j in range(disparity.shape[0]):
        #lst.append((disparity.iloc[j]['X']-means[i])**2)
    #variance.append((sum(lst)/disparity.shape[0]-1))

variance=np.random.randint(1,10,k).tolist()

#Initializing the priors
prior=[1/k,1/k]
#prior=[0.5429364,0.4570636]

#initialize the matrix by generating the univariate gaussian distribution
wt=compute_gaussian(disparity,means,variance,k)

#First log likelihood of the datapoints
#ll=check_convergence(disparity,means,variance,k)

print('Initial Variance',variance)
print('Initial Prior Weights',prior)
print('Initial Gaussian Distribution for each cluster:',wt)




#Final EM from which all above methods are called
iter=0
max_iter=20
while True:
    #E-step
    resp=cal_resp(wt,prior,k)
    
    #M-step
    
    #Compute the soft counts
    Nk=comp_soft_counts(resp,k)
    
    #Update mean
    old_means=means
    means=update_mean(disparity,resp,Nk,k)
    
    #update variance
    variance=update_variance(means,disparity,resp,Nk,k)
    
    #update prior 
    prior=update_prior(disparity,Nk,k)
    
    #check for convergence
    ll_new=check_convergence(disparity,means,old_means,k)
    #print(ll_new)
    #if ((ll_new-ll))<threshold and ll_new > -np.inf:
    
    if(ll_new<threshold):
        #print(ll_new-ll)
        break
    #ll=ll_new
    
    #recompute the univariate gaussian with new means and variance
    wt=compute_gaussian(disparity,means,variance,k)
    
    iter=iter+1
    print('Iteration:',iter)
    print('Means:',means)

print('Final means',means)
print('Total iters:',iter)


#Assigning points to their clusters
#def em_assign_cluster(disparity,resp,k):
ind=[]
for j in range(disparity.shape[0]):
    ind.append(resp.iloc[j].idxmax(axis=1))
#print(ind)

#Plotting the histogram to check the point distributions
disparity['clusters']=ind

#get the data points within cluster 0
x1=disparity[(disparity.clusters==0)]['X'].tolist()
#y1=np.where(disparity.clusters==0)[0].tolist()
y1=list(range(len(x1)))
#get the data points within clsuter 1
x2=disparity[(disparity.clusters==1)]['X'].tolist()
#y2=np.where(disparity.clusters==1)[0].tolist()
y2=list(range(len(x2)))
        
#plotting the scatterplots
fig = plt.figure()
a = fig.add_subplot(111)

a.scatter(x1,y1, s=10, c='g', marker="s", label='Cluster 0')
a.scatter(x2,y2, s=10, c='b', marker="o", label='Cluster 1')
plt.legend(loc='upper left');
plt.xlabel('Disparity Vals')
plt.ylabel('No. of Points')
plt.title('EM Clusters')
plt.show()

                      





