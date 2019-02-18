
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


contents=loadmat('june.mat')
june=contents['june']
june.shape

content=loadmat('december.mat')
dec=content['december']
dec.shape


#disparity matrix
disparity=dec-june
#Histogram of the disparity matrix
plt.hist(disparity[:,0])
plt.show()


#converting matrix to pandas dataframe
disparity=pd.DataFrame(disparity)
disparity.drop([1],axis=1,inplace=True)
disparity.rename(index=str,columns={0:'X'},inplace=True)


k=2
threshold=0.0001
#Initialize centriods randomly
rand_samples=disparity.sample(n=k,replace=True)
centriod=rand_samples['X'].tolist()
#centriod=rand_samples.T
#centriod.rename(index=str,columns={centriod.columns[0]:'C1',centriod.columns[1]:'C2'},inplace=True)
#centriod=[41,38]


#Initially randomly assign datapoints to k clusters
labels=np.random.choice(k,size=disparity.shape[0],replace=True).tolist()
disparity['labels']=labels


#Calculate distance function
def cal_dist(disparity,centriod):
    distance_matrix=pd.DataFrame()
    lst1=[]
    for i in range(len(centriod)):
            lst=[]
            for indx,row in disparity.iterrows():
                val=math.sqrt((row['X']-centriod[i])**2)
                lst.append(val)
            lst1.append(lst)
    distance_matrix=pd.DataFrame(lst1)
    print(distance_matrix)
    return distance_matrix

            

#Assign new clusters
def assign_clusters(disparity,distance_matrix):
    indices=[]
    for col in distance_matrix:
        indices.append(distance_matrix[col].idxmin())
    disparity['labels']=indices
    return disparity


#update centriod
def update_centriod(disparity,centriod):
    new_centriod=[]
    for i in range(len(centriod)):
        lst2=[]
        lst2=disparity[disparity['labels']==i]['X'].tolist()
        val=round((sum(lst2)/len(lst2)),0)
        new_centriod.append(val)
    #print(new_centriod)
    return new_centriod
        


#Calculate SSE
def cal_SSE(new_centriod,old_centriod):
    e1=[x1-x2 for(x1,x2) in zip(old_centriod,new_centriod)]
    e1=map(lambda x: x ** 2, e1)
    error=math.sqrt(sum(e1))
    return error


iter=0
distance_matrix=cal_dist(disparity,centriod)
print('Initial means:',centriod)
while True:
    #Assign clusters
    disparity=assign_clusters(disparity,distance_matrix)
    #Update Centriods
    new_centriods=update_centriod(disparity,centriod)
    #Check for convergence
    if(cal_SSE(new_centriods,centriod)<threshold):
        break
    #Calculate the distance between new centriod and the data
    centriod=new_centriods
    distance_matrix=cal_dist(disparity,centriod)
    iter=iter+1
print("Final Centriods:",centriod)
print("Number of iterations required:",iter)



#Displaying the clusters
#get the data points within cluster 0
x1=disparity[(disparity['labels']==0)]['X'].tolist()
#y1=np.where(disparity.clusters==0)[0].tolist()
y1=list(range(len(x1)))
#get the data points within clsuter 1
x2=disparity[(disparity['labels']==1)]['X'].tolist()
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
plt.title('K-means Clusters')
plt.show()


#Taking the standard deviation of each cluster
std1=np.std(x1)
print('Standard Deviation of Cluster 0',std1)
std2=np.std(x2)
print('Standard Deviation of Cluster 1',std2)

