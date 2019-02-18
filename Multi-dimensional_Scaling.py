import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt


exm=io.loadmat('MDS_pdist.mat')
dist_mat=exm['L']
dist_mat.shape




#steps for MDS

#average along rows and subtract it from original dist
mr=dist_mat-dist_mat.mean(axis=1)
#average of mr across cols
mc=mr.mean(axis=0)
#subtract mr and mc to get w
w=mr-mc

#eigendecomposition on w
evals,evecs=np.linalg.eig(w)


#Recovering the original src 
vec = np.diag(np.sqrt(np.abs(evals)))
recov = np.dot(evecs,vec)
recov=recov.real[:,0:2]


#theta and sine and cosine
t = np.radians(360)
cosine, sine = np.cos(t), np.sin(t)

# Creating the rotation matrix.
R = np.array([[cosine,sine], [-sine,cosine]])

rotated_matrix =  np.inner(recov,R)
plt.scatter(rotated_matrix[:,1],rotated_matrix[:,0])
plt.show

