#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:17:38 2019

@author: imen
"""

#%%
#%% 
from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from scipy import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os.path
import autograd.numpy as np
import sys
import cupy as cp

import tensorflow as tf
#import cvxpy as cvx

#from qpsolvers import solve_qp

from scipy.sparse import spdiags

from autograd.numpy.linalg import norm as norm
from Utilitaires import algorithm_u_permutations
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from tvkmeans import TVKernelKMeans
from itertools import accumulate
#%% Appel Foncttion


 #%%
#analyse kmeans: matrice de nombre de cycle par cluster pour chaque patient
    
def doAnalyse(y_kmeans,label,nb_clust,nb_parti):
 (ulabel, eff) = np.unique(label[:,0], return_counts=True)
 j=[];jj=[]
 for p in ulabel:  
  (inds,) = np.nonzero(label[:,0]==p)
  x=np.arange(len(inds))
  y=y_kmeans[inds]
  (yy, eff1) = np.unique(y, return_counts=True)
  j.append(eff1)
  jj.append(yy)
 liste=j+jj
 T=np.zeros(shape=(nb_clust,nb_parti))
 for i in range(0,nb_parti):
    A=np.asarray(j[i])
    B=np.asarray(jj[i])
    for ii in range(0,len(A)):
        s=B[ii]
        T[s,i]=A[ii]
 T=np.transpose(T)
 return(T)
  #%%
#analyse kmeans: matrice de nombre de cycle par cluster pour chaque patient
    
def doAnalyse1(y_kmeans,label,nb_clust,nb_parti):
 (ulabel, eff) = np.unique(label, return_counts=True)
 j=[];jj=[]
 for p in ulabel:  
  (inds,) = np.nonzero(label==p)
  x=np.arange(len(inds))
  y=y_kmeans[inds]
  (yy, eff1) = np.unique(y, return_counts=True)
  j.append(eff1)
  jj.append(yy)
 liste=j+jj
 T=np.zeros(shape=(nb_clust,nb_parti))
 for i in range(0,nb_parti):
    A=np.asarray(j[i])
    B=np.asarray(jj[i])
    for ii in range(0,len(A)):
        s=B[ii]
        T[s,i]=A[ii]
 T=np.transpose(T)
 return(T)



#%% Segmentation database

filename_glass = './Datasets/segment.dat'
df_seg = pd.read_csv(filename_glass)
df_seg=df_seg.sort_values(by = 'class')
df_seg=df_seg.to_numpy()
 
data_seg=df_seg[:,:-1]
label_seg = df_seg[:,-1:]

from sklearn.preprocessing import normalize
data_seg=normalize(data_seg, norm='l2', axis=0, copy=True, return_norm=False)
(yy, size) = np.unique(label_seg[:,0], return_counts=True)
ind = [0] + list(accumulate(size))
ts_seg=[data_seg[ind[i]:ind[i+1]] for i in range(len(ind)-1)]
tslab_seg=[label_seg[ind[i]:ind[i+1]] for i in range(len(ind)-1)]



tvkms = TVKernelKMeans(n_clusters=7, random_state=None, kernel="linear", param=None,
                       init_H='zeros',n_init=2, reg_param=1/250, verbose=1, admm_rho=1e1)


resultat_seg=tvkms.fit_predict(ts_seg) 


(ulabel, eff) = np.unique(label_seg[:,0], return_counts=True)




N_CLUSTERS = 7
clusters = [df_seg[resultat_seg== i] for i in range(N_CLUSTERS)]



N_CLUSTERS = 7
clustersOrig = [df_seg[label_seg[:,0]-1== i] for i in range(N_CLUSTERS)]
#compute mean per cluster

  # Premiere Méthode calcul accuracy ( comparaison mean)
meanAll=[]
for i in range(0,N_CLUSTERS):
   mean=np.mean(clusters[i])
   meanAll.append(mean)

meanAllOrig=[]
for i in range(0,N_CLUSTERS):
   mean1=np.mean(clustersOrig[i])
   meanAllOrig.append(mean1)
   
r1, r2= set(meanAll), set(meanAllOrig)
if len(r1.intersection(r2)) == N_CLUSTERS:
   print("Taux de reconnaissance égal à 100%")
else:
   print("passer à la deuxième méthode de calcul")

  # Une autre méthode pour calculer accuracy
  #https://www.rbtechblog.com/blog/cluster_accuracy

import numpy as np
from sklearn.metrics import accuracy_score
y=label_seg[:,0]-1
groups = algorithm_u_permutations(ns=np.unique(resultat_seg), m=np.unique(y).size)
group_accuracy = []

for group in groups:
    tmp = np.full(y.size, np.nan)

    for idx, items in enumerate(group):
        tmp[np.in1d(resultat_seg, items)] = idx

    group_accuracy.append(accuracy_score(y, tmp))

cluster_accuracy = max(group_accuracy)
print("accuracy",cluster_accuracy) 
   
   
   
   
   
   
#afficher les classes de label = 4
# plt.plot(clusters[0])
# plt.plot(clustersOrig[3])

# Trier le résultat obtenu pour faire une comparaison


# resultat_segT=resultat_seg.sort()
# from sklearn.metrics import accuracy_score
# accuracy_score(resultat_seg, label_seg[:,0]-1) # 100% de réussite

#%% digit database
  
filename_digit = './Datasets/optdigits.tra'
data = pd.read_csv(filename_digit)

df_dig=data .sort_values(by = 'class')
df_dig=df_dig.to_numpy()
 
data_dig=df_dig[:,:-1]
label_dig = df_dig[:,-1:]






from sklearn.preprocessing import normalize
data_dig=normalize(data_dig, norm='l2', axis=0, copy=True, return_norm=False)
(yy, size) = np.unique(label_dig[:,0], return_counts=True)
ind = [0] + list(accumulate(size))
ts_dig=[data_dig[ind[i]:ind[i+1]] for i in range(len(ind)-1)]
tslab_dig=[label_dig[ind[i]:ind[i+1]] for i in range(len(ind)-1)]

tvkms = TVKernelKMeans(n_clusters=10, random_state=None, kernel="linear", param=None,
                       init_H='zeros',n_init=2, reg_param=1/250, verbose=1, admm_rho=1e1)


resultat_dig=tvkms.fit_predict(ts_dig) 


(ulabel, eff) = np.unique(label_dig[:,0], return_counts=True)



#afficher les clusters
N_CLUSTERS = 10
clusters = [df_dig[resultat_dig== i] for i in range(N_CLUSTERS)]


clustersOrig = [df_dig[label_dig[:,0]== i] for i in range(N_CLUSTERS)]



# Premiere Méthode calcul accuracy ( comparaison mean)
meanAll=[]
for i in range(0,N_CLUSTERS):
   mean=np.mean(clusters[i])
   meanAll.append(mean)

meanAllOrig=[]
for i in range(0,N_CLUSTERS):
   mean1=np.mean(clustersOrig[i])
   meanAllOrig.append(mean1)
   
   
r1, r2= set(meanAll), set(meanAllOrig)
if len(r1.intersection(r2)) == N_CLUSTERS:
   print("Taux de reconnaissance égal à 100%")
else:
   print("passer à la deuxième méthode de calcul")




 # Une autre méthode pour calculer accuracy
  #https://www.rbtechblog.com/blog/cluster_accuracy
#%%
import numpy as np
from sklearn.metrics import accuracy_score
y=label_dig[:,0]
groups = algorithm_u_permutations(ns=np.unique(resultat_dig), m=np.unique(y).size)
group_accuracy = []

for group in groups:
    tmp = np.full(y.size, np.nan)

    for idx, items in enumerate(group):
        tmp[np.in1d(resultat_dig, items)] = idx

    group_accuracy.append(accuracy_score(y, tmp))

cluster_accuracy = max(group_accuracy)
print("accuracy",cluster_accuracy) 









#%% letter database

filename_letter = './Datasets/letter-recognition.data'
data = pd.read_csv(filename_letter)

df_letter=data.sort_values(by = 'class')

df_letter1=df_letter.to_numpy()

data_letter=df_letter1[:,1:]
label_letter=df_letter1[:,0]


from sklearn.preprocessing import normalize
data_letter=normalize(data_letter, norm='l2', axis=0, copy=True, return_norm=False)
(yy, size) = np.unique(label_letter, return_counts=True)
ind = [0] + list(accumulate(size))
ts_letter=[data_letter[ind[i]:ind[i+1]] for i in range(len(ind)-1)]
tslab_letter=[label_letter[ind[i]:ind[i+1]] for i in range(len(ind)-1)]

tvkms = TVKernelKMeans(n_clusters=26, random_state=10, kernel="linear", param=None,
                       init_H='zeros',n_init=2, reg_param=1/250, verbose=1, admm_rho=1e1)


resultat_letter=tvkms.fit_predict(ts_letter) 


(ulabel, eff) = np.unique(label_letter, return_counts=True)


#afficher les clusters
N_CLUSTERS = 26
clusters = [df_letter1[resultat_letter== i] for i in range(N_CLUSTERS)]

clustersOrig = [df_letter1[label_letter== i] for i in range(N_CLUSTERS)]
# Premiere Méthode calcul accuracy ( comparaison mean)
meanAll=[]
for i in range(0,N_CLUSTERS):
   mean=np.mean(clusters[i])
   meanAll.append(mean)

meanAllOrig=[]
for i in range(0,N_CLUSTERS):
   mean1=np.mean(clustersOrig[i])
   meanAllOrig.append(mean1)
   
   
   
r1, r2= set(meanAll), set(meanAllOrig)
r1.intersection(r2)   
   

if len(r1.intersection(r2)) == N_CLUSTERS:
   print("Taux de reconnaissance égal à 100%")
else:
   print("passer à la deuxième méthode de calcul")
   
   


 # Une autre méthode pour calculer accuracy 
  #https://www.rbtechblog.com/blog/cluster_accuracy
#%%
import numpy as np
from sklearn.metrics import accuracy_score
y=label_letter

groups = algorithm_u_permutations(ns=np.unique(resultat_letter), m=np.unique(y).size)
group_accuracy = []

for group in groups:
    tmp = np.full(y.size, np.nan)

    for idx, items in enumerate(group):
        tmp[np.in1d(resultat_letter, items)] = idx

    group_accuracy.append(accuracy_score(y, tmp))

cluster_accuracy = max(group_accuracy)
print("accuracy",cluster_accuracy) 

#%% letter database a,j

filename_letter = './Datasets/letter-recognition.data'
data = pd.read_csv(filename_letter)

df_letter=data.sort_values(by = 'class')

df_letter1=df_letter.to_numpy()

data_letter=df_letter1[0:7648:,1:]
label_letter=df_letter1[0:7648:,0]


from sklearn.preprocessing import normalize
data_letter=normalize(data_letter, norm='l2', axis=0, copy=True, return_norm=False)
(yy, size) = np.unique(label_letter, return_counts=True)
ind = [0] + list(accumulate(size))
ts_letter=[data_letter[ind[i]:ind[i+1]] for i in range(len(ind)-1)]
tslab_letter=[label_letter[ind[i]:ind[i+1]] for i in range(len(ind)-1)]

tvkms = TVKernelKMeans(n_clusters=10, random_state=10, kernel="linear", param=None,
                       init_H='zeros',n_init=2, reg_param=1/250, verbose=1, admm_rho=1e1)


resultat_letter=tvkms.fit_predict(ts_letter) 


(ulabel, eff) = np.unique(label_letter, return_counts=True)

T,eff=doAnalyse1(resultat_letter,label_letter,10,10)

#afficher les clusters
N_CLUSTERS = 10
clusters = [df_letter1[0:7648:,1:][resultat_letter== i] for i in range(N_CLUSTERS)]
clustersOrig = [df_letter1[0:7648:,1:][label_letter== i] for i in range(N_CLUSTERS)]
#afficher les classes de label = 4
plt.figure()
plt.plot(clusters[0])
plt.plot(clustersOrig[0])

# accuracy

# Premiere Méthode calcul accuracy ( comparaison mean)
meanAll=[]
for i in range(0,N_CLUSTERS):
   mean=np.mean(clusters[i])
   meanAll.append(mean)

meanAllOrig=[]
for i in range(0,N_CLUSTERS):
   mean1=np.mean(clustersOrig[i])
   meanAllOrig.append(mean1)
   
   
   
r1, r2= set(meanAll), set(meanAllOrig)
r1.intersection(r2)   
   

if len(r1.intersection(r2)) == N_CLUSTERS:
   print("Taux de reconnaissance égal à 100%")
else:
   print("passer à la deuxième méthode de calcul")

#%%
import numpy as np
from sklearn.metrics import accuracy_score
y=label_letter

groups = algorithm_u_permutations(ns=np.unique(resultat_dig), m=np.unique(y).size)
group_accuracy = []

for group in groups:
    tmp = np.full(y.size, np.nan)

    for idx, items in enumerate(group):
        tmp[np.in1d(resultat_dig, items)] = idx

    group_accuracy.append(accuracy_score(y, tmp))

cluster_accuracy = max(group_accuracy)
print("accuracy",cluster_accuracy) 
#%% Glass database

filename_glass = './Datasets/glass.data'
df_glass = pd.read_csv(filename_glass)
df_glass=df_glass.sort_values(by = 'Label')



df_glass=df_glass.to_numpy()
 
data_glass=df_glass[:,:-1]
label_glass = df_glass[:,-1:]

from sklearn.preprocessing import normalize
data_glass=normalize(data_glass, norm='l2', axis=0, copy=True, return_norm=False)
(yy, size) = np.unique(label_glass, return_counts=True)
ind = [0] + list(accumulate(size))
ts_glass=[data_glass[ind[i]:ind[i+1]] for i in range(len(ind)-1)]

from tvkmeans import TVKernelKMeans
tvkms = TVKernelKMeans(n_clusters=6, random_state='none', kernel="linear", param=None,
                       init_H='zeros',n_init=2, reg_param=1/250, verbose=1, admm_rho=1e1)


resultat_glass=tvkms.fit_predict(ts_glass) 


(ulabel, eff) = np.unique(label_glass, return_counts=True)

T,eff=doAnalyse1(resultat_glass,label_glass,6,6)


A=label_glass[0:163:,0]-1
B=label_glass[163::,0]-2



N_CLUSTERS = 6
clusters = [df_glass[resultat_glass== i] for i in range(N_CLUSTERS)]

clustersOrig = [df_glass[label_glass== i] for i in range(N_CLUSTERS)]
# Premiere Méthode calcul accuracy ( comparaison mean)
meanAll=[]
for i in range(0,N_CLUSTERS):
   mean=np.mean(clusters[i])
   meanAll.append(mean)

meanAllOrig=[]
for i in range(0,N_CLUSTERS):
   mean1=np.mean(clustersOrig[i])
   meanAllOrig.append(mean1)


r1, r2= set(meanAll), set(meanAllOrig)
r1.intersection(r2)   
   

if len(r1.intersection(r2)) == N_CLUSTERS:
   print("Taux de reconnaissance égal à 100%")
else:
   print("passer à la deuxième méthode de calcul")


#%%
import numpy as np
from sklearn.metrics import accuracy_score
y=label_glass

groups = algorithm_u_permutations(ns=np.unique(resultat_glass), m=np.unique(y).size)
group_accuracy = []

for group in groups:
    tmp = np.full(y.size, np.nan)

    for idx, items in enumerate(group):
        tmp[np.in1d(resultat_glass, items)] = idx

    group_accuracy.append(accuracy_score(y, tmp))

cluster_accuracy = max(group_accuracy)
print("accuracy",cluster_accuracy) 


# formatage label to be in (0,1,2,3,4,5)
#label_glass1=np.concatenate((A,B))
#resultat_glassT=resultat_glass.sort()
#from sklearn.metrics import accuracy_score
#accuracy_score(np.sort(resultat_glass), label_glass1)




#%% wine database
from sklearn.datasets import load_wine
data = load_wine() 
wine=pd.DataFrame(data.data)
wine=wine.to_numpy()
y=pd.DataFrame(data.target)
y=y.to_numpy()


from tvkmeans import TVKernelKMeans
tvkms = TVKernelKMeans(n_clusters=3, random_state=10, kernel="linear", param=None,
                       init_H='zeros',n_init=2, reg_param=1/250, verbose=1, admm_rho=1e1)

from sklearn.preprocessing import normalize
wine=normalize(wine, norm='l2', axis=0, copy=True, return_norm=False)


(yy, size) = np.unique(y[:,0], return_counts=True)
ind = [0] + list(accumulate(size))
ts=[wine[ind[i]:ind[i+1]] for i in range(len(ind)-1)]


#label=[yy[:,1][ind[i]:ind[i+1]] for i in range(len(ind)-1)]


resultatWine=tvkms.fit_predict(ts)
resultatWine1=resultatWine.sort()


N_CLUSTERS = 3
clusters = [wine[resultatWine== i] for i in range(N_CLUSTERS)]

clustersOrig = [wine[y== i] for i in range(N_CLUSTERS)]
# Premiere Méthode calcul accuracy ( comparaison mean)
meanAll=[]
for i in range(0,N_CLUSTERS):
   mean=np.mean(clusters[i])
   meanAll.append(mean)

meanAllOrig=[]
for i in range(0,N_CLUSTERS):
   mean1=np.mean(clustersOrig[i])
   meanAllOrig.append(mean1)


r1, r2= set(meanAll), set(meanAllOrig)
r1.intersection(r2)   
   

if len(r1.intersection(r2)) == N_CLUSTERS:
   print("Taux de reconnaissance égal à 100%")
else:
   print("passer à la deuxième méthode de calcul")


































