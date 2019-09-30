#%%A
from __future__ import print_function
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from scipy import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
import warnings
from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging, softdtw_barycenter
from tslearn.datasets import CachedDatasets
from tslearn.utils import to_time_series_dataset, check_equal_size, to_time_series
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.metrics import dtw_path, SquaredEuclidean, SoftDTW
from sklearn.ensemble import IsolationForest
from matplotlib.ticker import FuncFormatter
from tslearn.generators import random_walks
from tslearn.metrics import cdist_dtw
import os.path
#%%Read Data
def readData(chemin):
    
    soildata = io.loadmat(chemin)
    soildata.keys()
    soildata_varname = soildata['Resultat']
    data=soildata['Resultat'][:,:-2]
    label = soildata ['Resultat'][:,-2:]
    return(data,label)
#%%Isolation Forest to delete outliers for Tronc Before normalisation
def isolationForest(data):
    ami_preprocessed_train=data
    index = ['Row'+str(i) for i in range(1, len(data)+1)]
    data = pd.DataFrame(data, index=index)
    iforest = IsolationForest(n_estimators=300, contamination=0.05)
    iforest = iforest.fit(data)
    pred_isoF = iforest.predict(data)
    isoF_outliers = Dr[iforest.predict(data) == -1]
    data= data.drop(isoF_outliers.index.values.tolist())
    label=label.drop(isoF_outliers.index.values.tolist())
    return(data.values,label.values)
#%%Normalisation soustraction mean Ligne par ligne normalization1
def normaData1(data):
    data = data - data.mean(axis=1, keepdims=True)
    return(data)
#%%Normalisation soustraction mean All the matrix normalization 2
def normaData2(data):
    m=np.mean(data.ravel())
    data=data-m
    return(data)
#%% Normalisation StandardScaler
def normaData3(data):
    sc_X = StandardScaler()
    data= sc_X.fit_transform(data)
    return(data)
#%%Analyse sILHOUETTE
def Silhouette(X,n_clusters):
    s=np.array([])
    wcss = []
    for n_clusters in range(2,n_clusters):
     kmeans = KMeans(n_clusters, init = 'k-means++', random_state = 42)
     kmeans.fit(X)
     labels = kmeans.labels_
     wcss.append(kmeans.inertia_)
    # centroids = kmeans.cluster_centers_
     r=silhouette_score(X, labels, metric='euclidean')
    #  r=silhouette_samples_memory_saving(X, labels, metric='euclidean')
     s=np.append(s,r)
    return(wcss,s)
    
    
    #%%Analyse Elbow
def Elbow(X,n_clusters):
  wcss = []
  for i in range(1, n_clusters):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
  return(wcss) 
    #%%
 def FindMaxima(numbers):
  maxima = []
  length = len(numbers)
  if length >= 2:
    if numbers[0] > numbers[1]:
      maxima.append(numbers[0])
       
    if length > 3:
      for i in range(1, length-1):     
        if numbers[i] > numbers[i-1] and numbers[i] > numbers[i+1]:
          maxima.append(numbers[i])

    if numbers[length-1] > numbers[length-2]:    
      maxima.append(numbers[length-1])        
  return maxima
#%%
def dtwData(X,n_clusters):
 sdtw_km = TimeSeriesKMeans(n_clusters, metric="softdtw", metric_params={"gamma_sdtw": .01},
                           verbose=True)
 y_pred = sdtw_km.fit_predict(X)
 return y_pred
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
 return(T,eff)
#%% Nombre de clusters: changement de fréquence
def FindClusFeq(ide,i):      
  
   R=[]
   R1=[]
   V=[]
   V1=[]
   j=[];jj=[];x1=[];x2=[]
   for p in [ide]:
    
     (inds,) = np.nonzero(label[:,0]==p)
    
     y=y_kmeans[inds]+1
     x1=my_list[i][:]
     x2=my_list1[i][1:]
     T=np.zeros(shape=(5,len(x1)))
     print(i)
     zipped=zip(x1,x2)
     for v,vv in zipped:
           (yy, eff1) = np.unique(y[v:vv], return_counts=True)
           j.append(eff1)
           jj.append(yy)
        
   for y in range(0,len(x1)):
          A=np.asarray(j[y])
          B=np.asarray(jj[y])
          for ii in range(0,len(A)):
                     s=B[ii]
                     T[s-1,y]=A[ii]
          
   return(np.transpose(T))         
      #%%
def sumColumn(matrix):
    R= np.sum(matrix, axis=0) 
    return R
def sumLig(matrix):
    R= np.sum(matrix, axis=1) 
    return R
#%% save model
def saveModel(model,filename):
    return (pickle.dump(model,open(filename,'wb')) )
