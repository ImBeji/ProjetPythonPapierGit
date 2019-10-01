#%%
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

chemin='../ProjetPythonPapierGit/resultats/MatTcEch.mat'
chemin1='../ProjetPythonPapierGit/resultats/MatTeEch.mat'
chemin2='../ProjetPythonPapierGit/resultats/MatTrEch.mat'
#%%Read Data
trunk,label=readData(chemin2)
head,label=readData(chemin1)
elbow,label=readData(chemin)
#%%Isolation Forest to delete outliers for signals Before normalisation: facultatif
#elbow,label=isolationForest(elbow,label)
#trunk,label=isolationForest(trunk,label)
#head,label=isolationForest(head,label)
#%%Normalisation soustraction mean Ligne par ligne normalization1
XElbow = normaData1(elbow)
XTrunk = normaData1(trunk)
XHead = normaData1(head)
#%% Silhouette DTW
wcssElbow,sElbow=Silhouette(XElbow,20)
wcssHead,sHead=Silhouette(XHead,20)
wcssTrunk,sTrunk=Silhouette(XTrunk,20)
#%%plot figure "Silhouette" Trunk Example
maxima=FindMaxima(sTrunk)
del maxima[0]
npx=np.array(sTrunk)
max=argrelextrema(sTrunk, np.greater)
max=list(itertools.chain.from_iterable(max))
numberOfClusters=range(2,20)
plt.figure(figsize=(8, 8))
plt.title("Silhouette Analysis: Trunk")
plt.xticks(np.arange(2, 20, 1.0))
plt.plot(numberOfClusters,s1)
plt.plot(5, np.max(sTrunk),'r*')      
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Silhouette Index')


 #%%Elbow DTW
wcssTr=Elbow(XTrunk,20)
wcssEl=Elbow(XElbow,20)
wcssHe=Elbow(XHead,20)
 #%%plot figure "Elbow" Trunk Example
plt.figure(figsize=(8, 8))
plt.title("Analyse Elbow for Trunk")
plt.xticks(np.arange(2, 19, 1.0))
plt.plot(wcssTr)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');

#%% clustering data DTW
y_predElbow,sdtw_km_el = dtwData(XElbow,5)
y_predTrunk,sdtw_km_tr = sdtwData(XTrunk,5)
y_predHead,sdtw_km_he = sdtwData(XHead,4)
#%% Analyse des résultats post clustering
TEl,effEl=doAnalyse(y_predElbow,label,5,22)
THe,effHe=doAnalyse(y_predHead ,label,4,22)
TTr,effTr=doAnalyse(y_predTrunk,label,5,22)

#%% Distribution clusters riders vs nonriders for elbow: do the same for trunk and head
Riders=TEl[[1,2,6,12,13,16,17,18,19,20],:]
LabelRiders=effEl[[1,2,6,12,13,16,17,18,19,20]]
NRiders=TEl[[0,3,4,5,7,8,9,10,11,14,15],:]
NLabelRiders=effEl[[0,3,4,5,7,8,9,10,11,14,15]]

#%% Cycle distribution between Riders and Non Riders (matrix C)
PostRes=TEl[17:,:]
LabelPre=effEl[0:17]
LabelPost=effEl[17:]
Rc=sumColumn(Riders)
Moy=(Rc/sum(LabelRiders)) *100
Rp=sumColumn(NRiders)
Moy1=(Rp/sum(NLabelRiders)) *100
C=[Moy,Moy1]
C=np.mat(C)
#%%#List of frequencies
my_list =[[1,173,692,1315],	[1,175,710,1346],	[1,175,708,1351],	[1,161,661,1252],	[1,173,706,1341],	
 [1,1,229,476],	[1,206,815,1594],	[1,206,714,1315],	[1,173,688,1320],	[1,168,683,1300],	
 [1,216,715,1304],	[1,173,710,1345],	[1,171,692,1308],	[1,172,704,1338],	[1,170,702,1331],	[1,219,703,1282],
[1,172,706,1348],	[1,172,702,1326],	[1,175,712,1354],	[1,180,720,1369],	[1,216,727,1332],	[1,175,735,1377]]
my_list1 =[[1,173,692,1315,2178],	[1,175,710,1346,2209],	[1,175,708,1351,2228],	[1,161,661,1252,2029],	[1,173,706,1341,2203],	
 [1,1,229,476,685],	[1,206,815,1594,1950],	[1,206,714,1315,2102],	[1,173,688,1320,
2183],	[1,168,683,1300,2152],	
 [1,216,715,1304,2073],	[1,173,710,1345,2219],	[1,171,692,1308,2165],	[1,172,704,1338,
2203],	[1,170,702,1331,2198],	[1,219,703,1282,2040],
[1,172,706,1348,2216],	[1,172,702,1326,2199],	[1,175,712,1354,2232],
[1,180,720,1369,2251],	[1,216,727,1332,2142],	[1,175,735,1377,2245]]
#%% display clusters vs frequencies
n=0
for p in ulabel:   
 
    (inds,) = np.nonzero(label[:,0]==p)
    plt.figure()
    x=np.arange(len(inds))
    y=y_predElbow[inds]+1
    [plt.axvline(x, linewidth=1, color='g') for x in my_list[n]]
    n += 1
    plt.plot(x,y,'g+')
    my_path = os.path.abspath( './ProjetPythonPapier/figures')
    my_file = 'HeadDTW{}.png'
    plt.savefig(os.path.join(my_path, my_file.format(p))) 

 #%% Number of cycles per cluster for elbow riders and nonRiders
 ulabel=np.unique(labelAll[:,0])
 ulabel1=range(1,22)
 zipped=zip(ulabel,ulabel1)
 for i,j in zipped:
    # FindClusFeq(i,j,y_predElbow,label,my_list,my_list1)
    #FindClusFeq(i,j,y_kmeans,label,my_list,my_list1)
     T=FindClusFeq(i,j,y_predElbow,labelAll,my_list,my_list1)
 
 myarray = np.asarray(T1)   
 T1,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22=np.split(myarray,21)
 Tfinal=np.concatenate((T1, T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22), axis=1)   
    
#%% Number of cycles per cluster for riders
      
  ulabelRiders=[4,5,9,16,17,20,21,22,24,25]
  ulabel1Riders=[1,2,6,12,13,16,17,18,19,20]
 
  Riders=T[[1,2,6,12,13,16,17,18,19,20],:]  
  TfinalRiders=np.concatenate((T3,T4,T8,T14,T15,T18,T19,T20,T21,T22), axis=1)     
  Rc=sumLig(TfinalRiders)
  Moy=(Rc/sum(LabelRiders)) *100

   #%%  Number of cycles per cluster for  NonRiders

      TfinalNonRiders=np.concatenate((T1,T5,T6,T7,T9,T10,T11,T12,T13,T15,T16,T17), axis=1)   
      Rc=sumLig(TfinalNonRiders)
      Moy=(Rc/sum(NLabelRiders)) *100
#%% Trunk
N_CLUSTERS = 5
clustersTr = [XTunk[y_predTrunk==i] for i in range(N_CLUSTERS)]
#%%Elbow
N_CLUSTERS = 5
clustersEl = [XElbow[y_predElbow==i] for i in range(N_CLUSTERS)]
#%%Elbow Head
N_CLUSTERS = 4
clustersHe = [XHead[y_predHead==i] for i in range(N_CLUSTERS)]
    #%% 
    plotPH(XElbow,y_predElbow,5,sdtw_km_el)
    plotPH(XHead,y_predHead,4,sdtw_km_he)
    plotPH(XTrunk,y_predTrunk,5,sdtw_km_tr)
    
#%% save model
saveModel(y_predElbow,'modelELBOW.sav')
saveModel(y_predTrunk,'modelTrunk.sav')
saveModel(y_predHead,'modelHead.sav')
