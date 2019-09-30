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
chemin='./ProjetPythonPapier/Resultats/MatTcEch.mat'
chemin1='./ProjetPythonPapier/Resultats/MatTeEch.mat'
chemin2='./ProjetPythonPapier/Resultats/MatTrEch.mat'
#%%Read Data
trunk,label=readData(Chemin2)
head=readData(Chemin1)
elbow=readData(Chemin)
#%%Isolation Forest to delete outliers for Coude Before normalisation
elbow,label=isolationForest(elbow)
trunk=isolationForest(trunk)
head=isolationForest(head)
#%%Normalisation soustraction mean Ligne par ligne normalization1
XElbow = normaData1(elbow)
XTrunk = normaData1(trunk)
XHead = normaData1(head)
#%% Silhouette DTW
wcssElbow,sElbow=Silhouette(XElbow,20)
wcssHead,sHead=Silhouette(XHead,20)
wcssTrunk,sTrunk=Silhouette(XTrunk,20)
#%%plot figure "Silhouette" Trunk
maxima=FindMaxima(s1)
del maxima[0]
npx=np.array(s1)
max=argrelextrema(s1, np.greater)
max=list(itertools.chain.from_iterable(max))
numberOfClusters=range(2,20)
plt.figure(figsize=(8, 8))
plt.title("Silhouette Analysis: Trunk")
plt.xticks(np.arange(2, 20, 1.0))
plt.plot(numberOfClusters,s1)
plt.plot(5, np.max(s1),'r*')      
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Silhouette Index')


 #%%Elbow DTW
wcssTr=Elbow(XTrunk,20)
wcssEl=Elbow(XElbow,20)
wcssHe=Elbow(XHead,20)
 #%%plot figure "Elbow" Trunk
plt.figure(figsize=(8, 8))
plt.title("Analyse Elbow for Trunk")
plt.xticks(np.arange(2, 19, 1.0))
plt.plot(wcssTr)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');

#%% clustering data DTW
y_predElbow = dtwData(XCoude,5)
y_predTrunk = sdtwData(XTrunk,5)
y_predHead = sdtwData(XHead,4)
#%% Analyse des résultats post clustering
TEl,effEl=doAnalyse(y_predElbow,label,5,22)
THe,effHe=doAnalyse(y_predHead ,label,4,22)
TTr,effTr=doAnalyse(y_predTrunk,label,5,22)

#%% Distribution clusters riders vs nonriders 
Riders=T[[1,2,6,12,13,16,17,18,19,20],:]
LabelRiders=eff[[1,2,6,12,13,16,17,18,19,20]]
NRiders=T[[0,3,4,5,7,8,9,10,11,14,15],:]
NLabelRiders=eff[[0,3,4,5,7,8,9,10,11,14,15]]

#%% 
PostRes=T[17:,:]
LabelPre=eff[0:17]
LabelPost=eff[17:]
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
    y=y_predHead[inds]+1
    [plt.axvline(x, linewidth=1, color='g') for x in my_list[n]]
    n += 1
    plt.plot(x,y,'g+')
    my_path = os.path.abspath( './ProjetPythonPapier/Figures')
    my_file = 'HeadDTW{}.png'
    plt.savefig(os.path.join(my_path, my_file.format(p))) 

 #%% Number of cycles per cluster
      T1=FindClusFeq(1,0)
      T3=FindClusFeq(4,1)
      T4=FindClusFeq(5,2)
      T5=FindClusFeq(6,3)
      T6=FindClusFeq(7,4)
      T7=FindClusFeq(8,5)
      T8=FindClusFeq(9,6)
      T9=FindClusFeq(11,7)
      T10=FindClusFeq(12,8)
      T11=FindClusFeq(13,9)
      T12=FindClusFeq(14,10)
      T13=FindClusFeq(15,11)
      T14=FindClusFeq(16,12)
      T15=FindClusFeq(17,13)
      T16=FindClusFeq(18,14)
      T17=FindClusFeq(19,15)
      T18=FindClusFeq(20,16)
      T19=FindClusFeq(21,17)
      T20=FindClusFeq(22,18)
      T21=FindClusFeq(24,19)
      T22=FindClusFeq(25,20)
      Tfinal=np.concatenate((T1, T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22), axis=1)
#%% Number of cycles per cluster for riders
      Riders=T[[1,2,6,12,13,16,17,18,19,20],:]    
      T3=FindClusFeq(4,1)
      T4=FindClusFeq(5,2)
      T8=FindClusFeq(9,6)
      T14=FindClusFeq(16,12)
      T15=FindClusFeq(17,13)
      T18=FindClusFeq(20,16)
      T19=FindClusFeq(21,17)
      T20=FindClusFeq(22,18)
      T21=FindClusFeq(24,19)
      T22=FindClusFeq(25,20)
      TfinalRiders=np.concatenate((T3,T4,T8,T14,T15,T18,T19,T20,T21,T22), axis=1)     
      Rc=sumLig(TfinalRiders)
      Moy=(Rc/sum(LabelRiders)) *100
   #%%  Number of cycles per cluster for  NonRiders
      T1=FindClusFeq(1,0)
      T5=FindClusFeq(6,3)
      T6=FindClusFeq(7,4)
      T7=FindClusFeq(8,5)
      T9=FindClusFeq(10,7)
      T10=FindClusFeq(12,8)
      T11=FindClusFeq(13,9)
      T12=FindClusFeq(14,10)
      T13=FindClusFeq(15,11)
      T16=FindClusFeq(18,14)
      T17=FindClusFeq(19,15)
      TfinalNonRiders=np.concatenate((T1,T5,T6,T7,T9,T10,T11,T12,T13,T15,T16,T17), axis=1)   
      Rc=sumLig(TfinalNonRiders)
      Moy=(Rc/sum(NLabelRiders)) *100

#%%
Rc=sumLig(Tfinal)
Moy=(Rc/sum(effEl)) *100
Rp=sumColumn(PostRes)
Moy1=(Rp/sum(LabelPost)) *100
C=[Moy,Moy1]
C=np.mat(C)
#%% 
N_CLUSTERS = 5
clusters = [XTronc[y_predTrunk==i] for i in range(N_CLUSTERS)]
#%%Elbow
N_CLUSTERS = 5
clusters = [XCoude[y_predElbow==i] for i in range(N_CLUSTERS)]
#%%Elbow Head
N_CLUSTERS = 4
clusters = [XCoude[y_predElbow==i] for i in range(N_CLUSTERS)]
#%% figure Intervalle de confiance PH par cluster
for i, c in enumerate(clusters):
    clust=clusters[i]
    clustPh=clust
    MoyCol=sdtw1.cluster_centers_[i].ravel()
    stdCol=np.std(clustPh, 0)
    MoyP=MoyCol-stdCol
    MoyM=MoyCol+stdCol
    plt.figure()
    plt.plot(MoyP,'--', label="Mean - std")
    plt.plot(MoyM,'--', label="Mean + std")
    #plt.plot(MoyCol, label="Mean")
    plt.plot(MoyCol, "r-", label="Mean")
    plt.legend()
    plt.xlabel('Cycle Length')
    plt.ylabel('Confidence Interval')
    s1='Cluster '
    s1+=str(i+1)
    s=' : NUmber of cycle is= '
    s +=str(h)
    my_file = 'IntervalleDTWEl{}.png'
    outfile = os.path.abspath( './ProjetPythonPapier/Figures')
    plt.savefig(os.path.join(outfile, my_file.format(i))) 
#%% save model
saveModel(y_predElbow,'modelELBOW.sav')
saveModel(y_predTrunk,'modelTrunk.sav')
saveModel(y_predHead,'modelHead.sav')
