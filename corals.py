#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas

from sklearn import decomposition 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

import getdata
import plotting

df, benthic_raw, human, species, throphic = getdata.getdata()
# Separate benthic and biomass data (Benth and Biomass --> bab)
bab = df.iloc[:,7:22]   # up to 22 to exclude Large predators

lof = LocalOutlierFactor(n_neighbors=20, p=1, contamination=0.1)
# lof = LocalOutlierFactor()
bab_lof_labels = lof.fit_predict(bab)
# print(biomass_lof_labels)

# print(biomass.shape)
bab_inliers = bab.drop(np.where(bab_lof_labels<0)[0], axis='index', inplace=False) 
# print(biomass.shape)

benthic = bab_inliers.iloc[:,0:6]   
biomass = bab_inliers.iloc[:,6:]   

print(benthic)
print(biomass)
benthic_names = benthic.keys()
biomass_names = biomass.keys()

# # Calculate cumulative biomass on each level
# for i in range(0,len(biomass.keys())):
#     for j in range(0,i):
#         biomass.iloc[i] += biomass.iloc[j]

# Scale the data
scaler_benth = StandardScaler()
scaler_biom = StandardScaler()
scaler_benth.fit(benthic)
scaler_biom.fit(biomass)
# benthic=scaler_benth.transform(benthic)
# biomass=scaler_biom.transform(biomass)

pca_benth = decomposition.PCA (n_components = 5)
pca_biom = decomposition.PCA (n_components = 5)
benthic_pca = pca_benth.fit_transform(benthic)
biomass_pca = pca_biom.fit_transform(biomass)


# Uncomment to decide how many PCA variable you need
# print(pca.explained_variance_ratio_)
print('PC1\tPC2\tPC3\tPC4\tPC5')

for i in range(0,len(pca_benth.explained_variance_ratio_)):
    print("%.2f\t" % (100.0*np.sum(pca_benth.explained_variance_ratio_[0:i+1])),end='',flush=False)
print('')
benthic_pca_df = pd.DataFrame(data=benthic_pca[:,0:2], columns=['PC1', 'PC2'])
benthic_pca_df.insert(loc=0, column='Site', value=df['Site'])

for i in range(0,len(pca_biom.explained_variance_ratio_)):
    print("%.2f\t" % (100.0*np.sum(pca_biom.explained_variance_ratio_[0:i+1])),end='',flush=False)
print('')
biomass_pca_df = pd.DataFrame(data=biomass_pca[:,0:2], columns=['PC1', 'PC2'])
biomass_pca_df.insert(loc=0, column='Site', value=df['Site'])

# This plot is called biplot and it is very useful to understand the PCA results. 
# The length of the vectors it is just the values that each feature/variable has 
# on each Principal Component aka PCA loadings.
plotting.biplot(benthic_pca[:,0:2],pca_benth.components_,names=benthic_names,labels=None)
plotting.biplot(biomass_pca[:,0:2],pca_biom.components_,names=biomass_names,labels=None)
# plt.show()

# ----------------------------------------------------------------------------
# HAC - Hierarchical Agglomerative Clustering
# ----------------------------------------------------------------------------
pca_full = np.vstack((benthic_pca[:,0], biomass_pca[:,0], benthic_pca[:,1], biomass_pca[:,1])).T

import fastcluster as fc
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, inconsistent
from scipy.spatial.distance  import pdist
from sklearn.manifold import TSNE  
from time import time
from mpl_toolkits.mplot3d import Axes3D

method = [ 'single', 'complete', 'average', 'ward' ]

Y = pca_full
t_start = time ()
Y_links = fc.linkage (Y, method = 'ward')
t_hac   = time () - t_start

# Calculate Distance Statistics from Linkage Matrix Y_links []
# ------------------------------------------------------------
#dist_lim = np.quantile (Y_links [:,2], 0.99)
dist_lim = np.quantile (Y_links [:,2], 1)

# Get Clusters for the Linkage Matrix Y_links []
# ------------------------------------------------------------
#Y_labels = fcluster (Y_links, dist_lim, criterion ='distance', depth = 2)
Y_labels = fcluster (Y_links, t = 3, criterion = 'maxclust')

unique_labels, unique_count = np.unique(Y_labels, return_counts = True)

print ('\nPredicted Labels: {}'
       '\n       # Members: {}'.
       format (unique_labels, unique_count)
)

num_clusters = len (np.unique (Y_labels))

plt.figure(figsize=(11.49,8))
sns.scatterplot(
    x = pca_full [:, 0], y = pca_full [:, 1],
    hue     = Y_labels,
    size    = pca_full [:, 0],
    sizes   = (20, 300),
    palette = sns.color_palette ("Set1", num_clusters),
    data = pca_full,
    #legend="full",
    alpha=0.3
)

ax = plt.figure (figsize=(12,8)).gca(projection='3d')
ax.scatter(
    xs = pca_full [:, 0], 
    ys = pca_full [:, 1], 
    zs = pca_full [:, 2], 
    c  = Y_labels, 
    s  = 10 * Y_labels, 
    cmap ='Set1'
)

ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()

Y_labels_full = np.copy (Y_labels)
