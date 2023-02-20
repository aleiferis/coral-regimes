#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

import getdata
import plotting

df, benthic_raw, human, species, throphic = getdata.getdata()
# Separate benthic and biomass data (Benth and Biomass --> bab)
benthic = df.iloc[:,7:13]   
biomass = df.iloc[:,13:22]   # up to 22 to exclude Large predators
benthic_names = benthic.keys()
biomass_names = biomass.keys()

lof = LocalOutlierFactor(n_neighbors=20, p=1, contamination=0.3)
# lof = LocalOutlierFactor()
biomass_lof_labels = lof.fit_predict(biomass)
print(biomass_lof_labels)

print(biomass.shape)
biomass.drop(np.where(biomass_lof_labels<0)[0], axis='index', inplace=True) 
print(biomass.shape)

# Scale the data
scaler_benth = StandardScaler()
scaler_biom = StandardScaler()
scaler_benth.fit(benthic)
scaler_biom.fit(biomass)
benthic=scaler_benth.transform(benthic)
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

# print(pca_benth_df)



# plt.scatter(pca_benth_df.iloc[:,1], pca_benth_df.iloc[:,2])


