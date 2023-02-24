#!/usr/bin/python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas

from sklearn import decomposition 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

import fastcluster as fc
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, inconsistent
from scipy.spatial.distance  import pdist
from sklearn.manifold import TSNE  
from time import time
from mpl_toolkits.mplot3d import Axes3D

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.metrics import accuracy_score

from osgeo import gdal

import getdata
import plotting

df, benthic_raw, human, species, throphic = getdata.getdata()
# Separate benthic and biomass data (Benth and Biomass --> bab)
df_full = df
bab = df.iloc[:,7:22]   # up to 22 to exclude Large predators

lof = LocalOutlierFactor(n_neighbors=20, p=1, contamination=0.1)
# lof = LocalOutlierFactor()
bab_lof_labels = lof.fit_predict(bab)
# print(biomass_lof_labels)

# print(biomass.shape)
bab_inliers = bab.drop(np.where(bab_lof_labels<0)[0], axis='index', inplace=False) 
df.drop(np.where(bab_lof_labels<0)[0], axis='index', inplace=True) 
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
pca_full = np.vstack((benthic_pca[:,0], benthic_pca[:,1], biomass_pca[:,0], biomass_pca[:,1])).T
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

plotting.biplot(benthic_pca[:,0:2],pca_benth.components_,names=benthic_names,labels=Y_labels)
plotting.biplot(biomass_pca[:,0:2],pca_biom.components_,names=biomass_names,labels=Y_labels)

unique_labels, unique_count = np.unique(Y_labels, return_counts = True)

print ('\nPredicted Labels: {}'
       '\n       # Members: {}'.
       format (unique_labels, unique_count)
)

num_clusters = len (np.unique (Y_labels))

# plt.figure(figsize=(11.49,8))
# sns.scatterplot(
#     x = pca_full [:, 0], y = pca_full [:, 1],
#     hue     = Y_labels,
#     size    = pca_full [:, 0],
#     sizes   = (20, 300),
#     palette = sns.color_palette ("Set1", num_clusters),
#     data = pca_full,
#     #legend="full",
#     alpha=0.3
# )

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(
    xs = pca_full [:, 0], 
    ys = pca_full [:, 1], 
    zs = pca_full [:, 2], 
    c  = Y_labels, 
    s  = 10 * Y_labels, 
    cmap ='Set1'
)

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
# plt.show()

Y_labels_full = np.copy (Y_labels)

supervised = df.filter(['Site','Depth', 'Hard coral', 'Macroalgae', 'CCA', 'Turf algae', 'Sand',
       'Complexity','Herbivore (Grazer)', 'Herbivore (Scraper)','Herbivore (Browser)', 'Detritivore (exclusively)', 'Corallivore','Planktivore', 'Small Invert. Feeder', 'Large Invert. Feeder',
       'Small Predator','Latitude','Depth','Population','Effluent','DistCoast', 'DistStream',
       'Population', 'Effluent', 'UrbanIndex', 'PointIndex', 'AgrIndex','FormPlIndex', 'FragIndex', 'DitchIndex'], axis=1)
# supervised = df.filter(['Site','Latitude','Depth','Population','Effluent'], axis=1)
supervised['Label'] = Y_labels_full
supervised.to_csv('supervised.csv', sep = ',', header = True)

print(df.keys())
# ----------------------------------------------------------------------------
# Supervised - DecisionTree
# ----------------------------------------------------------------------------

# y = supervised.iloc[:,-1]
# X = supervised.iloc[:,1:-1]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# model = DecisionTreeRegressor(random_state=44)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)

# # plt.figure(dpi=150)
# # plot_tree(model, feature_names=X.columns)
# # print(tree.export_text(model))
# # plt.savefig('tree.png')
# # plt.close()


# print(accuracy_score(y_test, predictions))

# ----------------------------------------------------------------------------
# Supervised - MLP Classifier
# ----------------------------------------------------------------------------

import warnings
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

y = supervised.iloc[:,-1]
X = supervised.iloc[:,1:-1]

# Scale the data
scaler_x = StandardScaler()
scaler_x.fit(X)
X=scaler_x.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# set up MLP Classifier
mlp = MLPClassifier(    
    hidden_layer_sizes=(20,20),    
    max_iter=1500,    
    alpha=1e-4,    
    tol=1e-5,
    solver="sgd",    
    verbose=True,    
    random_state=1,    
    learning_rate_init=0.1
    )

with warnings.catch_warnings():    
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")    
    mlp.fit(X_train, y_train)

loss_curve = mlp.loss_curve_
# validation_scores = mlp.validation_scores_
epochs = range(mlp.n_iter_)

figlearn = plt.figure()
plt.plot(epochs,loss_curve)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("loss.png")

print(f"Training set score: {mlp.score(X_train, y_train)}")
print(f"Test set score: {mlp.score(X_test, y_test)}")

print(df.keys())

# plt.show()

# import geopandas as gpd
# import matplotlib.pyplot as plt
# import contextily as ctx
# from shapely.geometry import Point
# pd.options.display.max_rows = 10000
# pd.options.display.max_columns = 10000

# states = geopandas.read_file('cb_2018_us_aiannh_500k.shp')

# states = states.to_crs("EPSG:3395")

# hawaii = states[states['NAME'] == 'Hawaii']
# print(hawaii)

# states[states['NAME'] == 'Hawai'].plot(figsize=(12, 12))
# print(states['NAME'])

# plt.show()


# countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
# countries[countries["name"] == "Hawaii"].plot(color="lightgrey")


from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from shapely.geometry import Polygon
from shapely.geometry import Point
import missingno as msno
import os
import wget
import openpyxl
import math
import geopandas as gpd

# filename = wget.download("https://www.ers.usda.gov/media/rbmpu1zi/mapdata2021.xlsx")
df = pd.read_excel(os.getcwd()+'/mapdata2021.xlsx',skiprows=4)
df = df.rename(columns={'Unnamed: 0':'state','Percent':'pct_food_insecure'})

df = df[['state','pct_food_insecure']]

msno.matrix(df)
df = df[df.state.str.len()==2]
# df.pct_food_insecure.hist()

# wget.download("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip")
gdf = gpd.read_file(os.getcwd()+'/cb_2018_us_state_500k')

gdf = gdf.merge(df,left_on='STUSPS',right_on='state')

# gdf.plot()

final_crs = {'init': 'epsg:28992'}
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# NOTE: the convention for polygon points is (Long, Lat)....counterintuitive
# polygon = Polygon([(-175,50),(-175,72),(-140, 72),(-140,50)])
# polygon = Polygon([(-180,0),(-180,90),(-120,90),(-120,0)])

# polygon=hipolygon
# poly_gdf = gpd.GeoDataFrame( geometry=[polygon], crs=world.crs)

# fig, ax1 = plt.subplots(1, figsize=(8, 18))
# world.plot(ax=ax1)
# poly_gdf.boundary.plot(ax = ax1, color="red")
# ax1.set_title("The red polygon can be used to clip Alaska's western islands", fontsize=20)
# ax1.set_axis_off()

fig, ax = plt.subplots(figsize=(6.5, 6.5))
polygon = Polygon([(-180,18),(-180,30),(-150, 30),(-150,18)])
# apply1(alaska_gdf,0,36)
gdf.clip(polygon).plot(ax=ax,color='lightblue', linewidth=0.8, edgecolor='0.8')






amsterdamish = Point((-165, 24))
gdf_am = geopandas.GeoSeries([amsterdamish])
gdf_am.plot(ax=ax,markersize=3)

plt.show()



