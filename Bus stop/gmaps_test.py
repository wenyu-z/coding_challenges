# -*- coding: utf-8 -*-
"""
Created on Tue May 02 09:34:15 2017

@author: WZhao10
"""

"""
import googlemaps

home = 'Dairy Ashford Rd @ Briar Forest Dr'
work = '1430 Enclave Pkwy, Houston TX'

gClient = googlemaps.Client(key="key at https://console.cloud.google.com/apis/dashboard")
geocode_result = gClient.geocode(home)

lat = geocode_result[0]['geometry']['location']['lat']
lng = geocode_result[0]['geometry']['location']['lng']

matrix = gClient.distance_matrix(home, work, mode='walking')


#%%
import gmaps

m = gmaps.Map()

layer = gmaps.symbol_layer([(29.753806, -95.605500)], 
                            fill_color="green", stroke_color="green", scale=2)

m.add_layer(layer)

m
"""

#%%
import googlemaps
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier as knn

#%%
"""
gClient = googlemaps.Client(key="key at https://console.cloud.google.com/apis/dashboard")

def add2latlng(add):
    geocode_result = gClient.geocode(add)
    if len(geocode_result)>0:
        lat = geocode_result[0]['geometry']['location']['lat']
        lng = geocode_result[0]['geometry']['location']['lng']
    else:
        lat = np.nan
        lng = np.nan
    return {'lat':lat, 'lng':lng}
    
df_bus = pd.read_csv('Potentail_Bust_Stops.csv')
df_add = pd.read_csv('Employee_Addresses.csv')

df_bus['address'] = ['@'.join((df_bus.loc[k, 'Street_One'], df_bus.loc[k, 'Street_Two'])) for k in df_bus.index]
latlng_bus = [add2latlng(k) for k in df_bus['address']]
latlng_add = [add2latlng(k) for k in df_add['address']]
              
df_bus = pd.concat([df_bus, pd.DataFrame(latlng_bus)], axis=1)
df_add = pd.concat([df_add, pd.DataFrame(latlng_add)], axis=1)
"""

#%%

df_bus = pd.read_csv('Potentail_Bust_Stops_new.csv')
df_add = pd.read_csv('Employee_Addresses_new.csv')

plt.plot(df_add['lat'], df_add['lng'], 'g.')
plt.plot(df_bus['lat'], df_bus['lng'], 'rx')


#%%

def point_dist(point1, point2):
    sum_of_squares = 0
    for k in range(len(point1)):
        error = point1[k] - point2[k]
        square = error**2
        sum_of_squares += square
    return sum_of_squares

def centroid_dist(data_subset, centroid):
    sum_of_dist = 0
    for k in data_subset:
        dist = point_dist(k, centroid)
        sum_of_dist += dist
    return sum_of_dist

def nearest_centroid_point(point, centroids, selected_index):
    dist = 1e99
    no_centroid = None
    for ind in selected_index:
        dist_new = point_dist(point, centroids[ind])
        if dist_new < dist:
            dist = dist_new
            no_centroid = ind
    return no_centroid
    
    
def nearest_centroid_data(data_subset, centroids, selected_index):
    dist = 1e99
    no_centroid = None
    for ind in selected_index:
        dist_new = centroid_dist(data_subset, centroids[ind])
        if dist_new < dist:
            dist = dist_new
            no_centroid = ind
    return no_centroid
        
def assign_membership(data, centroids, selected_index):
    membership = np.array([nearest_centroid_point(point, centroids, selected_index) for point in data])
#    selected_index = np.unique(membership)
    return membership
    
def update_centroids(data, selected_index, membership, centroids):
    selected_index_new = []
    for k in selected_index:
        membership_index = np.where(membership == k)[0]
        data_cluster = data[membership_index, :]
        new_k = nearest_centroid_data(data_cluster, centroids, range(len(centroids)))
        selected_index_new.append(new_k)
    selected_index_new = np.array(selected_index_new)
    return selected_index_new

def total_distance(data, membership, centroids, selected_index):
    total_dist = 0
    for ind in selected_index:
        membership_index = np.where(membership == ind)[0]
        data_cluster = data[membership_index, :]
        dist = centroid_dist(data_cluster, centroids[ind])
        total_dist += dist
        
    return total_dist
    
#%%

data = df_add[['lat', 'lng']].values
all_centroids = df_bus[['lat', 'lng']].values

final_total_dist = 1e99
final_membership = []
final_index = []

for i in range(100):
    init_index = np.random.randint(len(all_centroids), size=10)
    
    membership = assign_membership(data, all_centroids, init_index)
    total_dist = total_distance(data, membership, all_centroids, init_index)
    
    selected_index = init_index
    step = 0; sigma = 1e99
    
    while (step < 100) & (sigma > 10e-12):
        selected_index = update_centroids(data, selected_index, membership, all_centroids)
        membership = assign_membership(data, all_centroids, selected_index)
        total_dist_new = total_distance(data, membership, all_centroids, selected_index)
        
        sigma = total_dist - total_dist_new
        
        total_dist = total_dist_new
        
        step += 1
    
    if total_dist < final_total_dist:
        final_total_dist = total_dist
        final_membership = membership
        final_index = selected_index
        print "{0}: {1}".format(i, final_total_dist)

#%%


#%%

n_clusters = range(1, 11)
inertias = []
for nc in n_clusters:
    cluster = KMeans(n_clusters=nc)
    cluster.fit(data)
    inertias.append(cluster.inertia_)

plt.figure(figsize=(12,6))
plt.plot(range(1,11), inertias, '.-'); plt.xlabel('No. of Clusters'); plt.ylabel('Inertia')

#%%

final_total_dist_km = 1e99
final_membership_km = []
final_index_km = []

for i in range(100):
    km_cluster = KMeans(n_clusters=10)
    cluster_membership = km_cluster.fit_predict(data)
    
    index_km = []
    for k in range(10):
        membership_index = np.where(cluster_membership == k)[0]
        data_cluster = data[membership_index, :]
        centroid_select = nearest_centroid_data(data_cluster, all_centroids, range(len(all_centroids)))
        index_km.append(centroid_select)
    
    centroid_membership = np.array([index_km[k] for k in cluster_membership])
    total_dist_km = total_distance(data, centroid_membership, all_centroids, index_km)
    
    if total_dist_km < final_total_dist_km:
        final_total_dist_km = total_dist_km
        final_membership_km = cluster_membership
        final_index_km = index_km
        print "{0}: {1}".format(i, final_total_dist_km)



#%%

plt.figure(figsize=(12,9))
plt.scatter(data[:,0], data[:,1], c=final_membership)
plt.scatter(all_centroids[final_index,0], all_centroids[final_index,1], color='r', marker = 'x', s=50, linewidths=3)
plt.xlabel('Latitude'); plt.ylabel('Longtitude')

#%%







