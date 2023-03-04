import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from math import radians, cos, sin, asin, sqrt
import collections
import time
import random
import multiprocessing

class union_find:
    '''
    Custom made union find class to construct disjoint sets via representatives.
    Representative will always be the highest population city in a cluster.
    '''
    def __init__(self, city_ids, populations, adj_list, max_pop_per_city_id):
        
        ## Need indexes to be 0 through length of list for tracking (and efficient storage)
        assert list(range(len(city_ids))) == city_ids
        
        self.root = city_ids
        self.populations = populations
        self.removed = set()
        self.adj_list = adj_list ## only nodes within some fixed distance of eachother are included here
        self.max_pop_per_city_id = max_pop_per_city_id
        self.prev_root = None
    
    def find(self, x):
        if x == self.root[x]:
            return x
        
        ## Path compression optimization
        self.root[x] = self.find(self.root[x])
        return self.root[x]
    
    def union(self, x, y):
        rootx = self.find(x)
        rooty = self.find(y)
        if rootx == rooty:
            return
        
        ## representative x has larger population than representative y
        if self.populations[rootx] >= self.populations[rooty]:
            self.root[rooty] = rootx
        
        ## representative x has smaller population than representative y
        else:
            self.root[rootx] = rooty

    def connected(self, x, y):
        return self.find(x) == self.find(y)
    
    def remove(self, x):
        self.root[x] = x
    
    def set_prev_root(self):
        self.prev_root = self.root.copy()
    
    def check_converged(self):
        
        return self.prev_root == self.root
    
    def check_clustering(self):

        for city_id, representative in enumerate(self.root):
            if self.populations[representative] < self.max_pop_per_city_id[city_id]:
                #print(city_id, representative)
                return False
        return True

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # Radius of earth in miles. Use 6371 for kilometers
    return c * r

def get_adj_list_and_best_representative(max_distance, df):
    '''
    Return:
        i) adj_list: data structure to hold all points within max_distance miles of each point
        ii) max_populations: max population within max_distance of each city
    '''
    records = df.to_records()
    adj_list = {i:[] for i in range(len(records))}
    max_populations = [0]*len(records)
    
    for i in range(len(records)):
        
        id1, lat1, lon1, population1 = records[i]
        for j in range(i, len(records)):
            
            id2, lat2, lon2, population2 = records[j]
            if haversine(lat1, lon1, lat2, lon2) <= max_distance:
                adj_list[i].append(j)
                adj_list[j].append(i)
                max_populations[id1] = max(max_populations[id1], population2)
                max_populations[id2] = max(max_populations[id2], population1)
                
    return adj_list, max_populations

              
def get_clusters(max_iters, adj_list, populations, max_pop_per_city_id):
    uf = union_find(list(adj_list.keys()), populations, adj_list, max_pop_per_city_id)
    for city_id in adj_list:
        
        # each city is unioned with each node in adj list and representative is highest population city
        for neighbor_city_id in adj_list[city_id]:
            uf.union(city_id, neighbor_city_id)
    
    
    iters = 0
    while not uf.check_converged() and iters < max_iters:
        uf.set_prev_root()

        ## now let's prune the overly large clusters
        for city_id, representative in enumerate(uf.root):
            if representative not in adj_list[city_id]:
                uf.remove(city_id)
                uf.removed.add(city_id)
        
        for city_id1 in uf.removed:
            for city_id2 in uf.removed:
                if city_id1 != city_id2 and city_id2 in adj_list[city_id1]:
                    uf.union(city_id1, city_id2)
        uf.removed = set() ## resetting 
        iters += 1

    print(f"Terminated on iteration {iters}")
    return uf

def plot_clusters(df, colors, title):

    fig, ax = plt.subplots(figsize=(20,18))

    worldmap = gpd.read_file(  
        gpd.datasets.get_path("naturalearth_lowres")
    )
    worldmap.plot(color="lightgrey", ax=ax)
    for i, cluster in enumerate(df.new_msa.unique()):
        color = colors[i % len(colors)]
        cluster_df = df[df.new_msa == cluster]
        cluster_df.plot(
            x="longitude", 
            y="latitude", 
            kind="scatter",  
            c=color,
            title=title, 
            ax=ax,
            s = 0.05
        )
        ax.grid(b=True, alpha=0.5)
    
    plt.show()

def create_clean_labels(df, distance_threshold, distance_function):
    '''
    For each MSA, remove the outliers by taking out examples that are hundreds
    of miles away from the median lat lon. Remove all NA msa examples since 
    we won't know if its an MSA or not without manually looking this up. 
    '''
    na_msas = df[df.msa.isna()].copy()
    non_metro_areas = df[(~df.msa.isna()) & 
                         (df.msa.str.contains("NONMETROPOLITAN AREA"))].copy()
    msas = df[(~df.msa.isna()) & 
              (df.msa.str.contains("MSA"))].copy()
    assert len(na_msas) + len(non_metro_areas) + len(msas) == len(df)
    assert (set(na_msas.index).union(
            set(non_metro_areas.index)).union(
            set(msas.index))) == set(df.index)
    print("Examples split into labels and all example accounted for")

    container = [non_metro_areas]
    for msa in msas.msa.unique():
        msa_df = df[df.msa == msa].copy()
        median = msa_df[['latitude','longitude']].median()
        lat, lon = median.latitude, median.longitude
        msa_df['distance_from_median'] = msa_df.apply(lambda x: distance_function(x.latitude, x.longitude, lat, lon), axis=1)
        
        container.append(msa_df[msa_df.distance_from_median < distance_threshold].copy())
        
        #change_msas = msa_df[msa_df.distance_from_median >= distance_threshold].copy()
        #change_msas['msa'] = np.NaN
        #container.append(change_msas)
    
    train_df = pd.concat(container)
    train_df['lat_lon'] = train_df.latitude.astype(str) + \
                          "," + \
                          train_df.longitude.astype(str)
    train_df['label'] = train_df.msa.str.contains("MSA").astype(int)
    return train_df

def get_feature_map(df):
    '''
    Make a feature tracker for each unique latitude longitude combination
    '''
    features = {
        'total_population_1mi':0,
        'total_population_2mi':0,
        'total_population_5mi':0,
        'total_population_10mi':0,
        'total_population_20mi':0,
        'total_population_30mi':0,
        'total_population_50mi':0,
        'total_population_100mi':0,
        
        'total_num_cities_1mi':0,
        'total_num_cities_2mi':0,
        'total_num_cities_5mi':0,
        'total_num_cities_10mi':0,
        'total_num_cities_20mi':0,
        'total_num_cities_30mi':0,
        'total_num_cities_50mi':0,
        'total_num_cities_100mi':0,
        
        'total_num_cities_w_pop_gt_100k_1mi':0,
        'total_num_cities_w_pop_gt_100k_2mi':0,
        'total_num_cities_w_pop_gt_100k_5mi':0,
        'total_num_cities_w_pop_gt_100k_10mi':0,
        'total_num_cities_w_pop_gt_100k_20mi':0,
        'total_num_cities_w_pop_gt_100k_30mi':0,
        'total_num_cities_w_pop_gt_100k_50mi':0,
        'total_num_cities_w_pop_gt_100k_100mi':0,
        
        'total_num_cities_w_pop_gt_50k_1mi':0,
        'total_num_cities_w_pop_gt_50k_2mi':0,
        'total_num_cities_w_pop_gt_50k_5mi':0,
        'total_num_cities_w_pop_gt_50k_10mi':0,
        'total_num_cities_w_pop_gt_50k_20mi':0,
        'total_num_cities_w_pop_gt_50k_30mi':0,
        'total_num_cities_w_pop_gt_50k_50mi':0,
        'total_num_cities_w_pop_gt_50k_100mi':0,

        'total_num_cities_w_pop_gt_10k_1mi':0,
        'total_num_cities_w_pop_gt_10k_2mi':0,
        'total_num_cities_w_pop_gt_10k_5mi':0,
        'total_num_cities_w_pop_gt_10k_10mi':0,
        'total_num_cities_w_pop_gt_10k_20mi':0,
        'total_num_cities_w_pop_gt_10k_30mi':0,
        'total_num_cities_w_pop_gt_10k_50mi':0,
        'total_num_cities_w_pop_gt_10k_100mi':0,
    }
    
    feature_map = {lat_lon:features.copy() for lat_lon in df.lat_lon.unique()}
    
    return feature_map

def increment(col_string, miles_list, increment_value, feature_map, lat_lon):
    '''
    Increment a feature map in place for a specific group of features
    '''

    cols = [f"{col_string}_{mile}mi" for mile in miles_list]
    for col in cols:
        feature_map[lat_lon][col] += increment_value
    return

def calculate_features(df, feature_map): 
    '''
    Given a dataframe with relevant columns and feature_map (dictionary store of data),
    calculate features we will train our model on
    '''

    records = df[['latitude','longitude','population','lat_lon']].to_records()

    for i in range(len(records)):

        id1, lat1, lon1, population1, lat_lon1 = records[i]
        for j in range(i, len(records)):
            
            if i == j:
                increment("total_population", [1,2,5,10,20,30,50,100], population1, feature_map, lat_lon1)
                increment("total_num_cities", [1,2,5,10,20,30,50,100], 1, feature_map, lat_lon1)
                if population1 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [1,2,5,10,20,30,50,100], 1, feature_map, lat_lon1)
                if population1 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [1,2,5,10,20,30,50,100], 1, feature_map, lat_lon1)
                if population1 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [1,2,5,10,20,30,50,100], 1, feature_map, lat_lon1)
                continue
                
            id2, lat2, lon2, population2, lat_lon2 = records[j]
            
            distance = haversine(lat1, lon1, lat2, lon2)
            
            if distance <= 1:
                increment("total_population", [1,2,5,10,20,30,50,100], population2, feature_map, lat_lon1)
                increment("total_population", [1,2,5,10,20,30,50,100], population1, feature_map, lat_lon2)
                increment("total_num_cities", [1,2,5,10,20,30,50,100], 1, feature_map, lat_lon1)
                increment("total_num_cities", [1,2,5,10,20,30,50,100], 1, feature_map, lat_lon2)
                
                if population2 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [1,2,5,10,20,30,50,100], 1, feature_map, lat_lon1)
                if population2 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [1,2,5,10,20,30,50,100], 1, feature_map, lat_lon1)
                if population2 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [1,2,5,10,20,30,50,100], 1, feature_map, lat_lon1)
                if population1 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [1,2,5,10,20,30,50,100], 1, feature_map, lat_lon2)
                if population1 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [1,2,5,10,20,30,50,100], 1, feature_map, lat_lon2)
                if population1 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [1,2,5,10,20,30,50,100], 1, feature_map, lat_lon2)
            
            elif distance <= 2:
                increment("total_population", [2,5,10,20,30,50,100], population2, feature_map, lat_lon1)
                increment("total_population", [2,5,10,20,30,50,100], population1, feature_map, lat_lon2)
                increment("total_num_cities", [2,5,10,20,30,50,100], 1, feature_map, lat_lon1)
                increment("total_num_cities", [2,5,10,20,30,50,100], 1, feature_map, lat_lon2)
                
                if population2 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [2,5,10,20,30,50,100], 1, feature_map, lat_lon1)
                if population2 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [2,5,10,20,30,50,100], 1, feature_map, lat_lon1)
                if population2 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [2,5,10,20,30,50,100], 1, feature_map, lat_lon1)
                if population1 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [2,5,10,20,30,50,100], 1, feature_map, lat_lon2)
                if population1 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [2,5,10,20,30,50,100], 1, feature_map, lat_lon2)
                if population1 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [2,5,10,20,30,50,100], 1, feature_map, lat_lon2)
            elif distance <= 5:
                increment("total_population", [5,10,20,30,50,100], population2, feature_map, lat_lon1)
                increment("total_population", [5,10,20,30,50,100], population1, feature_map, lat_lon2)
                increment("total_num_cities", [5,10,20,30,50,100], 1, feature_map, lat_lon1)
                increment("total_num_cities", [5,10,20,30,50,100], 1, feature_map, lat_lon2)
                
                if population2 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [5,10,20,30,50,100], 1, feature_map, lat_lon1)
                if population2 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [5,10,20,30,50,100], 1, feature_map, lat_lon1)
                if population2 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [5,10,20,30,50,100], 1, feature_map, lat_lon1)
                if population1 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [5,10,20,30,50,100], 1, feature_map, lat_lon2)
                if population1 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [5,10,20,30,50,100], 1, feature_map, lat_lon2)
                if population1 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [5,10,20,30,50,100], 1, feature_map, lat_lon2)
            elif distance <= 10:
                increment("total_population", [10,20,30,50,100], population2, feature_map, lat_lon1)
                increment("total_population", [10,20,30,50,100], population1, feature_map, lat_lon2)
                increment("total_num_cities", [10,20,30,50,100], 1, feature_map, lat_lon1)
                increment("total_num_cities", [10,20,30,50,100], 1, feature_map, lat_lon2)
                
                if population2 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [10,20,30,50,100], 1, feature_map, lat_lon1)
                if population2 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [10,20,30,50,100], 1, feature_map, lat_lon1)
                if population2 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [10,20,30,50,100], 1, feature_map, lat_lon1)
                if population1 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [10,20,30,50,100], 1, feature_map, lat_lon2)
                if population1 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [10,20,30,50,100], 1, feature_map, lat_lon2)
                if population1 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [10,20,30,50,100], 1, feature_map, lat_lon2)
            elif distance <= 20:
                increment("total_population", [20,30,50,100], population2, feature_map, lat_lon1)
                increment("total_population", [20,30,50,100], population1, feature_map, lat_lon2)
                increment("total_num_cities", [20,30,50,100], 1, feature_map, lat_lon1)
                increment("total_num_cities", [20,30,50,100], 1, feature_map, lat_lon2)
                
                if population2 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [20,30,50,100], 1, feature_map, lat_lon1)
                if population2 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [20,30,50,100], 1, feature_map, lat_lon1)
                if population2 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [20,30,50,100], 1, feature_map, lat_lon1)
                if population1 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [20,30,50,100], 1, feature_map, lat_lon2)
                if population1 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [20,30,50,100], 1, feature_map, lat_lon2)
                if population1 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [20,30,50,100], 1, feature_map, lat_lon2)
            elif distance <= 30:
                increment("total_population", [30,50,100], population2, feature_map, lat_lon1)
                increment("total_population", [30,50,100], population1, feature_map, lat_lon2)
                increment("total_num_cities", [30,50,100], 1, feature_map, lat_lon1)
                increment("total_num_cities", [30,50,100], 1, feature_map, lat_lon2)
                
                if population2 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [30,50,100], 1, feature_map, lat_lon1)
                if population2 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [30,50,100], 1, feature_map, lat_lon1)
                if population2 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [30,50,100], 1, feature_map, lat_lon1)
                if population1 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [30,50,100], 1, feature_map, lat_lon2)
                if population1 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [30,50,100], 1, feature_map, lat_lon2)
                if population1 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [30,50,100], 1, feature_map, lat_lon2)
            elif distance <= 50:
                increment("total_population", [50,100], population2, feature_map, lat_lon1)
                increment("total_population", [50,100], population1, feature_map, lat_lon2)
                increment("total_num_cities", [50,100], 1, feature_map, lat_lon1)
                increment("total_num_cities", [50,100], 1, feature_map, lat_lon2)
                
                if population2 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [50,100], 1, feature_map, lat_lon1)
                if population2 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [50,100], 1, feature_map, lat_lon1)
                if population2 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [50,100], 1, feature_map, lat_lon1)
                if population1 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [50,100], 1, feature_map, lat_lon2)
                if population1 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [50,100], 1, feature_map, lat_lon2)
                if population1 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [50,100], 1, feature_map, lat_lon2)
            elif distance <= 100:
                increment("total_population", [100], population2, feature_map, lat_lon1)
                increment("total_population", [100], population1, feature_map, lat_lon2)
                increment("total_num_cities", [100], 1, feature_map, lat_lon1)
                increment("total_num_cities", [100], 1, feature_map, lat_lon2)
                
                if population2 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [100], 1, feature_map, lat_lon1)
                if population2 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [100], 1, feature_map, lat_lon1)
                if population2 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [100], 1, feature_map, lat_lon1)
                if population1 >= 100000:
                    increment("total_num_cities_w_pop_gt_100k", [100], 1, feature_map, lat_lon2)
                if population1 >= 50000:
                    increment("total_num_cities_w_pop_gt_50k", [100], 1, feature_map, lat_lon2)
                if population1 >= 10000:
                    increment("total_num_cities_w_pop_gt_10k", [100], 1, feature_map, lat_lon2)

def get_best_threshold_gmeans(X_val, y_val, clf, roc_curve_fx):
    
    preds = clf.predict_proba(X_val)[:,1]
    fpr, tpr, thresholds = roc_curve_fx(y_val, preds)
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    
    return thresholdOpt

def get_best_threshold_custom(X_val, y_val, clf):
    
    thresholds = np.linspace(0.01,1, 100)
    accuracies = []
    for threshold in thresholds:
        preds_ = clf.predict_proba(X_val)[:,1]
        accuracy = np.mean((preds_ >= threshold).astype(int) == y_val)
        accuracies.append(accuracy)
    return thresholds[np.argmax(accuracies)]