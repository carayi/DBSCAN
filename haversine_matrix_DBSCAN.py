import numpy.random as rand
import pandas as pd
from scipy.spatial import distance_matrix 
from scipy.spatial.distance import pdist,squareform
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#helper function: calculate distance between coordinates
def haversine_np(point_1, point_2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [point_1[1], point_1[0], point_2[1], point_2[0]])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6367 * c * 1000
    return m

#load data
df = pd.read_csv("files")
#prepare points array
pnts = df[["avg_lat","avg_lon"]].as_matrix()

#calulate distance matrix
test = pdist(pnts,haversine_np)
dist_matrix = squareform(test)

#apply DBSCAN function, distance threshold is 30m
db = DBSCAN(eps=30, min_samples =1, metric='precomputed')
db.fit(dist_matrix)

#plot
plt.scatter(pnts[:,0],pnts[:,1],c=db.labels_.astype(float))
