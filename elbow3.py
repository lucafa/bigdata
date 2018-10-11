
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


variables = pandas.read_csv('alert.csv')

#X = variables[['timestamp',	'msg',	'proto','src_port','dst_port','ttl','tos','id','dgmlen','iplen','icmptype','icmp_code','icmp_id','icmp_seq']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform( X )

cluster_range = range( 1, 20 )
cluster_errors = []

for num_clusters in cluster_range:
  clusters = KMeans( num_clusters )
  clusters.fit( X_scaled )
  cluster_errors.append( clusters.inertia_ )

  clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

  clusters_df[0:10]