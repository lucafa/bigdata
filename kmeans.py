#!/usr/bin/python
#Convert date in a readable format
#****/
#    /
#    Write by Luca Facchin
#                        /
#                        /
#                        /
#

# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
plt.style.use('seaborn-whitegrid')
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implement kmean algorithm on a csv file created by Snort IDS.')
    parser.add_argument('-i', action='store', dest='file_name', metavar='<file.csv>', help='input file that contains dataset')
    parser.add_argument('-x', action='append', dest='X', metavar='-x <args1> -x <args2> etc', help='value to cluster in x axis, separed by comma')
    parser.add_argument('-y', action='append', dest='y', metavar='-y <args1> -y <args2> etc', help='value to cluster in y axis')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    #parser = argparse.ArgumentParser()
    #args = parser.parse_args()
    #print args.accumulate(args.integers)
    results = parser.parse_args()

file_name=(results.file_name)
x_value=(results.X)
y_value=(results.y)

print(x_value)
#data = pd.read_csv("alert_edited2.csv")
data = pd.read_csv(file_name)

#Take dataset header
data_header=data.head(0)

#Kmeans algorithm
#X=np.array(data2.drop(['iplen','dgmlen','dst_port'],1).astype(float))
data2=data.fillna(1)

#X=np.array(data2.drop(['sig generator','sig_id','sig_rev','msg','proto','src','src_port','dst','dst_port','ethsrc','ethdst','ethlen','txpflags','tcpseq','tcppack','tcplen','tcpwindow','ttl','tos','id','dgmlen','iplen','icmptype','icmp code','icmp_id','icmp_seq'],1).astype(float))
print(data.columns)
X=np.array(data2.drop(['dst_port','ttl','tos','id','dgmlen','iplen'],1).astype(float))
y=np.array(data2['dst_port'])

kmeans = KMeans(n_clusters=2) # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, -1], c=y_kmeans, s=50, cmap='viridis')

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',random_state=None, tol=0.0001, verbose=0)
centers = kmeans.cluster_centers_
print("******CENTROIDI*******")
print(centers)
print("*******Kmeans******")
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    print(prediction[0],y[i])
    if prediction[0] == y[i]:
        correct += 1

print("******RISULTATO********\n")
print(len(X))
print(correct)
print(correct/len(X))
#plt.plot(X,y,'o',color='red');
plt.scatter(centers[:, 0], centers[:, -1], c='black', s=200, alpha=0.5);
plt.show()
exit()
