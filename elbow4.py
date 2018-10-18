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
#%matplotlib inline

data = pd.read_csv("alert_edited2.csv")
print("******DATA_SET******")
tempo=data['timestamp']
print(tempo)
print(type(tempo))
#Verify missing values present in the data
#print(data.isna().head())

#Get the total number of missing values in both datasets
print(data.isna().sum())
#Replace all NaN values with 1 
data2=data.fillna(1)

#train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Print iplen respect dst port
#print(data2[['dst_port','iplen']].groupby(['dst_port'], as_index=False).mean().sort_values(by='iplen', ascending=False))

#Find right K with elbow method
# create new plot and data
#x1=np.array(data2.drop(['iplen','dgmlen','dst_port'],1).astype(float))
#x2=np.array(data2['dst_port'])

x1=np.array(data2['timestamp'])
x2=np.array(data2['dst_port'])

plt.plot()
matrice_dati = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(matrice_dati)
    kmeanModel.fit(matrice_dati)
    distortions.append(sum(np.min(cdist(matrice_dati, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / matrice_dati.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()




#Kmeans algorithm
#X=np.array(data2.drop(['iplen','dgmlen','dst_port'],1).astype(float))

#X=np.array(data2.drop(['sig generator','sig_id','sig_rev','msg','proto','src','src_port','dst','dst_port','ethsrc','ethdst','ethlen','txpflags','tcpseq','tcppack','tcplen','tcpwindow','ttl','tos','id','dgmlen','iplen','icmptype','icmp code','icmp_id','icmp_seq'],1).astype(float))

X=np.array(data2.drop(['sig generator','sig_id','sig_rev','msg','proto','src','dst','dst_port','ethsrc','ethdst','ethlen','txpflags','tcpseq','tcppack','tcplen','tcpwindow','ttl','tos','id','dgmlen','iplen','icmptype','icmp code','icmp_id','icmp_seq'],1).astype(float))
y=np.array(data2['dst_port'])

kmeans = KMeans(n_clusters=4) # You want cluster the passenger records into 2: Survived or Not survived
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
