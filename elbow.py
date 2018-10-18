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

data = pd.read_csv("alert_edited2.csv")
print("******DATA_SET******")
tempo=data['timestamp']

#Get the total number of missing values in both datasets
print(data.isna().sum())
#Replace all NaN values with 1 
data2=data.fillna(1)

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