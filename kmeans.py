# import statements
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
# create blobs
#data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)

#for n in range(0,49):
#    print data[n]


data = pandas.read_csv('alert2.csv')   
#print(data.head())
# create np array for data points
#points = data[0][0]
#prima riga del dataset
print(data.columns.values)
#exit()
#Numero di celle senza valore
print(data.isna().sum())
data.fillna(data.mean(), inplace=True)
#print(data.isna().sum())
#print(data['msg'].head())
data[['msg', 'ttl']].groupby(['ttl'], as_index=False).mean().sort_values(by='msg', ascending=False)
exit()
print(data['ttl'].head())
#data[['ttl', 'msg']].groupby(['ttl'], as_index=False).mean().sort_values(by='msg', ascending=False)
print(data.info())
X = np.array(data.drop(['msg'], 1).astype(object))
Y = np.array(data['msg'])
kmeans = KMeans(n_clusters=4) # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(X)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == Y[i]:
        correct += 1

print(correct/len(X))

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans.fit(X_scaled)

plt.plot()
plt.xlim([20, 50])
plt.ylim([-2, 5])
plt.title('Dataset')
plt.scatter(X, Y)
plt.show()


exit()
# create scatter plot
plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='viridis')
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.show()