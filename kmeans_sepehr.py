#KMEANS from scratch----------
import numpy as np
import matplotlib.pyplot as plt
class My_Kmean():
    def __init__(self , n_clusters):
        #determining the clusters
        self.n = n_clusters
    def fit(self , data):
        #save data as np array type
        data = np.array(data)
        #initializing the centroids
        ind = np.random.choice(len(data) , self.n)
        centers = data[ind]
        #initializing the labels of the data
        labels = np.zeros(len(data))
        #determining max iterations
        max_iter = 300
        for i in range(max_iter):
            for j in range(len(data)):
                #labeling each data
                f1 = (centers - data[j])
                f1 = np.sum(np.abs(f1)**2,axis=1)**(1./2)
                center = np.argmin(f1)
                labels[j] = center
            for m in range(self.n):
                #upfating the centers
                centers[m] = np.mean(data[labels == m] , axis = 0)
                #saving the computed labels and centroids
        self.labels = labels
        self.centers = centers
data = np.vstack([
np.random.normal(-1,0.5,size=(100,2)),
np.random.normal(1,0.5,size=(100,2)),
np.random.normal([2,1.5],0.5,size=(100,2)),
np.random.normal([0,-1.5],0.5,size=(100,2)),
])
kmeans = My_Kmean(n_clusters=4)
kmeans.fit(data)
centroids = kmeans.centers
data0 = data[kmeans.labels == 0]
data1 = data[kmeans.labels == 1]
data2 = data[kmeans.labels == 2]
data3 = data[kmeans.labels == 3]
#the plot after labeling
plt.scatter(data0[:,0] , data0[:,1])
plt.scatter(data1[:,0] , data1[:,1])
plt.scatter(data2[:,0] , data2[:,1])
plt.scatter(data3[:,0] , data3[:,1])
plt.scatter(centroids[:,0] , centroids[:,1])
plt.show()
