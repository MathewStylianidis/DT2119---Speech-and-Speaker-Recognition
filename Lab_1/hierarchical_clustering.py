import numpy as np
import matplotlib.pylab as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from tools import tidigit2labels as getLabels

dataset = np.load('./data/lab1_data.npz')['data']
labels = getLabels(dataset)

D = np.load('./data/global_distance_matrix.npy')
linkage_matrix = linkage(D, method = 'complete')
dendrogram(linkage_matrix, labels = labels)
plt.show()
