from proto import *

signal_features = np.load("./data/mspec_utterance_features.npz")['arr_0']

# Stack features on top of each other
feature_matrix = signal_features[0]
for i in range(1, len(signal_features)):
    feature_matrix = np.vstack((feature_matrix, signal_features[i]))
correlation_matix = np.corrcoef(feature_matrix, rowvar = False)
covariance_matrix = np.cov(feature_matrix, rowvar = False)
plt.pcolormesh(correlation_matix)
plt.show()
plt.pcolormesh(covariance_matrix)
plt.show()
