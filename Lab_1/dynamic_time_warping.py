from proto import *

signal_features = np.load("./data/lmfcc_utterance_features.npz")['arr_0']

# Dynamic Time Warping for 'seven' uttered by man and woman
d, LD, AD, path = dtw(signal_features[38], signal_features[16], np.linalg.norm)
#plt.pcolormesh(LD)
#plt.show()
#plt.pcolormesh(AD)
#plt.show()

# Dynamic time warping between all signal_features
global_distances = np.zeros((len(signal_features), len(signal_features)))
for i in range(len(signal_features)):
    for j in range(len(signal_features)):
        d, LD, AD, path = dtw(signal_features[i], signal_features[j], np.linalg.norm)
        global_distances[i, j] = d
        
np.save('./data/global_distance_matrix.npy', global_distances)
plt.pcolormesh(global_distances)
plt.show()
