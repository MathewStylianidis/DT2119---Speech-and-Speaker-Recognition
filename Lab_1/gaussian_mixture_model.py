import numpy as np
import matplotlib.pylab as plt
import matplotlib.pylab as plt
from sklearn.mixture import GaussianMixture
import proto



#READ FEATURES FROM FILE
signal_features = np.load("./data/lmfcc_utterance_features.npz")['arr_0']
features = signal_features[0]
for idx, sample_features in enumerate(signal_features):
    if(idx > 0):
        features = np.vstack((features, sample_features))

component_number = 32
utterances = [signal_features[16], signal_features[17], \
                signal_features[38], signal_features[39]]

model = GaussianMixture(component_number)
model.fit(features)
cluster_labels = []
posteriors = []
for utterance in utterances:
    cluster_labels.append(list(model.predict(utterance)))
    posteriors.append(model.predict_proba(utterance))


plt.pcolormesh(posteriors[0].T)
plt.savefig('PosteriorMan7_1_' + str(component_number) + '.png')
plt.clf()
plt.pcolormesh(posteriors[1].T)
plt.savefig('PosteriorMan7_2_' + str(component_number) + '.png')
plt.clf()
plt.pcolormesh(posteriors[2].T)
plt.savefig('PosteriorWoman7_1_' + str(component_number) + '.png')
plt.clf()
plt.pcolormesh(posteriors[3].T)
plt.savefig('PosteriorWoman7_2_' + str(component_number) + '.png')
plt.clf()
