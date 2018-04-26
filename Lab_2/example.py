import numpy as np
import matplotlib.pyplot as plt
from proto2 import *
from sklearn.mixture import log_multivariate_normal_density as log_mv

prondict = {}
prondict['o'] = ['ow']
prondict['z'] = ['z', 'iy', 'r', 'ow']
prondict['1'] = ['w', 'ah', 'n']
prondict['2'] = ['t', 'uw']
prondict['3'] = ['th', 'r', 'iy']
prondict['4'] = ['f', 'ao', 'r']
prondict['5'] = ['f', 'ay', 'v']
prondict['6'] = ['s', 'ih', 'k', 's']
prondict['7'] = ['s', 'eh', 'v', 'ah', 'n']
prondict['8'] = ['ey', 't']
prondict['9'] = ['n', 'ay', 'n']

data = np.load('lab2_data.npz')['data']
example = np.load('lab2_example.npz')['example'].item()
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()

for key in prondict.keys():
    prondict[key] = ['sil'] + prondict[key] + ['sil']

HMM = concatHMMs(phoneHMMs, prondict['o'])

obsloglik = log_mv(data[0]['lmfcc'], HMM['means'], HMM['covars'])
forward_probs = forward(obsloglik, np.log(HMM['startprob']), np.log((HMM['transmat'])))
backward_probs = backward(example['obsloglik'], np.log(HMM['startprob']), np.log((HMM['transmat'])))



#plt.pcolormesh(example['lmfcc'].T)
#plt.show()
#plt.pcolormesh(obsloglik.T)
#plt.show()
plt.pcolormesh(example['logalpha'].T)
plt.show()
plt.pcolormesh(forward_probs.T)
plt.show()
#plt.pcolormesh(example['logbeta'].T)
##plt.show()
#plt.pcolormesh(backward_probs.T)
#plt.show()
