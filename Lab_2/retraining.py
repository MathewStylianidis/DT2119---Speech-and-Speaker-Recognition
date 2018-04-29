import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import warnings
from proto2 import *
from sklearn.mixture import log_multivariate_normal_density as log_mv



if not sys.warnoptions:
    warnings.simplefilter("ignore")

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

# Retrain model for digit 4 using the hmm corresponding to 4
HMM = concatHMMs(phoneHMMs, prondict['4'])
A = HMM['transmat'][:-1,:-1]
pi = HMM['startprob'][:-1]
# Get log-likelihood for each observation in the utterance 'o'
obsloglik = log_mv(data[10]['lmfcc'], HMM['means'], HMM['covars'])
# Get forward probabilities
forward_probs, obs_seq_log_prob = forward(obsloglik, np.log(pi), np.log(A))
# Print likelihood before training
print("LOG PROBABILITY OF OBSERVATION SEQUENCE BEFORE TRAINING: " + str(obs_seq_log_prob))
HMM['transmat'] = HMM['transmat'][:-1,:-1]
HMM['startprob'] = HMM['startprob'][:-1]
# Estimate mean and cov based on state posteriors and data
HMM_retrained = BaumWelch(data[10]['lmfcc'], HMM)


# Retrain model for digit 4 using the hmm corresponding to 9
HMM = concatHMMs(phoneHMMs, prondict['9'])
A = HMM['transmat'][:-1,:-1]
pi = HMM['startprob'][:-1]
# Get log-likelihood for each observation in the utterance 'o'
obsloglik = log_mv(data[10]['lmfcc'], HMM['means'], HMM['covars'])
# Get forward probabilities
forward_probs, obs_seq_log_prob = forward(obsloglik, np.log(pi), np.log(A))
# Print likelihood before training
print("LOG PROBABILITY OF OBSERVATION SEQUENCE BEFORE TRAINING: " + str(obs_seq_log_prob))
HMM['transmat'] = HMM['transmat'][:-1,:-1]
HMM['startprob'] = HMM['startprob'][:-1]
# Estimate mean and cov based on state posteriors and data
HMM_retrained = BaumWelch(data[10]['lmfcc'], HMM)
