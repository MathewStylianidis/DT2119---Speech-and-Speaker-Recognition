import numpy as np
import matplotlib.pyplot as plt
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

keys = list(prondict.keys())

data = np.load('lab2_data.npz')['data']
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()

phoneConcatHMMs = []
for key in keys:
    prondict[key] = ['sil'] + prondict[key] + ['sil']
    phoneConcatHMMs.append(concatHMMs(phoneHMMs,prondict[key]))

print("USING FORWARD ALGORITHM DERIVED MAXIMUM LIKELIHOOD PREDICTIONS")
for key in keys:
    # Get  observation sequences for given key
    sequences = [x['lmfcc'] for x in data if x['digit'] == key ]
    for idx, sequence in enumerate(sequences):
        maxProb = None
        maxProbIdx = None
        for i in range(len(phoneConcatHMMs)):
            obsloglik = log_mv(sequence, phoneConcatHMMs[i]['means'], phoneConcatHMMs[i]['covars'])
            # Get forward probabilities
            forward_probs, obs_seq_log_prob = forward(obsloglik, \
                np.log(phoneConcatHMMs[i]['startprob'][:-1]), \
                np.log(phoneConcatHMMs[i]['transmat'][:-1,:-1]))
            if(maxProb is None or obs_seq_log_prob > maxProb):
                maxProbIdx = i
                maxProb = obs_seq_log_prob
        print("FOR UTTERANCE " + str(key) + " WINNER HMM: " + str(keys[maxProbIdx]))


print("USING VITERBI ALGORITHM DERIVED MAXIMUM LIKELIHOOD PREDICTIONS")
for key in keys:
    # Get observation sequences for given key
    sequences = [x['lmfcc'] for x in data if x['digit'] == key ]
    for idx, sequence in enumerate(sequences):
        maxProb = None
        maxProbIdx = None
        for i in range(len(phoneConcatHMMs)):
            obsloglik = log_mv(sequence, phoneConcatHMMs[i]['means'], phoneConcatHMMs[i]['covars'])
            # Get forward probabilities
            viterbi_loglik, viterbi_path = viterbi(obsloglik, \
                np.log(phoneConcatHMMs[i]['startprob'][:-1]), \
                np.log(phoneConcatHMMs[i]['transmat'][:-1,:-1]))
            if(maxProb is None or viterbi_loglik > maxProb):
                maxProbIdx = i
                maxProb = viterbi_loglik
        print("FOR UTTERANCE " + str(key) + " WINNER HMM: " + str(keys[maxProbIdx]))
