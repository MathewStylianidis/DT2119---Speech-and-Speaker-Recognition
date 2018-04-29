import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import warnings
from proto2 import *
from sklearn.mixture import log_multivariate_normal_density as log_mv



if not sys.warnoptions:
    warnings.simplefilter("ignore")

def makePlots(normal, example):
   plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.005, hspace=0.005)
   norm = colors.Normalize(vmin=-1.,vmax=1.)
   plt.subplot(3,1.5,1)
   plt.pcolormesh(np.ma.masked_invalid(normal))
   plt.title('Results')
   plt.subplot(3,1.5,3)
   plt.pcolormesh(np.ma.masked_invalid(example))
   plt.title('Example')
   plt.legend()
   plt.show()


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
A = HMM['transmat'][:-1,:-1]
pi = HMM['startprob'][:-1]

# Get log-likelihood for each observation in the utterance 'o'
obsloglik = log_mv(data[22]['lmfcc'], HMM['means'], HMM['covars'])
makePlots(obsloglik.T, example['obsloglik'].T)

# Get forward probabilities
forward_probs, obs_seq_log_prob = forward(obsloglik, np.log(pi), np.log(A))
makePlots(forward_probs.T, example['logalpha'].T)
print("LOG PROBABILITY OF OBSERVATION SEQUENCE: " + str(obs_seq_log_prob))
print("DESIRED LOG PROBABILITY OF OBSERVATION SEQUENCE: " + str(example['loglik']))

# Get backward probabilities
backward_probs = backward(obsloglik, np.log(pi), np.log(A))
makePlots(backward_probs.T, example['logbeta'].T)

# Get viterbi results
viterbi_loglik, viterbi_path = viterbi(obsloglik, np.log(pi), np.log(A))
plt.pcolormesh(np.ma.masked_invalid(forward_probs.T))
plt.plot(viterbi_path)
plt.show()
print(viterbi_loglik)
print(example['vloglik'][0])
print(viterbi_path)
print(example['vloglik'][1])

# Calculate state posteriors gamma
gamma = statePosteriors(forward_probs, backward_probs)
makePlots(gamma.T, example['loggamma'].T)
# Print sum of probs in axis = 1 which should sum up to 1
print(np.sum(np.exp(gamma), axis = 1))
# Print sum of probs in axis = 0 --- Shows which states are most probable across the sequence
print(np.sum(np.exp(gamma), axis = 0))
# Sum probs across both states and timesteps -- You get the number of timesteps (DUUH, sensible since each column sums up to 1)
print(np.sum(np.exp(gamma)))


# Estimate mean and cov based on state posteriors and data
means, covars = updateMeanAndVar(obsloglik, gamma)
