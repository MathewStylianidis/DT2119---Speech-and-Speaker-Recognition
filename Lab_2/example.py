import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from proto2 import *
from sklearn.mixture import log_multivariate_normal_density as log_mv

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
obsloglik = log_mv(data[0]['lmfcc'], HMM['means'], HMM['covars'])
forward_probs = forward(example['obsloglik'], np.log(pi), np.log(A))
backward_probs = backward(example['obsloglik'], np.log(pi), np.log(A))

makePlots(obsloglik.T, example['obsloglik'].T)
makePlots(forward_probs.T, example['logalpha'].T)
makePlots(backward_probs.T, example['logbeta'].T)


'''
viterbi_loglik, viterbi_path = viterbi(example['obsloglik'], np.log(pi), np.log(A))
print(example['logalpha'][5])
print(forward_probs[5])
#print(viterbi_loglik)
#print(example['vloglik'][0])
#print(len(viterbi_path))
#print(len(example['vloglik'][1]))
'''