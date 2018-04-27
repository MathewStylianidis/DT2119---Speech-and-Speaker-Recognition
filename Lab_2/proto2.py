import numpy as np
from scipy.linalg import block_diag
from tools2 import *

def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of states in each HMM model (could be different for each)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    combinedHMM = {}
    modelCount = len(namelist)
    N = hmmmodels[namelist[0]]['transmat'].shape[0]
    M = (modelCount) * (N - 1)
    D = hmmmodels[namelist[0]]['means'].shape[1]
    combinedHMM['name'] = ' '.join(namelist)
    combinedHMM['transmat'] = np.zeros((M + 1, M + 1))
    combinedHMM['means'] = np.zeros((M, D))
    combinedHMM['covars'] = np.zeros((M, D))
    combinedHMM['startprob'] = np.zeros((M + 1, 1))
    step = N - 1
    for idx, name in enumerate(namelist):
        if(idx == 0):
            combinedHMM['startprob'][idx*step:idx*step + step + 1] = \
                        hmmmodels[name]['startprob'].reshape(-1, 1)
        else:
            combinedHMM['startprob'][idx*step:idx*step + step + 1] = np.zeros((N, 1))
        combinedHMM['transmat'][idx*step:idx*step + step + 1, idx*step:idx*step + step + 1] += hmmmodels[name]['transmat']

        combinedHMM['means'][idx*step:idx*step + step] += hmmmodels[name]['means']
        combinedHMM['covars'][idx*step:idx*step + step] += hmmmodels[name]['covars']
    combinedHMM['transmat'][-1, -1] = 1.0
    return combinedHMM


def concatHMMs2(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of states in each HMM model (could be different for each)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    combinedhmm = {}
    combinedhmm['name'] = ''.join(namelist)
    combinedhmm['startprob'] = np.zeros(len(namelist) * 3)
    combinedhmm['startprob'][0:4] = hmmmodels[namelist[0]]['startprob']
    combinedhmm['transmat'] = np.zeros([len(namelist) * 3 + 1, len(namelist) * 3 + 1])
    for i, name in enumerate(namelist):
        combinedhmm['transmat'][i * 3:i * 3 + 4, i * 3:i * 3 + 4] = hmmmodels[name]['transmat']
        if i == 0:
            combinedhmm['means'] = hmmmodels[name]['means']
            combinedhmm['covars'] = hmmmodels[name]['covars']
        else:
            combinedhmm['means'] = np.concatenate([combinedhmm['means'], hmmmodels[name]['means']])
            combinedhmm['covars'] = np.concatenate([combinedhmm['covars'], hmmmodels[name]['covars']])
    combinedhmm['startprob'] = np.expand_dims(combinedhmm['startprob'], axis=1)
    combinedhmm['transmat'] = combinedhmm['transmat'][:-1, :-1]
    return combinedhmm



def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """


def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    alpha = np.zeros(log_emlik.shape)
    alpha[0][:] = log_startprob.T + log_emlik[0]

    for n in range(1,len(alpha)):
        for i in range(alpha.shape[1]):
            alpha[n, i] = logsumexp(alpha[n - 1] + log_transmat[:,i]) + log_emlik[n,i]
    return alpha




def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

    log_b = np.zeros(log_emlik.shape)# + 1.0 / log_emlik.shape[1]
    for n in reversed(range(log_emlik.shape[0] - 1)):
        for i in range(log_emlik.shape[1]):
            log_b[n, i] = logsumexp(log_transmat[i,:] + log_emlik[n + 1, :] + log_b[n + 1,:])
    return log_b


def viterbiBacktrack(B, lastIdx):
    """Does backtracking retrieving the viterbi path given the most probable
        previous indices in each timestep.

    Args:
        B: NxM array where N are the timesteps and M are the states and each
            element contains the most probable state in the previous timestep.
        lastIdx: index of the most probable state in timestep N
    Returns:
        A vector of N-1 elements with the viterbi path
    """
    viterbi_path = [lastIdx]
    for i in reversed(range(1, B.shape[0])):
        viterbi_path.append(B[i, viterbi_path[-1]])
    viterbi_path.reverse()
    return np.array(viterbi_path)

def viterbi(log_emlik, log_startprob, log_transmat):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    B = np.zeros(log_emlik.shape, dtype = int)
    V = np.zeros(log_emlik.shape)
    V[0] = log_startprob.flatten() + log_emlik[0]

    for n in range(1, log_emlik.shape[0]):
        for j in range(log_emlik.shape[1]):
            V[n][j] = np.max(V[n - 1,:] + log_transmat[:,j]) + log_emlik[n, j]
            B[n][j] = np.argmax(V[n - 1,:] + log_transmat[:,j])

    # Backtrack to take viteri path
    viterbi_path = viterbiBacktrack(B, np.argmax(V[ log_emlik.shape[0] - 1]))

    return np.max(V[ log_emlik.shape[0] - 1]), viterbi_path


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
