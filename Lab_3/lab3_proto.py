import numpy as np
from lab3_tools import *
from lab2_proto import *


def words2phones(wordList, pronDict, addSilence=True, addShortPause=False):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    if addSilence:
        phoneTrans = ['sil']

    if addShortPause:
        phoneTrans.extend([phoneme for word in wordList for phoneme in (pronDict[word] + ['sp'])])
        phoneTrans.pop()
    else:
        phoneTrans.extend([phoneme for word in wordList for phoneme in pronDict[word]])

    if addSilence:
        phoneTrans.append('sil')
    return phoneTrans

def concatAnyHMM(hmmModels, nameList):
    """ = ADD_SHORT_PAUSE
    Combines HMM models including the short silence phoneme between the words.

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
 = ADD_SHORT_PAUSE
    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
    """
    '''
    combinedHMM = {}
    modelCount = len(namelist)
    N = hmmmodels[namelist[0]]['transmat'].shape[0]
    M = (modelCount) * (N - 1)

    D = hmmmodels[namelist[0]]['means'].shape[1]
    combinedHMM['name'] = ' '.join(namelist)
    combinedHMM['transmat'] = np.zeros((M + 1, M + 1))
    combinedHMM['means'] = np.zeros((M, D)) = ADD_SHORT_PAUSE
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
    '''






def forcedAlignment(lmfcc, phoneHMMs, phoneTrans, state_list, addShortPause = False):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """

    if addShortPause:
        # Add short silence phoneme states
        utteranceHMM = concatAnyHMM(phoneHMMs, phoneTrans)
    else:
        utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)


    # Extract observations log likelihood
    obs_log_lik = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])
    # Perform Viterbi
    viterbi_likelihood, viterbi_path = viterbi(obs_log_lik, np.log(utteranceHMM['startprob'][:-1]), np.log(utteranceHMM['transmat'][:-1,:-1]))

    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
        for stateid in range(nstates[phone])]
    stateTransIndices = [state_list.index(stateTr) for stateTr in stateTrans]

    # Convert frame by frame sequence of symbols into standard format transcription
    symbol_sequence = [stateTrans[i] for i in viterbi_path]
    transcription = frames2trans(symbol_sequence, 'z43a.lab')
    #return symbol_sequence
    return transcription


def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """
