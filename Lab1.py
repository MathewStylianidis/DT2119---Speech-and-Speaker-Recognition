
# coding: utf-8

# In[1]:


import numpy as np


# In[12]:


# Names of files including the data
data_file = "lab1_data.npz"
debug_data_file = "lab1_example.npz" # File containing data used for debugging the extraction of MFCC


# ##### Load datasets

# In[67]:


# Load data 
dataset = np.load(data_file)['data']
utterances = [utterance['samples'] for utterance in dataset]
utterance_sampling_rate = [utterance['samplingrate'] for utterance in dataset]
digits = [utterance['digit'] for utterance in dataset]
repetitions = [utterance['repetition'] for utterance in dataset]
genders = [utterance['gender'] for utterance in dataset]
speakers = [utterance['speaker'] for utterance in dataset]


# In[68]:


# Load debugging data
debug_dataset = np.load(debug_data_file)['example'].item()
samples = debug_dataset['samples']
sampling_rate = debug_dataset['samplingrate']
frames = debug_dataset['frames']
preemph = debug_dataset['preemph']
windowed = debug_dataset['windowed']
spec = debug_dataset['spec']
mspec = debug_dataset['mspec']
mfcc = debug_dataset['mfcc']
lmfcc = debug_dataset['lmfcc']


# ### Feature extraction functions

# In[47]:



def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    return None
    #preemph = preemp(frames, preempcoeff)
    #windowed = windowing(preemph)
    #spec = powerSpectrum(windowed, nfft)
    #mspec = logMelSpectrum(spec, samplingrate)
    #ceps = cepstrum(mspec, nceps)
    #return lifter(ceps, liftercoeff)


# In[60]:


def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    
    return samples
    


# ### Main program

# In[73]:


#Calculate winlen and winshift based on samplingrate
#enframed_samples = [enframe(utterance, winlen = 20 * 1e-3, winshift = 10 * 1e-3) for utterance in utterances]


# In[78]:


print(len(utterances[0]))

