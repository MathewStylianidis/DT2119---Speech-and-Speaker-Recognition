# DT2119, Lab 1 Feature Extraction
import numpy as np
import matplotlib.pylab as plt
from scipy import signal, fftpack
from scipy.fftpack.realtransforms import dct
from tools import *
# Function given by the exercise ----------------------------------

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22, result = 'lmfcc'):
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
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    if(result == 'mspec'):
        return mspec
    ceps = cepstrum(mspec, nceps)
    if(result == 'mfcc'):
        return ceps
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

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
    A = np.array(samples[0:winlen].reshape((1, winlen)))
    stepsize = winlen - winshift
    for i in range(stepsize, len(samples) - winlen, stepsize):
        A = np.vstack((A, samples[i:i+winlen].reshape((1, winlen))))
    return A

def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    a = [1]
    b = [1, -p]
    return signal.lfilter(b, a, input, zi=None)

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    window = signal.hamming(input.shape[1], sym=False)
    for idx, row in enumerate(input):
        input[idx] = window * input[idx]
    return input

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    return np.power(np.abs(fftpack.fft(input, nfft)), 2)

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    T = trfbank(samplingrate, 512, lowfreq=133.33, linsc=200/3., logsc=1.0711703, nlinfilt=13, nlogfilt=27, equalareas=False)
    #Plot filter bank
    #for i in range(len(T)):
        #plt.plot(np.transpose(T[i]))
    #plt.show()
    Spec = np.dot(input, T.T)
    Spec = np.where(Spec == 0.0, np.finfo(float).eps, Spec)  # Numerical Stability
    return np.log(Spec)

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    return dct(input, type=2, axis=1, norm = 'ortho')[:,: nceps]


def minAD(accD, i, j):
    if(i == 0 and j == 0):
        return 0
    elif (i == 0):
        return accD[i, j - 1]
    elif (j == 0):
        return accD[i - 1, j]
    minimum = accD[i - 1, j]
    if(accD[i - 1, j - 1] < minimum):
        minimum = accD[i - 1, j - 1]
    if(accD[i, j - 1] < minimum):
        minimum = accD[i, j - 1]
    return minimum

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    N = x.shape[0]
    M = y.shape[0]
    LD = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            LD[i,j] = dist(x[i] - y[j])
    #plt.pcolormesh(locD)
    #plt.show()
    AD = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            AD[i,j] = LD[i, j] + minAD(AD, i, j)
    d = AD[N - 1, M - 1] / (N + M)
    path = backtrack(AD)
    #plt.pcolormesh(accD)
    #plt.show()
    return d, LD, AD, path

def backtrack(AD):
    """
    Returns best path through accumulated distance matrix AD
    """
    N = AD.shape[0]
    M = AD.shape[1]
    path = [(N - 1, M - 1)]
    i = N - 1
    j = M - 1
    while(i > 0 or j > 0):
        if( i > 0 and j > 0):
            argmin = np.argmin([AD[i - 1, j - 1], AD[i - 1, j], AD[i, j - 1]])
            if(argmin == 0):
                path.append((i - 1, j - 1))
                i = i - 1
                j = j - 1
            elif(argmin == 1):
                path.append((i - 1, j))
                i = i - 1
            elif(argmin == 2):
                path.append((i, j - 1))
                j = j - 1
        elif(i == 0 and j > 0):
            path.append((0, j - 1))
            j = j - 1
        else:
            path.append((i - 1, 0))
            i = i - 1
    return path

def compare(X, Y):
    """
    Plots a colormesh for each matrix and returns True if they are equal and
    false otherwise
    """
    plt.pcolormesh(X)
    plt.show()
    plt.pcolormesh(Y)
    plt.show()
    return np.allclose(X, Y)
