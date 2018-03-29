from proto import *

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


example = np.load('lab1_example.npz')['example'].item()
data = np.load('lab1_data.npz')['data']

#####STEP BY STEP FEATURE EXTRACTION FROM EXAMPLE DATA####
#A = enframe(example['samples'], 400, 200)
#print(compare(A, example['frames']))
#P = preemp(A, 0.97)
#print(compare(P, example['preemph']))
#H = windowing(P)
#print(compare(P, example['windowed']))
#S = powerSpectrum(H, 512)
#print(compare(S, example['spec']))
#L = logMelSpectrum(S, 20000)
#print(compare(L, example['mspec']))
#D = cepstrum(L, 13)
#print(compare(D, example['mfcc']))

#LMFCC = mfcc(example['samples'], winlen = 400, winshift = 200, preempcoeff=0.97,
                        #nfft=512, nceps=13, samplingrate=20000, liftercoeff=22)
#print(compare(LMFCC, example['lmfcc']))


##############EXTRACT FEATURES FROM UTTERANCES#####################
sampled_signals = []
signal_features = []
for idx, sample in enumerate(data):
    sampled_signal = sample['samples']
    sampled_signals.append(sampled_signal)
    extracted_features = mfcc(sampled_signal, winlen = 400, winshift = 200, preempcoeff=0.97,
                            nfft=512, nceps=13, samplingrate=20000, liftercoeff=22, result = 'mspec')
    signal_features.append(extracted_features)
    #plt.pcolormesh(extracted_features)
    #plt.savefig(str(idx) + "_" + sample['gender'] + "_" + sample['digit'] + "_"
                    #+ sample['speaker'] + "_" + sample['repetition'],
                    #bbox_inches='tight')


# PLOT FEATURE COVARIANCE MATRICES
feature_matrix = signal_features[0]
for i in range(1, len(signal_features)):
    feature_matrix = np.vstack((feature_matrix, signal_features[i]))
covariance_matrix = np.cov(feature_matrix, rowvar = False)
#plt.pcolormesh(covariance_matrix)
#plt.show()


dtw(signal_features[0], signal_features[1], np.linalg.norm)
