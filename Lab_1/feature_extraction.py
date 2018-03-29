from proto import *

data = np.load('./data/lab1_data.npz')['data']

##############EXTRACT FEATURES FROM ALL UTTERANCES#####################
sampled_signals = []
signal_features = []
for idx, sample in enumerate(data):
    sampled_signal = sample['samples']
    sampled_signals.append(sampled_signal)
    extracted_features = mfcc(sampled_signal, winlen = 400, winshift = 200, preempcoeff=0.97,
                            nfft=512, nceps=13, samplingrate=20000, liftercoeff=22, result = 'lmfcc')
    signal_features.append(extracted_features)
    #plt.pcolormesh(extracted_features)
    #plt.savefig(str(idx) + "_" + sample['gender'] + "_" + sample['digit'] + "_"
                    #+ sample['speaker'] + "_" + sample['repetition'],
                    #bbox_inches='tight')

np.savez("./data/lmfcc_utterance_features.npz",signal_features)
