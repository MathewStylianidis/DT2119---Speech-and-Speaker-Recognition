from proto import *

example = np.load('./data/lab1_example.npz')['example'].item()

#####STEP BY STEP FEATURE EXTRACTION FROM EXAMPLE DATA####
A = enframe(example['samples'], 400, 200)
print(compare(A, example['frames']))
P = preemp(A, 0.97)
print(compare(P, example['preemph']))
H = windowing(P)
print(compare(P, example['windowed']))
S = powerSpectrum(H, 512)
print(compare(S, example['spec']))
L = logMelSpectrum(S, 20000)
print(compare(L, example['mspec']))
D = cepstrum(L, 13)
print(compare(D, example['mfcc']))
## LIFTERED MFCC FEATURE EXTRACTION
LMFCC = mfcc(example['samples'], winlen = 400, winshift = 200, preempcoeff=0.97,
                        nfft=512, nceps=13, samplingrate=20000, liftercoeff=22)
print(compare(LMFCC, example['lmfcc']))
