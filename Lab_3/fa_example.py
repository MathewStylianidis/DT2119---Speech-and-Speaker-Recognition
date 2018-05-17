import argparse
from lab3_tools import *
from lab3_proto import *
from lab2_proto import *
from lab1_proto import *
from prondict import prondict
import matplotlib.pyplot as plt


DEFAULT_HMM_MODELS_PATH= "../Lab_2/lab2_models_v2.npz"
DEFAULT_STATE_LIST_PATH = "Lab3_files/state_list.npy"
DEFAULT_EXAMPLE_PATH = "Lab3_files/lab3_example.npz"
ADD_SHORT_PAUSE = False

def get_arguments():
    """
    Gets the list of arguments provided from the terminal.

    Returns:
        The list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ForcedAlignment-example")
    parser.add_argument("--sample-path", type=str, default="../../TIDIGITS/disc_4.1.1/tidigits/train/man/ae/7a.wav",
                        help="Path to the file to be used for the example.")
    parser.add_argument("--hmm-models-path", type=str, default=DEFAULT_HMM_MODELS_PATH,
                        help="Path to the npz file containing the hmm models for" \
                        + " each utterance.")
    parser.add_argument("--example-path", type=str, default=DEFAULT_EXAMPLE_PATH,
                        help="Path to the npz file with the example data")
    parser.add_argument("--state-list-path", type=str, default=DEFAULT_STATE_LIST_PATH,
                        help="Path to the npy file with the state list")
    return parser.parse_args()


args = get_arguments()
filename = args.sample_path
example_path = args.example_path

# Load phoneHMMs
phoneHMMs = np.load(args.hmm_models_path)['phoneHMMs'].item()
# Load example
example = np.load(example_path)['example'].item()
# Load state indices
state_list = list(np.load(args.state_list_path))

# Load audio data
samples, sampling_rate = loadAudio(filename)
# Extract liftered MFCC features from audio sample
lmfcc = mfcc(samples)
#print(np.allclose(lmfcc, example['lmfcc']))



# Recover sequence of digits
wordTrans = list(path2info(filename)[2])
#print(wordTrans)
#print(example['wordTrans'])

# Convert words to list of phonemes
phoneTrans = words2phones(wordTrans, prondict, addSilence = True, addShortPause = ADD_SHORT_PAUSE)
#print(example['phoneTrans'])
#print(phoneTrans)

# Concatenate hmms for the phonemes corresponding to the concatenated words
if ADD_SHORT_PAUSE:
    # Add short silence phoneme states
    utteranceHMM = concatAnyHMM(phoneHMMs, phoneTrans)
else:
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
#print(np.allclose(example['utteranceHMM']['transmat'], utteranceHMM['transmat']))


# Extract state list indices
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
    for stateid in range(nstates[phone])]
stateTransIndices = [state_list.index(stateTr) for stateTr in stateTrans]
#print(example['stateTrans'])
#print(stateTrans)


# Extract observations log likelihood
obs_log_lik = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])
# Perform Viterbi
viterbi_likelihood, viterbi_path = viterbi(obs_log_lik, np.log(utteranceHMM['startprob'][:-1]), np.log(utteranceHMM['transmat'][:-1,:-1]))
#print(np.allclose(example['viterbiPath'], viterbi_path))

# Convert frame by frame sequence of symbols into standard format transcription
symbol_sequence = [stateTrans[i] for i in viterbi_path]
transcription = frames2trans(symbol_sequence, '7a.lab')
#print(example['viterbiStateTrans'] == symbol_sequence)
