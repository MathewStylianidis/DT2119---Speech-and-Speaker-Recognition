import argparse
from lab3_tools import *
from lab3_proto import *
from lab2_proto import *
from lab1_proto import *
from prondict import prondict

DEFAULT_HMM_MODELS_PATH= "../Lab_2/lab2_models.npz"
DEFAULT_STATE_LIST_PATH = "Lab3_files/state_list.npy"

def get_arguments():
    """
    Gets the list of arguments provided from the terminal.

    Returns:
        The list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ForcedAlignment-example")
    parser.add_argument("--sample-path", type=str, default="../../TIDIGITS/disc_4.1.1/tidigits/train/man/nw/z43a.wav",
                        help="Path to the file to be used for the example.")
    parser.add_argument("--hmm-models-path", type=str, default=DEFAULT_HMM_MODELS_PATH,
                        help="Path to the npz file containing the hmm models for" \
                        + " each utterance.")
    return parser.parse_args()


args = get_arguments()
filename = args.sample_path

# Load phoneHMMs
phoneHMMs = np.load(args.hmm_models_path)['phoneHMMs'].item()

# Load audio data
samples, sampling_rate = loadAudio(filename)
# Extract liftered MFCC features from audio sample
lmfcc = mfcc(samples)

# Recover sequence of digits
wordTrans = list(path2info(filename)[2])

# TODO: Implement word2phones
phoneTrans = words2phones(wordTrans, prondict)

# Concatenate hmms for the phonemes corresponding to the concatenated words
utterance = concatHMMs(phoneHMMs, phoneTrans)
