import numpy as np
import argparse
import os
from tqdm import tqdm
from prondict import prondict
from lab1_proto import *
from lab1_tools import *
from lab3_proto import *
from lab3_tools import *


DEFAULT_DATASET_PATH = "../../TIDIGITS/disc_4.1.1/tidigits/train"
DEFAULT_SAVING_PATH = "Lab3_files/"
DEFAULT_HMM_MODELS_PATH= "../Lab_2/lab2_models_v2.npz"
DEFAULT_STATE_LIST_PATH = "Lab3_files/state_list.npy"
DEFAULT_SAVING_NAME = 'traindata.npz'

def get_arguments():
    """
    Gets the list of arguments provided from the terminal.

    Returns:
        The list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ForcedAlignment-example")
    parser.add_argument("--tidigits-path", type=str, default=DEFAULT_DATASET_PATH,
                        help="Path to TIDIGITS dataset")
    parser.add_argument("--saving-path", type=str, default=DEFAULT_SAVING_PATH,
                        help="Path to directory for saving extracted features.")
    parser.add_argument("--state-list-path", type=str, default=DEFAULT_STATE_LIST_PATH,
                        help="Path to the npy filek with the state list")
    parser.add_argument("--hmm-models-path", type=str, default=DEFAULT_HMM_MODELS_PATH,
                        help="Path to the npz file containing the hmm models for" \
                             + " each utterance.")
    parser.add_argument("--save-name", type=str, default=DEFAULT_SAVING_NAME,
                        help="File name of the npz file to save the extracted features.")
    return parser.parse_args()


def feature_extraction(dataset_path, save_path, save_name, phoneHMMs, state_list, add_short_pause = False):
    """
    """

    traindata = []
    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files):
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
                samples, samplingrate = loadAudio(filename)
                mspec = mfcc(samples, result = 'mspec', samplingrate = samplingrate)
                lmfcc = mfcc(samples, result = 'lmfcc', samplingrate = samplingrate)
                wordTrans = list(path2info(filename)[2])
                phoneTrans = words2phones(wordTrans, prondict, addSilence = True, addShortPause = add_short_pause)
                targets = forcedAlignment(lmfcc, phoneHMMs, phoneTrans, state_list, addShortPause = add_short_pause)
                target_indices =  np.array([state_list.index(target) for target in targets])
                traindata.append({'filename': filename, 'lmfcc': lmfcc,
                    'mspec': mspec, 'targets': target_indices})
    np.savez(save_path + save_name, traindata=traindata)

args = get_arguments()
tidigits_path = args.tidigits_path
saving_path = args.saving_path
save_name = args.save_name

# Load phoneHMMs
phoneHMMs = np.load(args.hmm_models_path)['phoneHMMs'].item()
# Load state list
state_list = list(np.load(args.state_list_path))

feature_extraction(tidigits_path, saving_path, save_name, phoneHMMs, state_list)
