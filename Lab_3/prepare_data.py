import numpy as np
import argparse

#import os
#import sys
from Spinner import *

DEFAULT_HMM_MODELS_PATH= "../Lab_2/lab2_models.npz"
DEFAULT_STATE_LIST_PATH = "Lab3_files/state_list.npy"

def get_arguments():
    """
    Gets the list of arguments provided from the terminal.

    Returns:
        The list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Speech-rec-preprocess")
    parser.add_argument("--hmm-models-path", type=str, default=DEFAULT_HMM_MODELS_PATH,
                        help="Path to the npz file containing the hmm models for" \
                        + " each utterance.")
    parser.add_argument("--state-list-path", type=str, default=DEFAULT_STATE_LIST_PATH,
                        help="Path of the directory where the list of state will" + \
                        " be saved. This should contain the file name in the end.")
    return parser.parse_args()



#spinner = Spinner()

args = get_arguments()

print("Loading HMM models...")
hmmModels = np.load(args.hmm_models_path)['phoneHMMs'].item()
print("Creating state list...")
phones = sorted(hmmModels.keys())
nstates = {phone: hmmModels[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
print("State list created...\n")



print("Saving state list...")
np.save(args.state_list_path, stateList)
print("State list completed.")







#
