"""
Turns the npz file acquired with feature_extraction.py to an npy file that can be used
to extract the dynamic features, etc.
"""


import argparse
import numpy as np

DEFAULT_FILE_PATH = "Lab3_files/testdata.npz"

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Phoneme recognition.")
    parser.add_argument("--file-path", type=str, default=DEFAULT_FILE_PATH,
                        help="Name of the file with the data.")
    return parser.parse_args()

args = get_arguments()
data = np.load(args.file_path)['traindata']
data = [x for x in data]

new_file_name = args.file_path.replace(".npz", ".npy")
np.save(new_file_name, data)