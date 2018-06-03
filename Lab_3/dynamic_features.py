import numpy as np
import argparse
from tqdm import tqdm

# Default values for CLI arguments
DEFAULT_FEATURES = 'lmfcc'
DEFAULT_FILE_NAME = "training_data.npy"
DEFAULT_SAVE_NAME = "d_training_data.npy"
DEFAULT_FILE_DIR = "Lab3_files/"

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Phoneme recognition.")
    parser.add_argument("--feature-type", type=str, default = DEFAULT_FEATURES,
                        help="Type of features to use.")
    parser.add_argument("--file-name", type=str, default=DEFAULT_FILE_NAME,
                        help="Name of the file with the data.")
    parser.add_argument("--file-dir", type=str, default=DEFAULT_FILE_DIR,
                        help="Name of the directory with the data files.")
    parser.add_argument("--save-name", type=str, default=DEFAULT_SAVE_NAME,
                        help="Name of the file with the dynamic features.")
    return parser.parse_args()

args = get_arguments()

training_data = np.load(args.file_dir + args.file_name)

tr_dynamic_features = []
val_dynamic_features = []

for sample in tqdm(training_data):
    dynamic_feature_list = []
    max_idx = len(sample[args.feature_type]) - 1
    for idx, mfcc in enumerate(sample[args.feature_type]):
        dynamic_feature = np.zeros((7, mfcc.shape[0]))

        dynamic_feature[0] = sample[args.feature_type][np.abs(idx - 3)]
        dynamic_feature[1] = sample[args.feature_type][np.abs(idx - 2)]
        dynamic_feature[2] = sample[args.feature_type][np.abs(idx - 1)]
        dynamic_feature[3] = sample[args.feature_type][idx]
        dynamic_feature[4] = sample[args.feature_type][max_idx - np.abs(max_idx - (idx + 1))]
        dynamic_feature[5] = sample[args.feature_type][max_idx - np.abs(max_idx - (idx + 2))]
        dynamic_feature[6] = sample[args.feature_type][max_idx - np.abs(max_idx - (idx + 3))]
        dynamic_feature_list.append(dynamic_feature)
    sample['features'] = np.array(dynamic_feature_list)

np.save(args.file_dir + args.save_name, training_data)