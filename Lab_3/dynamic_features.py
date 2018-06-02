import numpy as np
import argparse
from tqdm import tqdm

# Default values for CLI arguments
DEFAULT_FEATURES = 'lmfcc'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Phoneme recognition.")
    parser.add_argument("--feature-type", type=str, default = DEFAULT_FEATURES,
                        help="Type of features to use.")
    return parser.parse_args()

args = get_arguments()

training_data = np.load("Lab3_files/training_data.npy")
validation_data = np.load("Lab3_files/validation_data.npy")

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
    sample['dynamic_features'] = np.array(dynamic_feature_list)



for sample in tqdm(validation_data):
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
    sample['dynamic_features'] = np.array(dynamic_feature_list)

np.save("Lab3_files/d_training_data.npy", training_data)
np.save("Lab3_files/d_validation_data.npy", validation_data)
