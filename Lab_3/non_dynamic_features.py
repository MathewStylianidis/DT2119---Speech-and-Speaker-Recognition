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
    feature_list = []
    for idx, mfcc in enumerate(sample[args.feature_type]):
        feature = sample[args.feature_type][idx]
        feature_list.append(feature)
    sample['features'] = np.array(feature_list)

np.save("Lab3_files/d_training_data.npy", training_data)


for sample in tqdm(validation_data):
    feature_list = []
    for idx, mfcc in enumerate(sample[args.feature_type]):
        feature = sample[args.feature_type][idx]
        feature_list.append(feature)
    sample['features'] = np.array(feature_list)

np.save("Lab3_files/d_validation_data.npy", validation_data)
