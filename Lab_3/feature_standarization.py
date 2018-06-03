import numpy as np
import argparse
from sklearn import preprocessing



# Default values for CLI arguments
DEFAULT_TRAIN_PATH = "Lab3_files/d_training_data.npy"
DEFAULT_VAL_PATH =  "Lab3_files/d_validation_data.npy"
DEFAULT_STATE_LIST_PATH = "Lab3_files/state_list.npy"


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Phoneme recognition.")
    parser.add_argument("--train-path", type=int, default = DEFAULT_TRAIN_PATH,
                        help="Number of hidden layers.")
    parser.add_argument("--val-path", type=str, default = DEFAULT_VAL_PATH,
                        help="Training input data path.")
    parser.add_argument("--state-list-path", type=str, default = DEFAULT_STATE_LIST_PATH,
                        help="Training labels data path..")
    return parser.parse_args()

args = get_arguments()



training_data = np.load(args.train_path)
validation_data = np.load(args.val_path)
state_list = list(np.load(args.state_list_path))


N = 0
D = np.prod(np.array(training_data[0]['features']).shape[1:3])
for sample in training_data:
    N += sample['features'].shape[0]



X_train = np.zeros((N, D))
y_train = np.zeros((N, 1))
prev_idx = 0
for sample in training_data:
    dynamic_features = np.array(sample['features'])
    n = dynamic_features.shape[0]
    X_train[prev_idx:prev_idx + n] = dynamic_features.reshape((n, D))
    y_train[prev_idx:prev_idx + n, 0] = sample['targets']
    prev_idx += n



N = 0
for sample in validation_data:
    N += np.array(sample['features']).shape[0]

X_val = np.zeros((N, D))
y_val = np.zeros((N, 1))
prev_idx = 0
for sample in validation_data:
    dynamic_features = np.array(sample['features'])
    n = dynamic_features.shape[0]
    X_val[prev_idx:prev_idx + n] = dynamic_features.reshape((n, D))
    y_val[prev_idx:prev_idx + n, 0] = sample['targets']
    prev_idx += n


scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
#X_test = scaler.transform(X_test)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
#X_test = X_test.astype('float32')

print(X_train.shape)
print(X_val.shape)


np.save("Lab3_files/X_train.npy", X_train)
np.save("Lab3_files/X_val.npy", X_val)

np.save("Lab3_files/y_train.npy", y_train)
np.save("Lab3_files/y_val.npy", y_val)
