"""
Tests the saved Keras model on a single utterance from the test set
"""
import argparse
import numpy as np
from sklearn.externals import joblib
from prondict import prondict
from lab1_proto import *
from lab1_tools import *
from lab3_proto import *
from lab3_tools import *
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model


#FEED THE INDEX OF THE UTTERANCE IN THE TEST SET, CHECK THE NUMBER OF FRAMES IT HAS
# LOAD THAT MANY FRAMES FROM THE TEST SET, PREDICT FOR EACH FRAME WITH LOADED MODEL
# PLOT PREDICTIONS + ONE HOT ENCODED VECTOR OF THE LABEL

DEFAULT_TEST_SAMPLE_PATH = "../../TIDIGITS/disc_4.2.1/tidigits/train/man/ah/1a.wav"
DEFAULT_PREPROCESSOR_PATH = "Lab3_files/scaler.save"
DEFAULT_SAVED_MODEL_PATH =  "Lab3_files/my_model.h5"
DEFAULT_STATE_LIST_PATH = "Lab3_files/state_list.npy"
DEFAULT_HMM_MODELS_PATH= "../Lab_2/lab2_models_v2.npz"
DEFAULT_INDEX = 0
DEFAULT_FEATURES = 'lmfcc'
DEFAULT_DYNAMIC = True

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Phoneme recognition.")
    parser.add_argument("--feature-type", type=str, default = DEFAULT_FEATURES,
                        help="Type of features to use.")
    parser.add_argument("--sample-path", type=str, default = DEFAULT_TEST_SAMPLE_PATH,
                        help="Path to the test sample.")
    parser.add_argument("--preproc-path", type=str, default = DEFAULT_PREPROCESSOR_PATH,
                        help="Path to preprocessor saved object.")
    parser.add_argument("--keras-model-path", type=str, default=DEFAULT_SAVED_MODEL_PATH,
                        help="Saved model path.")
    parser.add_argument("--utterance-index", type=int, default=DEFAULT_INDEX,
                        help="Index of utterance to be used for the predictions.")
    parser.add_argument("--state-list-path", type=str, default=DEFAULT_STATE_LIST_PATH,
                        help="Path to the npy filek with the state list")
    parser.add_argument("--hmm-models-path", type=str, default=DEFAULT_HMM_MODELS_PATH,
                        help="Path to the npz file containing the hmm models for" \
                             + " each utterance.")
    parser.add_argument("--dynamic", type=bool, default = DEFAULT_DYNAMIC,
                        help="If true dynamic features are used.")
    return parser.parse_args()

args = get_arguments()

# Load phoneHMMs
phoneHMMs = np.load(args.hmm_models_path)['phoneHMMs'].item()
# Load state list
state_list = list(np.load(args.state_list_path))
output_dim = len(state_list)

# Extract mspec or/and lmfcc features
samples, samplingrate = loadAudio(args.sample_path)
mspec = mfcc(samples, result='mspec', samplingrate=samplingrate)
lmfcc = mfcc(samples, result='lmfcc', samplingrate=samplingrate)
wordTrans = list(path2info(args.sample_path)[2])
phoneTrans = words2phones(wordTrans, prondict, addSilence=True, addShortPause=False)
targets = forcedAlignment(lmfcc, phoneHMMs, phoneTrans, state_list, addShortPause=False)
target_indices = np.array([state_list.index(target) for target in targets])
sample = {'filename': args.sample_path, 'lmfcc': lmfcc,
                  'mspec': mspec, 'targets': target_indices}


# Create dynamic or non dynamic features in np array format
if args.dynamic is True:
    # Extract dynamic features
    dynamic_feature_list = []
    max_idx = len(sample[args.feature_type]) - 1
    for idx, feature in enumerate(sample[args.feature_type]):
        dynamic_feature = np.zeros((7, feature.shape[0]))
        dynamic_feature[0] = sample[args.feature_type][np.abs(idx - 3)]
        dynamic_feature[1] = sample[args.feature_type][np.abs(idx - 2)]
        dynamic_feature[2] = sample[args.feature_type][np.abs(idx - 1)]
        dynamic_feature[3] = sample[args.feature_type][idx]
        dynamic_feature[4] = sample[args.feature_type][max_idx - np.abs(max_idx - (idx + 1))]
        dynamic_feature[5] = sample[args.feature_type][max_idx - np.abs(max_idx - (idx + 2))]
        dynamic_feature[6] = sample[args.feature_type][max_idx - np.abs(max_idx - (idx + 3))]
        dynamic_feature_list.append(dynamic_feature.flatten())
    sample['features'] = np.array(dynamic_feature_list)
else:
    feature_list = [np.array(feature) for feature in sample[args.feature_type]]
    sample['features'] = np.array(feature_list)


# Standarize sample according to saved pre-processing object StandardScaler
scaler = joblib.load(args.preproc_path)
X = scaler.transform(sample['features'])
y = np_utils.to_categorical(sample['targets'], output_dim)


# Load neural network model trained in keras
model = load_model(args.keras_model_path)

# Run inference
predictions = model.predict(X)

# Plot results
plt.title("Posterior")
plt.pcolormesh(predictions.T)
plt.colorbar()
plt.show()

plt.title("Ground truth")
plt.pcolormesh(y.T)
plt.colorbar()
plt.show()
