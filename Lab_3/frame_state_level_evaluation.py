"""
Tests the saved Keras model on a whole test set and evaluates its accuracy
on the state level frame by frame.
"""
import argparse
import numpy as np
from prondict import prondict
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model


DEFAULT_TEST_INPUTS_PATH = "Lab3_files/X_test.npy"
DEFAULT_TEST_OUTPUTS_PATH = "Lab3_files/y_test.npy"
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
    parser.add_argument("--test-inputs-path", type=str, default = DEFAULT_TEST_INPUTS_PATH,
                        help="Path to the test inputs.")
    parser.add_argument("--test-outputs-path", type=str, default = DEFAULT_TEST_OUTPUTS_PATH,
                        help="Path to the test outputs.")
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

# Load test dataset
X_test = np.load(args.test_inputs_path)
y_test = np.load(args.test_outputs_path)

# Load neural network model trained in keras
model = load_model(args.keras_model_path)

# Run inference
predictions = model.predict(X_test)

# Use argmax to get the predicted class index
ground_truths = np.argmax(y_test, axis = 1)
predicted_classes = np.argmax(predictions, axis = 1)

accuracy = np.count_nonzero(ground_truths == predicted_classes) / float(len(ground_truths))

print("Model accuracy - frame by frame - state level: " + str(accuracy))
