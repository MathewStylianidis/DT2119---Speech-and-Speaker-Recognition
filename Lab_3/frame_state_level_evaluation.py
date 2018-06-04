"""
Tests the saved Keras model on a whole test set and evaluates its accuracy
on the state level frame by frame.
"""
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
from prondict import prondict
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model


DEFAULT_TEST_INPUTS_PATH = "Lab3_files/X_test.npy"
DEFAULT_TEST_OUTPUTS_PATH = "Lab3_files/y_test.npy"
DEFAULT_SAVED_MODEL_PATH =  "Lab3_files/my_model.h5"
DEFAULT_STATE_LIST_PATH = "Lab3_files/state_list.npy"
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
    parser.add_argument("--state-list-path", type=str, default=DEFAULT_STATE_LIST_PATH,
                        help="Path to the npy filek with the state list")
    return parser.parse_args()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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

# Load state list
state_list = list(np.load(args.state_list_path))
labels = [i for i in range(len(state_list))]# List of labels to index the confusion matrix

# Compute confusion matrix
cnf_matrix = confusion_matrix(ground_truths, predicted_classes, labels = labels)
plot_confusion_matrix(cnf_matrix, classes = state_list, normalize=True)
plt.show()
