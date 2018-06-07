"""
Tests the saved Keras model on a whole test set and evaluates its accuracy
on the state level frame by frame.
"""
import argparse
import numpy as np
import itertools
import matplotlib.pyplot as plt
from Levenshtein import distance
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
DEFAULT_PHONEME_LIST_PATH = "Lab3_files/phoneme_list.npy"
DEFAULT_FEATURES = "lmfcc"
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
                        help="Path to the npy file with the state list")
    parser.add_argument("--phoneme-list-path", type=str, default=DEFAULT_PHONEME_LIST_PATH,
                        help="Path to the npy file with the phoneme list")
    return parser.parse_args()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.viridis):
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
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

args = get_arguments()

# Load state list
state_list = list(np.load(args.state_list_path))
labels = [i for i in range(len(state_list))] # List of labels to index the confusion matrix
output_dim = len(state_list)



# Load test dataset
X_test = np.load(args.test_inputs_path)
y_test = np.load(args.test_outputs_path)
y_test = np_utils.to_categorical(y_test, output_dim)

# Load neural network model trained in keras
model = load_model(args.keras_model_path)

# Run inference
predictions = model.predict(X_test)

# Use argmax to get the predicted class index
ground_truths = np.argmax(y_test, axis = 1)
predicted_classes = np.argmax(predictions, axis = 1)

############ Compute accuracy on a frame by frame basis at the state level ##############
accuracy = np.count_nonzero(ground_truths == predicted_classes) / float(len(ground_truths))
print("Model accuracy - frame by frame - state level: " + str(accuracy))

# Compute confusion matrix
cnf_matrix = confusion_matrix(ground_truths, predicted_classes, labels = labels)
plot_confusion_matrix(cnf_matrix, classes = state_list, normalize=True)
plt.show()






############## Compute accuracy on a frame by frame basis at the phoneme level ################
phoneme_dict = {phoneme: index for (index, phoneme) in enumerate(np.load(args.phoneme_list_path))}

# Convert state prediction to phonemes
phoneme_ground_truths = np.array([phoneme_dict[state_list[gt][:-2]] for gt in ground_truths])
phoneme_predicted_classes = np.array([phoneme_dict[state_list[pred][:-2]] for pred in predicted_classes])

phoneme_accuracy = np.count_nonzero(phoneme_ground_truths == phoneme_predicted_classes) / float(len(phoneme_ground_truths))
print("Model accuracy - frame by frame - phoneme level: " + str(phoneme_accuracy))

# Compute confusion matrix
cnf_matrix = confusion_matrix(phoneme_ground_truths, phoneme_predicted_classes, labels = list(phoneme_dict.values()))
plot_confusion_matrix(cnf_matrix, classes = list(phoneme_dict.keys()), normalize=True)
plt.show()





########## Compute accuracy on a frame by frame basis at the state level, merging adjacent identical states ###########
# Merge adjacent identical states
gt_transcription = [ground_truths[0]]
for i in range(1, len(ground_truths)):
    if ground_truths[i] != gt_transcription[-1]:
        gt_transcription.append(ground_truths[i])
predicted_transcription = [predicted_classes[0]]
for i in range(1, len(predicted_classes)):
    if predicted_classes[i] != predicted_transcription[-1]:
        predicted_transcription.append(predicted_classes[i])

gt_transcription = ''.join(str(x) for x in gt_transcription)
predicted_transcription = ''.join(str(x) for x in predicted_transcription)

edit_distance = distance(gt_transcription, predicted_transcription) / max(len(gt_transcription), len(predicted_transcription))
print("Edit distance - state level: " + str(edit_distance))

########## Compute accuracy on a frame by frame basis at the phoneme level, merging adjacent identical states ###########
# Merge adjacent identical states
gt_phon_transcription = [phoneme_ground_truths[0]]
for i in range(1, len(phoneme_ground_truths)):
    if phoneme_ground_truths[i] != gt_phon_transcription[-1]:
        gt_phon_transcription.append(phoneme_ground_truths[i])

predicted_phon_transcription = [phoneme_predicted_classes[0]]
for i in range(1, len(phoneme_predicted_classes)):
    if phoneme_predicted_classes[i] != predicted_phon_transcription[-1]:
        predicted_phon_transcription.append(phoneme_predicted_classes[i])

gt_phon_transcription = ''.join(str(x) for x in gt_phon_transcription)
predicted_phon_transcription = ''.join(str(x) for x in predicted_phon_transcription)

edit_distance_phon = distance(gt_phon_transcription, predicted_phon_transcription) / max(len(gt_phon_transcription), len(predicted_phon_transcription))
print("Edit distance - phoneme level: " + str(edit_distance_phon))
