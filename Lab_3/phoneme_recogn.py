import argparse
import os
import sys
import keras
import numpy as np
import pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation

# Default values for CLI arguments
DEFAULT_HIDDEN_LAYER_NO = 4
DEFAULT_X_TRAIN_PATH = "Lab3_files/X_train.npy"
DEFAULT_Y_TRAIN_PATH =  "Lab3_files/y_train.npy"
DEFAULT_X_VAL_PATH =  "Lab3_files/X_val.npy"
DEFAULT_Y_VAL_PATH =  "Lab3_files/y_val.npy"
DEFAULT_SAVE_OPTION = False


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Phoneme recognition.")
    parser.add_argument("--hidden-layer-no", type=int, default = DEFAULT_HIDDEN_LAYER_NO,
                        help="Number of hidden layers.")
    parser.add_argument("--X-train-path", type=str, default = DEFAULT_X_TRAIN_PATH,
                        help="Training input data path.")
    parser.add_argument("--y-train-path", type=str, default = DEFAULT_Y_TRAIN_PATH,
                        help="Training labels data path..")
    parser.add_argument("--X-val-path", type=str, default = DEFAULT_X_VAL_PATH,
                        help="Validation input data path..")
    parser.add_argument("--y-val-path", type=str, default = DEFAULT_Y_VAL_PATH,
                        help="Validation labels data path..")
    parser.add_argument("--save", type=bool, default=DEFAULT_SAVE_OPTION,
                        help="If true then the final model is saved.")
    return parser.parse_args()

args = get_arguments()


X_train = np.load(args.X_train_path).astype('float32')
y_train = np.load(args.y_train_path)

X_val = np.load(args.X_val_path).astype('float32')
y_val = np.load(args.y_val_path)

# Convert target arrays to Keras categorical format
stateList = np.load("Lab3_files/state_list.npy")
output_dim = len(stateList)
y_train = np_utils.to_categorical(y_train, output_dim)
y_val = np_utils.to_categorical(y_val, output_dim)




input_dim = X_train.shape[1]

# Define the deep neural network
model = Sequential()
model.add(Dense(256, input_dim=input_dim, activation='relu'))
for i in range(args.hidden_layer_no - 1):
    model.add(Dense(256, activation='relu'))
model.add(Dense(output_dim, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
history = model.fit(X_train, y_train, epochs=25, batch_size=256, \
            validation_data = (X_val, y_val), verbose = 1)

with open("Lab3_files/trainHistoryDict_" + str(args.hidden_layer_no), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if args.save is True:
    model.save('Lab3_files/my_model.h5')
    print("Learned weights are saved in Lab3_files/my_model.h5")
