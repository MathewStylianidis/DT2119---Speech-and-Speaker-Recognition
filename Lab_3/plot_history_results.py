import argparse
import os
import sys
import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Phoneme recognition.")
    parser.add_argument("--history-path", type=str,
                        help="Pickle file with history path.")
    return parser.parse_args()

args = get_arguments()



with open(args.history_path, 'rb') as file_pi:
        history = pickle.load(file_pi)
        val_loss = history['val_loss']
        loss = history['loss']
        val_acc = history['val_acc']
        acc = history['acc']

        # Plot loss
        plt.title('Training/Validation loss')
        plt.plot(val_loss, label="Validation loss")
        plt.plot(loss, label="Training loss")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()

        # Plot accuracy
        plt.title('Training/Validation accuracy')
        plt.plot(val_acc, label="Validation accuracy")
        plt.plot(acc, label="Training accuracy")
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()
