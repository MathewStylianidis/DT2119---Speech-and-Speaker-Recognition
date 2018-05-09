import argparse
from lab3_tools import *
from lab1_proto import *

def get_arguments():
    """
    Gets the list of arguments provided from the terminal.

    Returns:
        The list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Speech-rec-preprocess")
    parser.add_argument("--sample-path", type=str, default="../../TIDIGITS/disc_4.1.1/tidigits/train/man/nw/z43a.wav",
                        help="Path to the file to be used for the example.")
    return parser.parse_args()


args = get_arguments()
filename = args.sample_path

# Load audio data
samples, sampling_rate = loadAudio(filename)
# Extract liftered MFCC features from audio sample
lmfcc = mfcc(samples)
