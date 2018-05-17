import numpy as np
from tqdm import tqdm

training_data = np.load("Lab3_files/training_data.npy")
validation_data = np.load("Lab3_files/validation_data.npy")

tr_dynamic_features = []
val_dynamic_features = []

for sample in tqdm(training_data):
    dynamic_feature_list = []
    max_idx = len(sample['lmfcc']) - 1
    for idx, mfcc in enumerate(sample['lmfcc']):
        dynamic_feature = np.zeros((7, mfcc.shape[0]))
<<<<<<< HEAD
        dynamic_feature[0] = sample['lmfcc'][np.abs(idx - 3)]
        dynamic_feature[1] = sample['lmfcc'][np.abs(idx - 2)]
        dynamic_feature[2] = sample['lmfcc'][np.abs(idx - 1)]
        dynamic_feature[3] = sample['lmfcc'][idx]
        dynamic_feature[4] = sample['lmfcc'][max_idx - np.abs(max_idx - (idx + 1))]
        dynamic_feature[5] = sample['lmfcc'][max_idx - np.abs(max_idx - (idx + 2))]
        dynamic_feature[6] = sample['lmfcc'][max_idx - np.abs(max_idx - (idx + 3))]
        dynamic_feature_list.append(dynamic_feature)
    sample['dynamic_features'] = np.array(dynamic_feature_list)
=======
        dynamic_feature = sample['lmfcc'][np.abs(idx - 3)]
        dynamic_feature = sample['lmfcc'][np.abs(idx - 2)]
        dynamic_feature = sample['lmfcc'][np.abs(idx - 1)]
        dynamic_feature = sample['lmfcc'][idx]
        dynamic_feature = sample['lmfcc'][max_idx - np.abs(max_idx - (idx + 1))]
        dynamic_feature = sample['lmfcc'][max_idx - np.abs(max_idx - (idx + 2))]
        dynamic_feature = sample['lmfcc'][max_idx - np.abs(max_idx - (idx + 3))]
        dynamic_feature_list.append(dynamic_feature)
    sample['dynamic_features'] = list(dynamic_feature_list)
>>>>>>> e19d015d1020544d6b79fe276ad57c396011d08c

for sample in tqdm(validation_data):
    dynamic_feature_list = []
    max_idx = len(sample['lmfcc']) - 1
    for idx, mfcc in enumerate(sample['lmfcc']):
        dynamic_feature = np.zeros((7, mfcc.shape[0]))
<<<<<<< HEAD
        dynamic_feature[0] = sample['lmfcc'][np.abs(idx - 3)]
        dynamic_feature[1] = sample['lmfcc'][np.abs(idx - 2)]
        dynamic_feature[2] = sample['lmfcc'][np.abs(idx - 1)]
        dynamic_feature[3] = sample['lmfcc'][idx]
        dynamic_feature[4] = sample['lmfcc'][max_idx - np.abs(max_idx - (idx + 1))]
        dynamic_feature[5] = sample['lmfcc'][max_idx - np.abs(max_idx - (idx + 2))]
        dynamic_feature[6] = sample['lmfcc'][max_idx - np.abs(max_idx - (idx + 3))]
        dynamic_feature_list.append(dynamic_feature)
    sample['dynamic_features'] = np.array(dynamic_feature_list)

np.save("Lab3_files/d_training_data.npy", training_data)
np.save("Lab3_files/d_validation_data.npy", validation_data)
=======
        dynamic_feature = sample['lmfcc'][np.abs(idx - 3)]
        dynamic_feature = sample['lmfcc'][np.abs(idx - 2)]
        dynamic_feature = sample['lmfcc'][np.abs(idx - 1)]
        dynamic_feature = sample['lmfcc'][idx]
        dynamic_feature = sample['lmfcc'][max_idx - np.abs(max_idx - (idx + 1))]
        dynamic_feature = sample['lmfcc'][max_idx - np.abs(max_idx - (idx + 2))]
        dynamic_feature = sample['lmfcc'][max_idx - np.abs(max_idx - (idx + 3))]
        dynamic_feature_list.append(dynamic_feature)
    sample['dynamic_features'] = list(dynamic_feature_list)

np.save("Lab3_files/training_data.npy", training_data)
np.save("Lab3_files/validation_data.npy", validation_data)
>>>>>>> e19d015d1020544d6b79fe276ad57c396011d08c
