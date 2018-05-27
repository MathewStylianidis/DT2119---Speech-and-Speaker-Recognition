import numpy as np
from sklearn import preprocessing

training_data = np.load("Lab3_files/d_training_data.npy")
validation_data = np.load("Lab3_files/d_validation_data.npy")

N = 0
D = np.prod(np.array(training_data[0]['dynamic_features']).shape[1:3])
for sample in training_data:
    N += sample['dynamic_features'].shape[0]



X_train = np.zeros((N, D))
prev_idx = 0
for sample in training_data:
    dynamic_features = np.array(sample['dynamic_features'])
    n = dynamic_features.shape[0]
    X_train[prev_idx:prev_idx + n] = dynamic_features.reshape((n, D))
    prev_idx += n



N = 0
for sample in validation_data:
    N += np.array(sample['dynamic_features']).shape[0]

X_val = np.zeros((N, D))
prev_idx = 0
for sample in validation_data:
    dynamic_features = np.array(sample['dynamic_features'])
    n = dynamic_features.shape[0]
    X_val[prev_idx:prev_idx + n] = dynamic_features.reshape((n, D))
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


np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
