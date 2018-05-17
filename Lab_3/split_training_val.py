import numpy as np

DEFAULT_SAVING_PATH = "Lab3_files/"
MAN_SAMPLES = 4235
WOMAN_SAMPLES = 4388
VAL_PER = 0.1
SAMPLES_PER_PERSON = 77

data = np.load(DEFAULT_SAVING_PATH + 'traindata.npz')['traindata']
sample_no = len(data)
approx_val_size = int(0.1 * sample_no)
val_class_people_count = int(approx_val_size / (SAMPLES_PER_PERSON))

if val_class_people_count % 2 != 0:
    val_class_people_count += 1

val_size = val_class_people_count * SAMPLES_PER_PERSON
training_size = sample_no - val_size

samples_per_gender = int(val_size / 2)

val_data = [data[i] for i in range(0, samples_per_gender)]
val_data.extend([data[i] for i in range(MAN_SAMPLES, MAN_SAMPLES + samples_per_gender)])
training_data = [sample for sample in data if sample['filename'] not in [x['filename'] for x in val_data]]

np.save(DEFAULT_SAVING_PATH + 'training_data.npy', training_data)
np.save(DEFAULT_SAVING_PATH + 'validation_data.npy', val_data)
