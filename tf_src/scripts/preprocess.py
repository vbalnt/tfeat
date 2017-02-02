import os
import pickle
import numpy as np

from tqdm import tqdm


def normalize_data(data, mean, std):
    normalized_data = []
    pbar = tqdm(data)
    for sample in pbar:
        pbar.set_description('Normalizing data')
        sample_norm = (sample - mean) / std
        normalized_data.append(sample_norm)
    return np.asarray(normalized_data)


def load_stats(dataset_name, data_dir='data'):
    file_name = os.path.join(data_dir, 'stats_{}.pkl'.format(dataset_name))
    if not os.path.isfile(file_name):
        raise Exception('File {} does not exist'.format(file_name))

    file_obj = open(file_name, 'r')
    mean, std = pickle.load(file_obj)
    return mean, std



