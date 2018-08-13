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

