import sys
import pickle

from datasets import UBCDataset

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print '[ERROR] Not enough parameters'
        print '[INFO]  Usage: python generate_stats.py [<data_dir<] [<dataset_name>]'
        sys.exit()

    DATASET_DIR  = str(sys.argv[1])
    DATASET_NAME = str(sys.argv[2])

    # instantiate UBC dataset
    dataset = UBCDataset(DATASET_DIR)

    # load dataset
    dataset.load_by_name(DATASET_NAME)

    # generate 'N' triplets
    mean, std = dataset.generate_stats(DATASET_NAME)
    
    print 'Saving data ...'
    data_fname = 'stats_%s.pkl' % DATASET_NAME
    data_file  = open(data_fname, 'wb')
    pickle.dump([mean, std], data_file)
    print 'Saving data ... OK'
