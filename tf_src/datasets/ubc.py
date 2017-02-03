import os
import cv2

import numpy as np

from tqdm import tqdm


class UBCDataset(object):
    # the extension of the images containing the patches
    IMAGE_EXT = 'bmp'
    # the file containing the matches
    matches_filename = 'm50_100000_100000_0.txt'

    def __init__(self, base_dir, test=False):
        # check that the directories exist
        assert os.path.isdir(base_dir), \
            "The given directory doesn't exist: %s" % base_dir

        # the dataset base directory
        self._base_dir = base_dir

        # testing variables
        self.test = test
        self.n    = 128

        # the loaded patches
        self._data = {
            'liberty':   None,
            'notredame': None,
            'yosemite':  None
        }

    def load_by_name(self, name, patch_size=32, num_channels=1, debug=True):
        assert name in self._data.keys(), \
            "Dataset doesn't exist: %s" % name
        assert os.path.exists(os.path.join(self._base_dir, name)), \
            "The dataset directory doesn't exist: %s" % name
        # check if the dataset is already loaded
        if self._data[name] is not None:
            print '[INFO] Dataset is cached: %s' % name
            return
        # load the patches from the images
        patches = self._load_patches(name, patch_size, num_channels)
        # load the labels
        labels = self._load_labels(self._base_dir, name)
        # load the dataset ground truth matches
        matches = self._load_matches(self._base_dir, name)
        # append data to cache
        # since might happen that there are some black patches in the end of
        # the dataset, we keep only those that have labels
        self._data[name] = dict()
        self._data[name]['patches'] = patches[0:min(len(labels), len(patches))]
        self._data[name]['labels'] = labels[0:min(len(labels), len(patches))]
        self._data[name]['matches'] = matches

        # debug info after loading
        if debug:
             print '-- Dataset loaded:    %s' % name
             print '-- Number of patches: %s' % len(self._data[name]['patches'])
             print '-- Number of labels:  %s' % len(self._data[name]['labels'])
             print '-- Number of ulabels: %s' % len(np.unique(labels))
             print '-- Number of matches: %s' % len(matches)

    def generate_stats(self, name):
        print '-- Computing dataset mean: %s ...' % name
        # compute the mean and std of all patches
        patches = self.get_patches(name)
        mean, std = self._compute_mean_and_std(patches)
        print '-- Computing dataset mean: %s ... OK' % name
        print '-- Mean: %s' % mean
        print '-- Std : %s' % std
        return mean, std
 
    def _load_matches(self, base_dir, name):
        """
        Return a list containing the ground truth matches
        """
        file_name = os.path.join(base_dir, name, self.matches_filename)
        assert os.path.isfile(file_name), 'Not a file: %s' % file
        # read file and keep only 3D point ID and 1 if is the same, otherwise 0
        matches = []
        with open(file_name, 'r') as f:
            for line in f:
                l = line.split()
                matches.append([int(l[0]), int(l[3]), int(l[1] == l[4])])
        return np.asarray(matches)

    def _load_patches(self, name, img_sz, n_channels):
        """
        Return a list containing all the patches
        """
        def load_image_filenames(base_dir, dir_name):
            """
            Return a list with the file names of the images containing the patches
            """
            files = []
            # find those files with the specified extension
            dataset_dir = os.path.join(base_dir, dir_name)
            for file_name in os.listdir(dataset_dir):
                if file_name.endswith(self.IMAGE_EXT):
                    files.append(os.path.join(dataset_dir, file_name))
            return sorted(files)  # sort files in ascend order to keep relations

        images = []
        list_files = load_image_filenames(self._base_dir, name)
        # reduce the number of files to load if we are in testing mode
        list_files = list_files[0:self.n] if self.test else list_files

        # load patches
        pbar = tqdm(list_files)
        for file_name in pbar:
            pbar.set_description('Loading dataset %s' % name)
            # pick file name
            assert os.path.isfile(file_name), 'Not a file: %s' % file
            # load the image containing the patches and convert to float point
            # and make sure that que use only one single channel
            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            # split the image into patches and
            # add patches to buffer as individual elements
            patches_row = np.split(img, 16, axis=0)
            for row in patches_row:
                patches = np.split(row, 16, axis=1)
                for patch in patches:
                    # resize the patch
                    patch_resize = cv2.resize(patch, (img_sz, img_sz),
                                              cv2.INTER_LINEAR)
                    # convert to tensor [w x h x d]
                    patch_tensor = patch_resize.reshape(img_sz,
                                                        img_sz,
                                                        n_channels)
                    images.append(patch_tensor / 255.)
        return np.array(images) if not self.test \
            else np.asarray(images[0:self.n])

    def _load_labels(self, base_dir, dir_name):
        """
        Return a list containing all the labels for each patch
        """
        info_filename = os.path.join(base_dir, dir_name, 'info.txt')
        assert os.path.isfile(info_filename), 'Not a file: %s' % file
        # read file and keep only 3D point ID
        labels = []
        with open(info_filename, 'r') as f:
            for line in f:
                labels.append(int(line.split()[0]))
        return np.asarray(labels) if not self.test \
            else np.asarray(labels[0:self.n])

    def _compute_mean_and_std(self, patches):
        """
        Return the mean and the std given a set of patches.
        """
        assert len(patches) > 0, 'Patches list is empty!'
        # compute the mean
        mean = np.mean(patches)
        # compute the standard deviation
        std = np.std(patches)
        return mean, std

    def _get_data(self, name):
        assert self._data[name] is not None, 'Dataset not loaded: %s' % name
        return self._data[name]

    def get_patches(self, name):
        return self._get_data(name)['patches']

    def get_matches(self, name):
        return self._get_data(name)['matches']

    def get_labels(self, name):
        return self._get_data(name)['labels']
