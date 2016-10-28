import sys
import os

import random
import collections
from tqdm import tqdm

import cv2
import numpy as np

class UBCDataset(object):
    # the extension of the images containing the patches
    IMAGE_EXT  = 'bmp'
    # the size of the patches once extracted
    PATCH_SIZE = 64
    # the number of patches per row/column in the
    # image containing all the patches
    PATCHES_PER_ROW   = 16
    PATCHES_PER_IMAGE = PATCHES_PER_ROW**2

    def __init__(self, base_dir, test=False):
        # check that the directories exist
        assert os.path.isdir(base_dir) == True, \
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

    def get_batch(self, data, data_ids, step, batch_size):
        # compute the offset to get the correct batch
        offset = step * batch_size % len(data_ids)
        # get a triplet batch from the training data
        ids = data_ids[offset:offset + batch_size]

        a, p, n = [[] for _ in range(3)]
        for id in ids:
                a.append(data[id[0]])
                p.append(data[id[1]])
                n.append(data[id[2]])
        return a, p, n

    def load_by_name(self, name, patch_size=32, num_channels=1, debug=True):
        assert name in self._data.keys(), \
            "Dataset doesn't exist: %s" % name
        assert os.path.exists(os.path.join(self._base_dir, name)) == True, \
            "The dataset directory doesn't exist: %s" % name
        # check if the dataset is already loaded
        if self._data[name] is not None:
            print '[INFO] Dataset is cached: %s' % name
            return
        # load the images containing the patches
        img_files = self._load_image_fnames(self._base_dir, name)
        # load the patches from the images
        patches = self._load_patches(img_files, name, patch_size, num_channels)
        # load the labels
        labels = self._load_labels(self._base_dir, name)
        # load the dataset ground truth matches
        matches = self._load_matches(self._base_dir, name)
	# append data to cache
	# since may happen that there are some black patches in the end of
	# the dataset, we keep only those that we have labels
	self._data[name] = dict()
	self._data[name]['patches'] = patches[0:min(len(labels), len(patches))]
	self._data[name]['labels']  =  labels[0:min(len(labels), len(patches))]
	self._data[name]['matches'] = matches

	# debug info after loading
	if debug:
	     print '-- Dataset loaded:    %s' % name
	     print '-- Number of images:  %s' % len(img_files)
	     print '-- Number of patches: %s' % len(self._data[name]['patches'])
	     print '-- Number of labels:  %s' % len(self._data[name]['labels'])
	     print '-- Number of ulabels: %s' % len(np.unique(labels))
	     print '-- Number of matches: %s' % len(matches)

    def generate_triplets(self, name, n_triplets):
        # retrieve loaded patches and labels
        labels = self._get_labels(name)
        # group labels in order to have O(1) search
        count = collections.Counter(labels)
        # index the labels in order to have O(1) search
        indices = self._create_indices(labels)
        # range for the sampling
        labels_size = len(labels) - 1
        # triplets ids
        triplets = []
	# generate the triplets
        pbar = tqdm(xrange(n_triplets))
	for x in pbar:
            pbar.set_description('Generating triplets %s' % name)
            # pick a random id for anchor
            idx = random.randint(0, labels_size)
            # count number of anchor occurrences
	    num_samples = count[labels[idx]]
            # the global index to the id
            begin_positives = indices[labels[idx]]
            # generate two samples to the id
            offset_a, offset_p = random.sample(xrange(num_samples), 2)
            idx_a = begin_positives + offset_a
            idx_p = begin_positives + offset_p
            # find index of the same 3D but not same as before
            idx_n = random.randint(0, labels_size)
            while labels[idx_n] == labels[idx_a] and \
                  labels[idx_n] == labels[idx_p]:
                idx_n = random.randint(0, labels_size)
	    # pick and append triplets to the buffer
	    triplets.append([idx_a, idx_p, idx_n])
        return triplets
    
    def generate_stats(self, name):
        print '-- Computing dataset mean: %s ...' % name
        # compute the mean and std of all patches
        patches = self._get_patches(name)
        mean, std = self._compute_mean_and_std(patches)
        print '-- Computing dataset mean: %s ... OK' % name
        print '-- Mean: %s' % mean
        print '-- Std : %s' % std
        return mean, std

    def prune(self, name, min=2):
        labels  = self._get_labels(name)
        # filter the labels
        ids, labels = self._prune(labels, min)
        # return only the filtered patches
        return ids, labels

    def _prune(self, labels, min):
        # count the number of labels
        c = collections.Counter(labels)
        # create a list with globals indices
        ids = range(len(labels))
        # remove ocurrences
        ids, labels = self._rename_and_prune(labels, ids, c, min)
        return np.asarray(ids), np.asarray(labels)
 
    def _rename_and_prune(self, labels, ids, c, min):
        count, x = 0, 0
        labels_new, ids_new = [[] for _ in range(2)]
        while x < len(labels):
            num = c[labels[x]]
            if num >= min:
                for i in xrange(num):
                    labels_new.append(count)
                    ids_new.append(ids[x+i])
                count += 1
            x += num
        return ids_new, labels_new
 
    def _load_matches(self, base_dir, name):
        """
	Return a list containing the ground truth matches
	"""
        fname = os.path.join(base_dir, name, 'm50_50000_50000_0.txt')
        assert os.path.isfile(fname), 'Not a file: %s' % file 
        # read file and keep only 3D point ID and 1 if is the same, otherwise 0
	matches = []
        with open(fname, 'r') as f:
            for line in f:
                l = line.split()
		matches.append([int(l[0]), int(l[3]), int(l[1] == l[4])])
	return np.asarray(matches)

    def _load_image_fnames(self, base_dir, dir_name):
        """
        Return a list with the file names of the images containing the patches
        """
        files = []
        # find those files with the specified extension
        dataset_dir = os.path.join(base_dir, dir_name)
        for file in os.listdir(dataset_dir):
            if file.endswith(self.IMAGE_EXT):
                files.append(os.path.join(dataset_dir, file))
        return sorted(files) # sort files in ascend order to keep relations

    def _load_patches(self, img_files, name, patch_size, num_channels):
        """
        Return a list containing all the patches
        """
        patches_all = []
        # reduce the number of files to load if we are in testing mode
        img_files = img_files [0:self.n] if self.test else img_files
        # load patches
        pbar = tqdm(img_files)
        for file in pbar:
            pbar.set_description('Loading dataset %s' % name)
            # pick file name
            assert os.path.isfile(file), 'Not a file: %s' % file
            # load the image containing the patches and convert to float point
            # and make sure that que use only one single channel
            img = cv2.imread(file)[:,:,0] / 255.
            # split the image into patches and
            # add patches to buffer as individual elements
            patches_row = np.split(img, self.PATCHES_PER_ROW, axis=0)
            for row in patches_row:
                patches = np.split(row, self.PATCHES_PER_ROW, axis=1)
                for patch in patches:
                    # resize the patch
                    patch_resize = cv2.resize(patch, (patch_size,patch_size))
                    # convert to tensor [w x h x d]
                    patch_tensor = patch_resize.reshape(patch_size,
                                                        patch_size,
                                                        num_channels)
                    patches_all.append(patch_tensor)
        return np.asarray(patches_all) if not self.test \
            else np.asarray(patches_all[0:self.n])

    def _load_labels(self, base_dir, dir_name):
	"""
	Return a list containing all the labels for each patch
	"""
        info_fname = os.path.join(base_dir, dir_name, 'info.txt')
        assert os.path.isfile(info_fname), 'Not a file: %s' % file 
        # read file and keep only 3D point ID
	labels = []
        with open(info_fname, 'r') as f:
             for line in f:
		  labels.append(int(line.split()[0]))
	return np.asarray(labels) if not self.test \
            else np.asarray(labels[0:self.n])

    def _create_indices(self, labels):
        old = labels[0]
        indices = dict()
        indices[old] = 0
        for x in xrange(len(labels)-1):
            new = labels[x+1]
            if old != new:
                indices[new] = x+1
            old = new
        return indices
 
    def _compute_mean_and_std(self, patches):
        """
        Return the mean and the std given a set of patches.
        """
        assert len(patches) > 0, 'Patches list is empty!'
        # compute the mean
        mean = np.mean(patches)
	# compute the standard deviation
	std  = np.std(patches)
        return mean, std

    def _get_data(self, name):
        assert self._data[name] is not None, 'Dataset not loaded: %s' % name
        return self._data[name]

    def _get_patches(self, name):
        return self._get_data(name)['patches']

    def _get_matches(self, name):
        return self._get_data(name)['matches']

    def _get_labels(self, name):
        return self._get_data(name)['labels']
