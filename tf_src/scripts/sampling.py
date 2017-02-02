import collections
import numpy as np


def generate_triplets(labels, num_triplets):
    def create_indices(_labels):
        """Generates a dict to store the index of each labels in order
           to avoid a linear search each time that we call list(labels).index(x)
        """
        old = labels[0]
        indices = dict()
        indices[old] = 0
        for x in range(len(_labels) - 1):
            new = labels[x + 1]
            if old != new:
                indices[new] = x + 1
            old = new
        return indices

    triplets = []

    # group labels in order to have O(1) search
    count = collections.Counter(labels)
    # index the labels in order to have O(1) search
    indices = create_indices(labels)
    # range for the sampling
    labels_size = len(labels) - 1
    # generate the triplets
    for x in range(num_triplets):
        # pick a random id for anchor
        idx = np.random.randint(labels_size)
        # count number of anchor occurrences
        num_samples = count[labels[idx]]
        # the global index to the id
        begin_positives = indices[labels[idx]]
        # generate two samples to the id
        offset_a, offset_p = np.random.choice(np.arange(num_samples),
                                              size=2, replace=False)
        idx_a = begin_positives + offset_a
        idx_p = begin_positives + offset_p
        # find index of the same 3D but not same as before
        idx_n = np.random.randint(labels_size)
        while labels[idx_n] == labels[idx_a] and \
            labels[idx_n] == labels[idx_p]:
            idx_n = np.random.randint(labels_size)
        # pick and append triplets to the buffer
        triplets.append([idx_a, idx_p, idx_n])
    return np.array(triplets)


def generate_batch(data, data_ids, step, batch_size, train=True):
    # compute the offset to get the correct batch
    offset = step * batch_size % len(data_ids)
    # get a sample batch from the training data
    ids = data_ids[offset:offset + batch_size]

    out1 = data[ids[:, 0]]
    out2 = data[ids[:, 1]]
    out3 = data[ids[:, 2]]

    if not train:
        out3 = ids[:, 2]

    return out1, out2, out3
