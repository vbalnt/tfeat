import os
import random
import errno
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torchvision import transforms
import cv2

def augment_patches(patches):
    if random.random() > 0.5:
        rot = np.random.randint(1, 4)
        patches = [np.rot90(p,rot) for p in patches]
    if random.random() > 0.5:
        patches = [np.flipud(p) for p in patches]
    return patches

def resize_patches(patches,N):
    patches = [cv2.resize(p,(N,N)) for p in patches]
    return patches

# def norm_patches(patches):
#     patches = [(p-0.48544601108437)/0.18649942105166 for p in patches]
    
#     return patches


class PhotoTour(data.Dataset):
    """`Learning Local Image Descriptors Data <http://phototour.cs.washington.edu/patches/default.htm>`_ Dataset.


    Args:
        root (string): Root directory where images are.
        name (string): Name of the dataset to load.
        mode (string): Mode of output (training only). `pairs` or `triplets`. Default is pairs. 
            Testing mode is always pairs.
        nsamples  (int):    Number of training pairs/triplets. Default is 10e6.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    urls = {
        'notredame_harris': [
            'http://matthewalunbrown.com/patchdata/notredame_harris.zip',
            'notredame_harris.zip',
            '69f8c90f78e171349abdf0307afefe4d'
        ],
        'yosemite_harris': [
            'http://matthewalunbrown.com/patchdata/yosemite_harris.zip',
            'yosemite_harris.zip',
            'a73253d1c6fbd3ba2613c45065c00d46'
        ],
        'liberty_harris': [
            'http://matthewalunbrown.com/patchdata/liberty_harris.zip',
            'liberty_harris.zip',
            'c731fcfb3abb4091110d0ae8c7ba182c'
        ],
        'notredame': [
            'http://icvl.ee.ic.ac.uk/vbalnt/notredame.zip',
            'notredame.zip',
            '509eda8535847b8c0a90bbb210c83484'
        ],
        'yosemite': [
            'http://icvl.ee.ic.ac.uk/vbalnt/yosemite.zip',
            'yosemite.zip',
            '533b2e8eb7ede31be40abc317b2fd4f0'
        ],
        'liberty': [
            'http://icvl.ee.ic.ac.uk/vbalnt/liberty.zip',
            'liberty.zip',
            'fdd9152f138ea5ef2091746689176414'
        ],
    }
    mean = {'notredame': 0.4854, 'yosemite': 0.4844, 'liberty': 0.4437,
            'notredame_harris': 0.4854, 'yosemite_harris': 0.4844, 'liberty_harris': 0.4437}
    std = {'notredame': 0.1864, 'yosemite': 0.1818, 'liberty': 0.2019,
           'notredame_harris': 0.1864, 'yosemite_harris': 0.1818, 'liberty_harris': 0.2019}
    lens = {'notredame': 468159, 'yosemite': 633587, 'liberty': 450092,
            'liberty_harris': 379587, 'yosemite_harris': 450912, 'notredame_harris': 325295}
    image_ext = 'bmp'
    info_file = 'info.txt'
    matches_files = 'm50_100000_100000_0.txt'

    def __init__(self, root, name, mode='pairs', nsamples = 1e6, train=True, transform=None, download=False, augment=False):
        self.root = os.path.expanduser(root)
        self.name = name
        self.data_dir = os.path.join(self.root, name)
        self.data_down = os.path.join(self.root, '{}.zip'.format(name))
        self.data_file = os.path.join(self.root, '{}.pt'.format(name))

        self.train = train
        self.mode = mode
        self.nsamples = int(nsamples)
        self.mean = self.mean[name]
        self.std = self.std[name]
        self.augment = augment
        
        if download:
            self.download()

        if not self._check_datafile_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # load the serialized data
        self.data, self.labels, self.matches = torch.load(self.data_file)
        self.data_len = self.data.shape[0]
        self.labels = self.labels.numpy()
        self.ids = np.unique(self.labels)
        self.nids = self.ids.shape[0]

        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (data1, data2, matches)
        """

        # fix the random seed issue with numpy and multiprocessing
        #seed = random.randrange(4294967295)
        #np.random.seed(seed=seed)

        # testing mode: 100k pairs from Brown's original paper
        # note: testing mode can only be pairs
        if not self.train:
            m = self.matches[index]
            data1, data2 = self.data[m[0]], self.data[m[1]]
            patches = [data1, data2]
            patches = [p.numpy() for p in patches]
            data1,data2 = resize_patches(patches,32)
#             data1, data2 = norm_patches(patches)
            return data1, data2, m[2]

        # train mode: either random pairs or random triplets 
        if self.train:
            if self.mode == 'pairs':
                lbl = random.randint(0, 1)
                if lbl==0: #negative pair
                    idx_L = random.randrange(self.data_len)
                    L_label = self.labels[idx_L]
                    idx_R = random.randrange(self.data_len)
                    R_label = self.labels[idx_R]
                    while R_label==L_label :
                        idx_R = random.randrange(self.data_len)
                        R_label = self.labels[
                            idx_R]
                else: #positive pair
                    idx_L = random.randrange(self.data_len)
                    L_label = self.labels[idx_L]
                    label_search_range_start = max(0,idx_L-20)
                    label_search_range_end = min(self.data_len,idx_L+20)
                    sub_labels = self.labels[label_search_range_start:label_search_range_end]
                    mask_pos = np.where(sub_labels==L_label)[0]
                    idx_L,idx_R = np.random.choice(mask_pos,2, replace=False)
                    idx_L = idx_L + label_search_range_start
                    idx_R = idx_R + label_search_range_start

                data1, data2 = self.data[idx_L], self.data[idx_R]
                patches = [data1,data2]
                patches = [p.numpy() for p in patches]
                [data1,data2] = resize_patches(patches,32)

                return data1,data2,lbl

            elif self.mode == 'triplets':
                idx_a = random.randrange(self.data_len)
                a_label = self.labels[idx_a]
                idx_n = random.randrange(self.data_len)
                n_label = self.labels[idx_n]
                while n_label==a_label :
                    idx_n = random.randrange(self.data_len)
                    n_label = self.labels[idx_n]
                #find the next idx_p
                label_search_range_start = max(0,idx_a-20)
                label_search_range_end = min(self.data_len,idx_a+20)
                sub_labels = self.labels[label_search_range_start:label_search_range_end]
                mask_pos = np.where(sub_labels==a_label)[0]
                idx_a,idx_p = np.random.choice(mask_pos,2, replace=False)
                idx_a = idx_a + label_search_range_start
                idx_p = idx_p + label_search_range_start
                    
                data_a, data_p, data_n = self.data[idx_a], self.data[idx_p], self.data[idx_n]
                patches = [data_a, data_p, data_n]
                patches = [p.numpy() for p in patches]
                if self.augment:
                    patches = augment_patches(patches)
                patches = resize_patches(patches,32)
#                 patches = norm_patches(patches)
                return patches
            else:
                raise ValueError('Uknown training output mode. Valid ones are pairs,triplets')


    def __len__(self):
        if not self.train:
            return self.matches.size(0)
        return self.nsamples

    def _check_datafile_exists(self):
        return os.path.exists(self.data_file)

    def _check_downloaded(self):
        return os.path.exists(self.data_dir)

    def download(self):
        if self._check_datafile_exists():
            print('# Found cached data {}'.format(self.data_file))
            return

        if not self._check_downloaded():
            # download files
            url = self.urls[self.name][0]
            filename = self.urls[self.name][1]
            md5 = self.urls[self.name][2]
            fpath = os.path.join(self.root, filename)

            download_url(url, self.root, filename, md5)

            print('# Extracting data {}\n'.format(self.data_down))

            import zipfile
            with zipfile.ZipFile(fpath, 'r') as z:
                z.extractall(self.data_dir)

            os.unlink(fpath)

        # process and save as torch files
        print('# Caching data {}'.format(self.data_file))

        dataset = (
            read_image_file(self.data_dir, self.image_ext, self.lens[self.name]),
            read_info_file(self.data_dir, self.info_file),
            read_matches_files(self.data_dir, self.matches_files)
        )

        with open(self.data_file, 'wb') as f:
            torch.save(dataset, f)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def read_image_file(data_dir, image_ext, n):
    """Return a Tensor containing the patches
    """
    def PIL2array(_img):
        """Convert PIL image type to numpy 2D array
        """
        return np.array(_img.getdata(), dtype=np.uint8).reshape(64, 64)

    def find_files(_data_dir, _image_ext):
        """Return a list with the file names of the images containing the patches
        """
        files = []
        # find those files with the specified extension
        for file_dir in os.listdir(_data_dir):
            if file_dir.endswith(_image_ext):
                files.append(os.path.join(_data_dir, file_dir))
        return sorted(files)  # sort files in ascend order to keep relations

    patches = []
    list_files = find_files(data_dir, image_ext)

    for fpath in list_files:
        img = Image.open(fpath)
        for y in range(0, 1024, 64):
            for x in range(0, 1024, 64):
                patch = img.crop((x, y, x + 64, y + 64))
                patches.append(PIL2array(patch))
    return torch.ByteTensor(np.array(patches[:n]))


def read_info_file(data_dir, info_file):
    """Return a Tensor containing the list of labels
       Read the file and keep only the ID of the 3D point.
    """
    labels = []
    with open(os.path.join(data_dir, info_file), 'r') as f:
        labels = [int(line.split()[0]) for line in f]
    return torch.LongTensor(labels)


def read_matches_files(data_dir, matches_file):
    """Return a Tensor containing the ground truth matches
       Read the file and keep only 3D point ID.
       Matches are represented with a 1, non matches with a 0.
    """
    matches = []
    with open(os.path.join(data_dir, matches_file), 'r') as f:
        for line in f:
            l = line.split()
            matches.append([int(l[0]), int(l[3]), int(l[1] == l[4])])
    return torch.LongTensor(matches)
from tqdm import tqdm 
import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',help='this is root directory where training/evaluation data are stored/downloaded',required=True)
    parser.add_argument('--mode',help='training mode of tfeat descriptor, can be pair or triplet,it is enum type, could be pair|triplet',required=False,default='triplets')
    parser.add_argument('--name',help='The name of dataset to test,it is enum type, could be liberty|yosemite|notredame ',required=False,default='liberty')

    args = vars(parser.parse_args())
    
    data_path = args['data_dir']
    dataset_name = args['name']
    train_db = PhotoTour(data_path,dataset_name, download=True, train=True, mode = args['mode'], augment = True, nsamples=1000000)
    train_loader = torch.utils.data.DataLoader(train_db,
                                             batch_size=300, shuffle=False,
                                             num_workers=2)
    for batch_idx, (data_a, data_p, data_n) in tqdm(enumerate(train_loader)): 
        if batch_idx==100000: 
            break