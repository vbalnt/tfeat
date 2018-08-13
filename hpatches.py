from __future__ import print_function
import resource
import torch.utils.data as data
from PIL import Image
import os
import errno
import numpy as np
import torch
import tarfile
import json
import random
import itertools
from multiprocessing import Pool, Process, Manager
from .utils import download_url, check_integrity
from ..transforms import functional as F
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))

splits_seqs = {'a': 'train', 'b': 'train', 'c': 'train', 'illum': 'test',
               'view': 'test', 'full': 'test'}
tps = ['ref', 'e1', 'e2', 'e3', 'e4', 'e5', 'h1', 'h2', 'h3', 'h4', 'h5',
       't1', 't2', 't3', 't4', 't5']


class HPatches(data.Dataset):
    """`HPatches <https://hpatches.github.io/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``hpatches-release``
            will be downloaded.
        split (string, optional): If given, only sequences from a specific split will be
        used. Default is the 'full' split, i.e. all sequences. Possible values are
            "a,b,c,illum,view,full"
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If the dataset is already downloaded, it is not
            downloaded again. Default is True.
        output (string, optional): Type of output from the dataset generator: {pairs,
            triplets, sets}. Pairs are suitable for siamese networks, triplets for
            triplet networks, and sets for generic metric learning method. Default is
            'pairs'.
        n_samples (int, optional): Number of items to generate at each epoch. Default
            is 1e7.
        cache_to_ram (bool, optional): Whether to cache to all sequences to RAM to avoid
            constant disk I/O. Default is True. Note that this needs ~10GB of free memory.
        n_negs (int, optional): Number of negative patches returned in the case of the
            'sets' output. Default is 16.
    """
    url = 'http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-release.tar.gz'
    splits_url = 'http://icvl.ee.ic.ac.uk/vbalnt/hpatches/splits.json'
    filename = 'hpatches-release.tar.gz'
    folder = 'hpatches-release'
    md5 = '0ab830d37fceb2b4c86cb1cc6cc79a61'

    def __init__(self, root, split='full', transform=None, download=False, output="pairs",
                 n_samples=1e7, cache_to_ram=False, n_negs=16):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.output = output
        self.n_samples = int(n_samples)
        self.cache_to_ram = cache_to_ram
        self.n_negs = n_negs

        if not os.path.isfile(os.path.join(self.root, 'splits.json')):
            download_url(self.splits_url, self.root, 'splits.json', 'b08cae8889120e339f5512c10fad4d7f')
        self.all_splits = json.load(open(os.path.join(self.root, 'splits.json')))
        self.all_seqs = self.all_splits['full']['test']
        self.all_seqs_all_tps = list(itertools.product(self.all_seqs, tps))

        if download:
            self.do_download()

        try:
            self.sequences = self.all_splits[self.split][splits_seqs[self.split]]
        except:
            raise RuntimeError('Unknown split. ' +
                               'Valid ones are: a,b,c,illum,view,full')

        if cache_to_ram:
            manager = Manager()
            d = manager.dict()
            job = [Process(target=self.do_cache_sequence_to_ram, args=(d, par)) for par in self.all_seqs]
            _ = [p.start() for p in job]
            _ = [p.join() for p in job]
            self.cached_sequences = d

        if not self.do_check_exists():
            raise RuntimeError('Dataset not found. ' +
                               'Please use download=True to download it')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'left':, 'right':, 'label':} when `output` is `pairs`.
                           OR
            dict: {'anchor':, 'pos':, 'neg':} when `output` is `triplets`.
                           OR
            dict: {'pos':, 'neg':} when `output` is `sets`.
        """
        # fix the random seed issue with numpy and multiprocessing
        seed = random.randrange(4294967295)
        np.random.seed(seed=seed)

        # randomly sample a sequence
        seq = np.random.choice(self.sequences)
        if self.output == 'pairs':
            lbl = random.randint(0, 1)
            tpL, tpR = np.random.choice(tps, 2, replace=False)
            if self.cache_to_ram:
                patchesL = self.cached_sequences[seq][tpL]
                patchesR = self.cached_sequences[seq][tpR]
            else:
                patchesL = torch.load(os.path.join(self.root, self.folder, seq, tpL + ".pth"))
                patchesR = torch.load(os.path.join(self.root, self.folder, seq, tpR + ".pth"))

            n_patches = patchesL.size(0)
            idx_1, idx_2 = np.random.choice(range(n_patches), 2, replace=False)
            patchL = patchesL[idx_1]
            if lbl:
                patchR = patchesR[idx_1]
            else:
                patchR = patchesR[idx_2]

            patchL = F.to_pil_image(patchL)
            patchR = F.to_pil_image(patchR)
            if self.transform is not None:
                patchL = self.transform(patchL)
                patchR = self.transform(patchR)

            sample = {'left': patchL, 'right': patchR, 'label': lbl}
            return sample

        elif self.output == 'triplets':
            tpL, tpR = np.random.choice(tps, 2, replace=False)

            if self.cache_to_ram:
                patchesL = self.cached_sequences[seq][tpL]
                patchesR = self.cached_sequences[seq][tpR]
            else:
                patchesL = torch.load(os.path.join(self.root, self.folder, seq, tpL + ".pth"))
                patchesR = torch.load(os.path.join(self.root, self.folder, seq, tpR + ".pth"))

            n_patches = patchesL.size(0)
            idx_1, idx_2 = np.random.choice(range(n_patches), 2, replace=False)
            patch_a = patchesL[idx_1]
            patch_p = patchesR[idx_1]
            patch_n = patchesR[idx_2]

            patch_a = F.to_pil_image(patch_a)
            patch_p = F.to_pil_image(patch_p)
            patch_n = F.to_pil_image(patch_n)
            if self.transform is not None:
                patch_a = self.transform(patch_a)
                patch_p = self.transform(patch_p)
                patch_n = self.transform(patch_n)

            sample = {'anchor': patch_a, 'pos': patch_p, 'neg': patch_n}
            return sample

        elif self.output == 'sets':
            patches_pos = []
            patches_neg = []

            seq_patches = {}
            for tp in tps:
                if self.cache_to_ram:
                    tp_patches = self.cached_sequences[seq][tp]
                else:
                    tp_patches = torch.load(os.path.join(self.root, self.folder, seq, tp + ".pth"))
                seq_patches[tp] = tp_patches

            n_patches = seq_patches['ref'].size(0)
            idxs = np.random.choice(range(n_patches), self.n_negs + 1, replace=False)

            patches_pos = []
            patches_neg = []

            for ik, tp in enumerate(tps):
                patch_pos = seq_patches[tp][idxs[0]]
                patch_neg = seq_patches[tp][idxs[ik + 1]]
                patch_pos = F.to_pil_image(patch_pos)
                patch_neg = F.to_pil_image(patch_neg)
                if self.transform is not None:
                    patch_pos = self.transform(patch_pos)
                    patch_neg = self.transform(patch_neg)
                patches_pos.append(patch_pos)
                patches_neg.append(patch_neg)

            patches_pos = torch.stack(patches_pos)
            patches_neg = torch.stack(patches_neg)
            sample = {'pos': patches_pos, 'neg': patches_neg}
            return sample
        else:
            raise RuntimeError('Unknown output type. ' +
                               'Supported ones are pairs,triplets,sets')

    def __len__(self):
        return self.n_samples

    def do_cache_sequence_to_disk(self, par):
        seq = par[0]
        tp = par[1]
        img = Image.open(os.path.join(self.root, self.folder, seq, tp + ".png"))
        img_tensor = 255 * F.to_tensor(img).squeeze()
        n_patches = int(img_tensor.size(0) / 65)
        patches = torch.stack(torch.chunk(img_tensor, n_patches, 0)).unsqueeze(1).byte()
        torch.save(patches, os.path.join(self.root, self.folder, seq, tp + ".pth"))

    def do_cache_sequence_to_ram(self, d, seq):
        all_patches = {}
        for tp in tps:
            patches = torch.load(os.path.join(self.root, self.folder, seq, tp + ".pth"))
            all_patches[tp] = patches
        d[seq] = all_patches

    def do_check_exists(self):
        return os.path.exists(os.path.join(self.root, self.folder))

    def do_download(self):
        p = Pool()
        from six.moves import urllib
        import gzip

        if self.do_check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        download_url(self.url, self.root, self.filename, self.md5)
        print("> Extracting the dataset.")
        tar = tarfile.open(os.path.join(self.root, self.filename), 'r:gz')
        tar.extractall(os.path.join(self.root))
        tar.close()

        print("> Caching the images to pytorch tensors.")
        print("  This only needs to be done once.")
        p.map(self.do_cache_sequence_to_disk, self.all_seqs_all_tps)
