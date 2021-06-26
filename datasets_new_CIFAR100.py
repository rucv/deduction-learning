from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

import pdb

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

#from .vision import VisionDataset
from vision import VisionDataset
#from .utils import check_integrity, download_and_extract_archive
from utils import check_integrity, download_and_extract_archive

############ wenchi 01/18/2020 ##############
import random
#m0 = len(dict0)
#m1 = len(dict1)
#############################################

class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        dict0 = [2,11,35,46,98]     #"people"#
        dict1 = [1,3,4,6,7,14,15,18,21,24,26,27,29,30,31,32,34,36,38,42,43,44,45,50,55,63,64,65,66,67,72,73,74,75,77,78,79,80,88,91,93,95,97,99]      #"animal"#
        dict2 = [5,9,10,12,16,19,20,22,25,28,39,40,41,61,69,84,86,87,94]      #"man-made"#
        dict3 = [8,13,48,58,81,85,89,90]        #"transportation"#
        dict4 = [0,53,57]                       #"food"#
        dict5 = [47,51,52,54,56,59,62,70,82,83,92,96]        #"plants"#
        dict6 = [17,37,68,76]                   #"building"#
        dict7 = [23,33,49,60,71]                #"nature"# 

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            #pdb.set_trace()
            print("load data!!!")
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
############### wenchi 01/18/2020 ########################################
                m = len(entry['fine_labels'])
                res_pos = random.sample(range(m), int(m*0.1)) #[random.randrange(0, m, 1) for i in range(int(m*0.1))]        ### randomly select 1000 numbers from [0~9999]
                for i in res_pos:
                    if entry['fine_labels'][i] in dict0:
                       dict0_tmp = [x for x in dict0 if x != entry['fine_labels'][i]]#dict0.pop(entry['labels'][i])
                       m0 = len(dict0_tmp)                       
                       temp = random.sample(range(m0), 1)#[random.randrange(0, m0, 1) for i in range(1)]
                       entry['fine_labels'][i] = dict0_tmp[temp[0]]
                       #dict0 = [0,1,8,9]
                    elif entry['fine_labels'][i] in dict1:
                       dict1_tmp = [x for x in dict1 if x != entry['fine_labels'][i]]#dict1.pop(entry['labels'][i])
                       m1 = len(dict1_tmp)
                       temp = random.sample(range(m1), 1)#[random.randrange(0, m1, 1) for i in range(1)]
                       entry['fine_labels'][i] = dict1_tmp[temp[0]]
                       #dict1 = [2,3,4,5,6,7]
                    elif entry['fine_labels'][i] in dict2:
                       dict2_tmp = [x for x in dict2 if x != entry['fine_labels'][i]]#dict1.pop(entry['labels'][i])
                       m2 = len(dict2_tmp)
                       temp = random.sample(range(m2), 1)#[random.randrange(0, m1, 1) for i in range(1)]
                       entry['fine_labels'][i] = dict2_tmp[temp[0]]
                    elif entry['fine_labels'][i] in dict3:
                       dict3_tmp = [x for x in dict3 if x != entry['fine_labels'][i]]#dict1.pop(entry['labels'][i])
                       m3 = len(dict3_tmp)
                       temp = random.sample(range(m3), 1)#[random.randrange(0, m1, 1) for i in range(1)]
                       entry['fine_labels'][i] = dict3_tmp[temp[0]]
                    elif entry['fine_labels'][i] in dict4:
                       dict4_tmp = [x for x in dict4 if x != entry['fine_labels'][i]]#dict1.pop(entry['labels'][i])
                       m4 = len(dict4_tmp)
                       temp = random.sample(range(m4), 1)#[random.randrange(0, m1, 1) for i in range(1)]
                       entry['fine_labels'][i] = dict4_tmp[temp[0]]
                    elif entry['fine_labels'][i] in dict5:
                       dict5_tmp = [x for x in dict5 if x != entry['fine_labels'][i]]#dict1.pop(entry['labels'][i])
                       m5 = len(dict5_tmp)
                       temp = random.sample(range(m5), 1)#[random.randrange(0, m1, 1) for i in range(1)]
                       entry['fine_labels'][i] = dict5_tmp[temp[0]]
                    elif entry['fine_labels'][i] in dict6:
                       dict6_tmp = [x for x in dict6 if x != entry['fine_labels'][i]]#dict1.pop(entry['labels'][i])
                       m6 = len(dict6_tmp)
                       temp = random.sample(range(m6), 1)#[random.randrange(0, m1, 1) for i in range(1)]
                       entry['fine_labels'][i] = dict6_tmp[temp[0]]
                    elif entry['fine_labels'][i] in dict7:
                       dict7_tmp = [x for x in dict7 if x != entry['fine_labels'][i]]#dict1.pop(entry['labels'][i])
                       m7 = len(dict7_tmp)
                       temp = random.sample(range(m7), 1)#[random.randrange(0, m1, 1) for i in range(1)]
                       entry['fine_labels'][i] = dict7_tmp[temp[0]]                                       
##########################################################################
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")



class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
