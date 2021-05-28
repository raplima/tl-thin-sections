"""Defines a pl.LightningDataModule to load image data organized in folders.
"""
import os

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch import stack
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class DataFolders(pl.LightningDataModule):
    """Assumes the data is organized as torch ImageFolders.
    (samples are stored in different folders).
    Additionally, assumes train and test folders:

    root:
        -train:
            -class1:
                -sample1
                -sample2
                ...
            -class2:
                -sample1
                -sample2
                ...
        -test:
            -class1:
                -sample1
                -sample2
                ...
            -class2:
                -sample1
                -sample2
                ...
    """

    def __init__(self, data_dir='./',
                 batch_size=64,
                 val_prop=0.2,
                 dset_mean=None,
                 dset_std=None,
                 size_hw=(224, 224),
                 five_crop=False):
        """

        Args:
            data_dir (str, optional): Location of the image folders. Defaults to './'.
            batch_size (int, optional): Batch size. Defaults to 64.
            val_prop (float, optional): Proportion of the dataset to include in the
                validation split. Defaults to 0.2.
            dset_mean (list): Dataset mean values. Defaults to None.
            dset_std (list): Dataset standard deviation values. Defaults to None
            size_hw (sequence or int): Desired image size. 
                If size is a sequence like (h, w), output size will be matched to this.
            five_crop (boolean): Whether or not use five crop augmentaiton
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.val_prop = val_prop

        self.five_crop = five_crop

        # define attributes that will be populated later:
        self.dset = None
        self.dset_test = None

        self.dm_train = None
        self.dm_val = None
        self.dm_test = None

        self.classes = None
        self.class_to_idx = None
        self.idx_to_class = None

        normalize = transforms.Normalize(mean=dset_mean,
                                         std=dset_std)

        if five_crop:

            # It seems Windows does not work well with multiprocessing Lambda:
            self.num_workers = 0

            self.transform = transforms.Compose([
                transforms.FiveCrop(size_hw),  # this is a list of PIL Images
                transforms.Lambda(lambda crops: stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                normalize,
            ])

            self.transform_train = transforms.Compose([
                transforms.FiveCrop(size_hw),  # this is a list of PIL Images
                transforms.Lambda(lambda crops: stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                normalize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ])

        else:
            self.num_workers = os.cpu_count()-1

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                transforms.Resize(size=size_hw)],
            )

            self.transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                transforms.Resize(size=size_hw),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ])

    def prepare_data(self):
        """prepare the dataset, set up some attributes
        """
        self.dset = ImageFolder(os.path.join(self.data_dir, 'train'),
                                transform=self.transform_train)
        self.classes = self.dset.classes
        self.class_to_idx = self.dset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def setup(self, stage=None):
        """Setup the DataModule.

        Args:
            stage ([type], optional): Train or Test. Defaults to None.
        """

        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:

            label_indexes = np.array(
                [[idx, res[1]] for idx, res in enumerate(self.dset.samples)])
            x_ind_train = label_indexes[:, 0]
            y_train = label_indexes[:, 1]

            # do the split:
            x_ind_train, x_ind_val, y_train, _ = train_test_split(x_ind_train, y_train,
                                                                  test_size=self.val_prop,
                                                                  stratify=y_train)

            # the validation subset should not have data augmentation:
            dset_val = ImageFolder(os.path.join(self.data_dir, 'train'),
                                   transform=self.transform)

            # select the subsets based on train_test_splits and appropriate dsets:
            self.dm_train = Subset(self.dset, x_ind_train)
            self.dm_val = Subset(dset_val, x_ind_val)

            print(f'Train len: {len(self.dm_train)}')
            print(f'Val len: {len(self.dm_val)}')

        if stage == 'test' or stage is None:
            self.dset_test = ImageFolder(os.path.join(self.data_dir, 'test'),
                                         transform=self.transform)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        """
        Returns:
            DataLoader: torch dataloader used for training.
        """
        dm_train = DataLoader(self.dm_train,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers)
        return dm_train

    def val_dataloader(self):
        """
        Returns:
            DataLoader: torch dataloader used for validation.
        """
        dm_val = DataLoader(self.dm_val,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)
        return dm_val

    def test_dataloader(self):
        """
        Returns:
            DataLoader: torch dataloader used for testing.
        """

        dm_test = DataLoader(self.dset_test,
                             batch_size=self.batch_size,
                             num_workers=self.num_workers)
        return dm_test
