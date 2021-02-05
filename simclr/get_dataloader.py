import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import lmdb

import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

from utils import GaussianBlur, FixedRandomRotation
import neptune


def get_dataloader(opt):
    if opt.dataset == 'cam' or opt.dataset == 'patchcam':
        train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset = get_camelyon_dataloader(
            opt
        )
    elif opt.dataset == 'multidata':
        train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset = get_multidata_dataloader(opt)
    else:
        raise Exception("Invalid option")

    return (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    )


def get_transforms(eval=False, aug=None):

    trans = []

    if aug["resize"]:
       trans.append(transforms.Resize(aug["resize"]))

    if aug["randcrop"] and not eval:
        trans.append(transforms.RandomResizedCrop(aug["randcrop"], scale=(0.2, 1.)))

    if aug["randcrop"] and eval:
        trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip(p=0.5))
        trans.append(transforms.RandomVerticalFlip(p=0.5))

    if aug["color_jitter"] and not eval:
        trans.append(transforms.RandomApply(
            [transforms.ColorJitter(aug["color_jitter"], aug["color_jitter"], aug["color_jitter"], aug["color_value"])], p=0.8))

    if aug["gaussian_blur"] and not eval:
        trans.append(transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5))

    if aug["rotation"] and not eval:
        # rotation_transform = FixedRandomRotation(angles=[0, 90, 180, 270])
        trans.append(FixedRandomRotation(angles=[0, 90, 180, 270]))

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    else:
        trans.append(transforms.ToTensor())

    # trans = transforms.Compose(trans)
    return trans


def get_weighted_sampler(dataset, num_samples):
    df = dataset.dataframe
    # Get number of sampler per label. Weight = 1/num sampels
    class_weights = { row.label: 1/row[0] for _, row in df.groupby(['label']).size().reset_index().iterrows()}
    print(class_weights)
    # Set weights per sample in dataset
    weights = [class_weights[row.label] for _, row in df.iterrows()]
    return WeightedRandomSampler(weights=weights, num_samples=num_samples)

def get_lnco_weighted_sampler(dataset, num_samples):
    df = dataset.dataframe
    # Get number of sampler per label. Weight = 1/num sampels
    class_weights = { row.label_int: 1/row[0] for _, row in df.groupby(['label_int']).size().reset_index().iterrows()}
    print(class_weights)
    # Set weights per sample in dataset
    weights = [class_weights[row.label_int] for _, row in df.iterrows()]
    return WeightedRandomSampler(weights=weights, num_samples=num_samples)

def clean_data(img_dir, dataframe):
    """ Clean the data """
    for idx, row in dataframe.iterrows():
        if not os.path.isfile(f'{img_dir}/{row.filename}'):
            print(f"Removing non-existing file from dataset: {img_dir}/{row.filename}")
            dataframe = dataframe.drop(idx)
    return dataframe


def get_dataframes(opt):
    if os.path.isfile(opt.training_data_csv):
        print("reading csv file: ", opt.training_data_csv)
        train_df = pd.read_csv(opt.training_data_csv)
    else:
        raise Exception(f'Cannot find file: {opt.training_data_csv}')

    if os.path.isfile(opt.test_data_csv):
        print("reading csv file: ", opt.test_data_csv)
        test_df = pd.read_csv(opt.test_data_csv)
    else:
        raise Exception(f'Cannot find file: {opt.test_data_csv}')

    if opt.trainingset_split:
        # Split train_df into train and val
        slide_ids = train_df.slide_id.unique()
        random.shuffle(slide_ids)
        train_req_ids = []
        valid_req_ids = []
        # Take same number of slides from each site
        training_size = int(len(slide_ids)*opt.trainingset_split)
        validation_size = len(slide_ids) - training_size
        train_req_ids.extend([slide_id for slide_id in slide_ids[:training_size]])  # take first
        valid_req_ids.extend([
            slide_id for slide_id in slide_ids[training_size:training_size+validation_size]])  # take last

        print("train / valid / total")
        print(f"{len(train_req_ids)} / {len(valid_req_ids)} / {len(slide_ids)}")

        val_df = train_df[train_df.slide_id.isin(valid_req_ids)] # First, take the slides for validation
        train_df = train_df[train_df.slide_id.isin(train_req_ids)] # Update train_df

    else:
        if os.path.isfile(opt.validation_data_csv):
            print("reading csv file: ", opt.validation_data_csv)
            val_df = pd.read_csv(opt.validation_data_csv)
        else:
            raise Exception(f'Cannot find file: {opt.test_data_csv}')

    if opt.balanced_validation_set:
        print('Use uniform validation set')
        samples_to_take = val_df.groupby('label').size().min()
        val_df = pd.concat([val_df[val_df.label == label].sample(samples_to_take) for label in val_df.label.unique()])

        print('Use uniform test set')
        samples_to_take = test_df.groupby('label').size().min()
        test_df = pd.concat([test_df[test_df.label == label].sample(samples_to_take) for label in test_df.label.unique()])

    #train_df = train_df.sample(1000)
    #val_df = val_df.sample(100)
    #test_df = test_df.sample(1000)

    train_df = clean_data(opt.data_input_dir, train_df)
    val_df = clean_data(opt.data_input_dir, val_df)
    test_df = clean_data(opt.data_input_dir, test_df)

    return train_df, val_df, test_df



def get_camelyon_dataloader(opt):
    base_folder = opt.data_input_dir
    print('opt.data_input_dir: ', opt.data_input_dir)

    aug = {
        "cam": {
            "resize": None,
            "randcrop": 224,
            "flip": True,
            "color_jitter": 0.8,
            "color_value": 0.2,
            "grayscale": opt.grayscale,
            "gaussian_blur": True,
            "rotation": True,
            "mean": [0.4914, 0.4822, 0.4465],  # values for train+unsupervised combined
            "std": [0.2023, 0.1994, 0.2010],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        } ,
        "cam_supervised": {
            "resize": None,
            "randcrop": 224,
            "flip": True,
            "color_jitter": 0.8,
            "color_value": 0.2,
            "grayscale": opt.grayscale,
            "gaussian_blur": None,
            "rotation": True,
            "mean": [0.4914, 0.4822, 0.4465],  # values for train+unsupervised combined
            "std": [0.2023, 0.1994, 0.2010],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }
    }
    aug_choice = "cam_supervised" if opt.train_supervised else "cam"
    transform_train = transforms.Compose(get_transforms(eval=False, aug=aug[aug_choice]))
    transform_valid = transforms.Compose(get_transforms(eval=True, aug=aug[aug_choice]))

    train_df, val_df, test_df = get_dataframes(opt)

    print("training patches: ", train_df.groupby('label').size())
    print("Validation patches: ", val_df.groupby('label').size())
    print("Test patches: ", test_df.groupby('label').size())

    print("Saving training/val set to file")
    train_df.to_csv(f'{opt.log_path}/training_patches.csv', index=False)
    val_df.to_csv(f'{opt.log_path}/val_patches.csv', index=False)

    if opt.dataset == 'cam':
        train_dataset = ImagePatchesDataset(train_df, image_dir=base_folder, transform=transform_train, supervised=opt.train_supervised)
        val_dataset = ImagePatchesDataset(val_df, image_dir=base_folder, transform=transform_valid, supervised=opt.train_supervised)
        test_dataset = ImagePatchesDataset(test_df, image_dir=base_folder, transform=transform_valid, supervised=opt.train_supervised)
    elif opt.dataset == 'patchcam':
        train_dataset = H5Dataset(train_df, image_dir=base_folder, transform=transform_train)
        val_dataset = H5Dataset(val_df, image_dir=base_folder, transform=transform_valid)
        test_dataset = H5Dataset(test_df, image_dir=base_folder, transform=transform_valid)

    # Weighted sampler to handle class imbalance
    print('Weighted validation sampler')
    val_sampler = get_weighted_sampler(val_dataset, num_samples=len(val_dataset))
    if opt.train_supervised:
        print('Weighted training sampler')
        train_sampler = get_weighted_sampler(train_dataset, num_samples=len(train_dataset))

    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=True,
        num_workers=opt.num_workers,
        drop_last=True,
    )

    if opt.train_supervised:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size_multiGPU,
            sampler=train_sampler,
            num_workers=opt.num_workers,
            drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size_multiGPU//2,
        sampler=val_sampler,
        num_workers=opt.num_workers,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size_multiGPU//2,
        shuffle=False,
        num_workers=opt.num_workers,
        drop_last=True,
    )

    return (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    )

def get_multidata_dataloader(opt):
    '''
    Loads pathologydataset from lmdb files, performing augmentaions.
    Only supporting pretraining, as no labels are available
    args:
        opt (dict): Program/commandline arguments.
    Returns:
        dataloaders (): pretraindataloaders.
    '''

    # Base train and test augmentaions
    aug = {
        "multidata": {
            "resize": None,
            "randcrop": None,
            "flip": True,
            "color_jitter": 0.8,
            "color_value": 0.2,
            "grayscale": opt.grayscale,
            "gaussian_blur": True,
            "rotation": True,
            "mean": [0.4914, 0.4822, 0.4465],  # values for train+unsupervised combined
            "std": [0.2023, 0.1994, 0.2010],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        },
    }

    transform_train = transforms.Compose(get_transforms(eval=False, aug=aug['multidata']))

    train_dataset = LmdbDataset(lmdb_path=opt.data_input_dir,
                        transform=transform_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=opt.num_workers,
                                        pin_memory=True, drop_last=True,
                                        shuffle=True,
                                        batch_size=opt.batch_size_multiGPU)

    return (
        train_loader,
        train_dataset,
        None,
        None,
        None,
        None,
    )


class ImagePatchesDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, supervised=False):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.supervised = supervised

        self.label_enum = {'TUMOR': 1, 'NONTUMOR': 0}

        self.targets = list(dataframe.label.apply(lambda x: self.label_enum[x]))

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        path = f"{self.image_dir}/{row.filename}"
        try:
            image = Image.open(path)
        except IOError:
            print(f"could not open {path}")
            return None

        if self.transform is not None:
            pos_1 = self.transform(image)
            if self.supervised:
                pos_2 = None
            else:
                pos_2 = self.transform(image)
        else:
            raise NotImplementedError

        label = self.label_enum[row.label]

        if self.supervised:
            return pos_1, label, row.patch_id, row.slide_id
        return pos_1, pos_2, label, row.patch_id, row.slide_id


class LmdbDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, transform):
        self.cursor_access = False
        self.lmdb_path = lmdb_path
        self.image_dimensions = (224, 224, 3) # size of files in lmdb
        self.transform = transform

        self._init_db()

    def __len__(self):
        return self.length

    def _init_db(self):
        num_readers = 999

        self.env = lmdb.open(self.lmdb_path,
                             max_readers=num_readers,
                             readonly=1,
                             lock=0,
                             readahead=0,
                             meminit=0)

        self.txn = self.env.begin(write=False)
        self.cursor = self.txn.cursor()

        self.length = self.txn.stat()['entries']
        print('Generating keys to lmdb dataset, this takes a while...')
        self.keys = [key for key, _ in self.txn.cursor()] # not so fast...

    def close(self):
        self.env.close()

    def __getitem__(self, index):
        ' cursor in lmdb is much faster than random access '
        if self.cursor_access:
            if not self.cursor.next():
                self.cursor.first()
            image = self.cursor.value()
        else:
            image = self.txn.get(self.keys[index])

        image = np.frombuffer(image, dtype=np.uint8)
        image = image.reshape(self.image_dimensions)
        image = Image.fromarray(image)

        pos_1 = self.transform(image)
        pos_2 = self.transform(image)

        return pos_1, pos_2

class H5Dataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

        self.label_enum = {'TUMOR': 1, 'NONTUMOR': 0}

        with h5py.File(f'{image_dir}', 'r') as f:
            self.data = f['x'][:]


    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        try:
            image = Image.fromarray(self.data[index])
        except IOError:
            print(f"could not open image at index {index}")
            return None

        if self.transform is not None:
            pos_1 = self.transform(image)
            pos_2 = self.transform(image)
        else:
            raise NotImplementedError

        label = self.label_enum[row.label]

        return pos_1, pos_2, label, row.patch_id, row.slide_id

