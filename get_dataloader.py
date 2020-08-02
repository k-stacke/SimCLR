import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

import neptune


def get_dataloader(opt):
    if opt.dataset == 'cam17':
        train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset = get_camelyon_dataloader(
            opt
        )
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

    if eval:
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
        )
        return test_transform

    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomResizedCrop(32, (0.8,1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    return train_transform

    # trans = []

    # if aug["randcrop"] and not eval:
    #     trans.append(transforms.RandomCrop(aug["randcrop"]))

    # if aug["randcrop"] and eval:
    #     trans.append(transforms.CenterCrop(aug["randcrop"]))

    # if aug["flip"] and not eval:
    #     trans.append(transforms.RandomHorizontalFlip())

    # if aug["hue"] and not eval:
    #     trans.append(transforms.ColorJitter(hue=0.1))

    # if aug["grayscale"]:
    #     trans.append(transforms.Grayscale())
    #     trans.append(transforms.ToTensor())
    #     trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    # elif aug["mean"]:
    #     trans.append(transforms.ToTensor())
    #     trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    # else:
    #     trans.append(transforms.ToTensor())

    # trans = transforms.Compose(trans)
    # return trans

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


def get_camelyon_dataloader(opt):
    base_folder = opt.data_input_dir
    print('opt.data_input_dir: ', opt.data_input_dir)

    # randcrop = 128 if opt.big_patches else 64

    # aug = {
    #     "cam17": {
    #         "randcrop": randcrop,
    #         "flip": True,
    #         "hue": True,
    #         "grayscale": opt.grayscale,
    #         "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined #TODO: find new mean
    #         "std": [0.2683, 0.2610, 0.2687],
    #         "bw_mean": [0.4120],  # values for train+unsupervised combined
    #         "bw_std": [0.2570],
    #     }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    # }
    # transform_train = transforms.Compose(
    #     [get_transforms(eval=False, aug=aug["cam17"])]
    # )
    # transform_valid = transforms.Compose(
    #     [get_transforms(eval=True, aug=aug["cam17"])]
    # )
    transform_train = get_transforms(eval=False)
    transform_valid = get_transforms(eval=True)


    if opt.training_data_csv:
        if os.path.isfile(opt.training_data_csv):
            print("reading csv file: ", opt.training_data_csv)
            train_df = pd.read_csv(opt.training_data_csv)
        else:
            raise Exception(f'Cannot find file: {opt.training_data_csv}')

        if os.path.isfile(opt.validation_data_csv):
            print("reading csv file: ", opt.validation_data_csv)
            val_df = pd.read_csv(opt.validation_data_csv)
        else:
            raise Exception(f'Cannot find file: {opt.test_data_csv}')

        if os.path.isfile(opt.test_data_csv):
            print("reading csv file: ", opt.test_data_csv)
            test_df = pd.read_csv(opt.test_data_csv)
        else:
            raise Exception(f'Cannot find file: {opt.test_data_csv}')

        train_df = clean_data(opt.data_input_dir, train_df)
        val_df = clean_data(opt.data_input_dir, val_df)
        test_df = clean_data(opt.data_input_dir, test_df)
    else:
        file_ = f"{opt.data_input_dir}/camelyon17_patches_unbiased.csv"
        #file_ = f"{opt.data_input_dir}/lnco_camelyon_patches.csv"
        if os.path.isfile(file_):
            print(f"reading {file_} file")
            df = pd.read_csv(file_)
        else:
            raise Exception(f"Cannot find file {file_}")

        df = clean_data(opt.data_input_dir, df)

        # df = df.sample(2000)
        slide_ids = df.slide_id.unique()
        random.shuffle(slide_ids)
        train_req_ids = []
        valid_req_ids = []
        # Take same number of slides from each site
        training_size = int(len(slide_ids)*0.8) # 80% training data
        validation_size = len(slide_ids) - training_size
        train_req_ids.extend([slide_id for slide_id in slide_ids[:training_size]])  # take first
        valid_req_ids.extend([
            slide_id for slide_id in slide_ids[training_size:training_size+validation_size]])  # take last

        print("train / valid / total")
        print(f"{len(train_req_ids)} / {len(valid_req_ids)} / {len(slide_ids)}")

        train_df = df[df.slide_id.isin(train_req_ids)]
        val_df = df[df.slide_id.isin(valid_req_ids)]

        print("Saving training/test set to file")
        train_df.to_csv(f'{opt.log_path}/training_patches.csv', index=False)
        val_df.to_csv(f'{opt.log_path}/test_patches.csv', index=False)

    if opt.balanced_validation_set:
        print('Use uniform validation set')
        samples_to_take = val_df.groupby('label').size().min()
        val_df = pd.concat([val_df[val_df.label == label].sample(samples_to_take) for label in val_df.label.unique()])

        print('Use uniform test set')
        samples_to_take = test_df.groupby('label').size().min()
        test_df = pd.concat([test_df[test_df.label == label].sample(samples_to_take) for label in test_df.label.unique()])

    print("training patches: ", train_df.groupby('label').size())
    print("Validation patches: ", val_df.groupby('label').size())
    print("Test patches: ", test_df.groupby('label').size())

    train_dataset = ImagePatchesDataset(train_df, image_dir=base_folder, transform=transform_train)
    val_dataset = ImagePatchesDataset(val_df, image_dir=base_folder, transform=transform_valid)
    test_dataset = ImagePatchesDataset(test_df, image_dir=base_folder, transform=transform_valid)

    # Weighted sampler to handle class imbalance
    print('Weighted validation sampler')
    val_sampler = get_weighted_sampler(val_dataset, num_samples=len(val_dataset))
    #train_sampler = get_lnco_weighted_sampler(train_dataset, num_samples=len(train_dataset))

    # default dataset loaders

    num_workers = 16
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=opt.batch_size_multiGPU,
        shuffle=True,
        num_workers=num_workers, 
        drop_last=True,
    )


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size_multiGPU//2,
        sampler=val_sampler,
        num_workers=num_workers,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=opt.batch_size_multiGPU//2, 
        shuffle=False, 
        num_workers=num_workers, 
        drop_last=True,
    )

    # create train/val split
    if opt.validate and not opt.training_data_csv:
        print("Use train / val split")

        df = train_df

        slide_ids = df.slide_id.unique()
        random.shuffle(slide_ids)
        train_req_ids = []
        valid_req_ids = []
        # Take same number of slides from each site
        training_size = int(len(slide_ids)*0.9) # 90% of 80% training data
        validation_size = len(slide_ids) - training_size
        train_req_ids.extend([slide_id for slide_id in slide_ids[:training_size]])  # take first
        valid_req_ids.extend([
            slide_id for slide_id in slide_ids[training_size:training_size+validation_size]])  # take last

        print("train / valid / total")
        print(f"{len(train_req_ids)} / {len(valid_req_ids)} / {len(slide_ids)}")

        train_df = df[df.slide_id.isin(train_req_ids)]
        val_df = df[df.slide_id.isin(valid_req_ids)]

        print("training patches: ", train_df.groupby('label').size())
        print("validation patches: ", val_df.groupby('label').size())

        print("Saving training/test set to file")
        train_df.to_csv(f'{opt.log_path}/training_patches_exl_val.csv', index=False)
        val_df.to_csv(f'{opt.log_path}/validation_patches.csv', index=False)

        train_dataset = ImagePatchesDataset(train_df, image_dir=base_folder, transform=transform_train)
        test_dataset = ImagePatchesDataset(val_df, image_dir=base_folder, transform=transform_valid)

        train_loader = torch.utils.data.DataLoader(
              train_dataset,
              batch_size=opt.batch_size_multiGPU,
              shuffle=True,
              num_workers=num_workers,
              drop_last=True,
        )

        # overwrite test_dataset and _loader with validation set
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size_multiGPU,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
        )

    else:
        print("Use (train+val) / test split")

    return (
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        test_loader,
        test_dataset,
    )

def clean_data(img_dir, dataframe):
    """ Clean the data """
    available_images = {f'camelyon17_imagedata/{file_}' for file_ in os.listdir(f"{img_dir}/camelyon17_imagedata")}
    for idx, row in dataframe.iterrows():
        if row.filename not in available_images:
            print(f"Removing non-existing file from dataset: {img_dir}/{row.filename}")
            dataframe = dataframe.drop(idx)
    return dataframe


def clean_lnco_data(img_dir, dataframe):
    """ Clean the data """
    available_images = {f'colon_imagedata/roi_lgl_norm/{file_}' for file_ in os.listdir(f"{img_dir}/colon_imagedata/roi_lgl_norm")}
    available_images2 = {f'colon_imagedata/tumor/{file_}' for file_ in os.listdir(f"{img_dir}/colon_imagedata/tumor")}
    for idx, row in dataframe.iterrows():
        if row.filename not in available_images:
            if row.filename not in available_images2:
                print(f"Removing non-existing file from dataset: {img_dir}/{row.filename}")
                dataframe = dataframe.drop(idx)
    return dataframe


class ImagePatchesDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

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
            pos_2 = self.transform(image)
        else:
            raise NotImplementedError

        # label = row.label_int
        label = self.label_enum[row.label]
        #one_hot = np.eye(5, dtype = np.float64)[:, label]

        return pos_1, pos_2, label, row.patch_id, row.slide_id

