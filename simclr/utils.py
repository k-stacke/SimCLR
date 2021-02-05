import random
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

def reload_weights(args, model, optimizer):
    # Load the pretrained model
    checkpoint = torch.load(args.load_checkpoint_dir, map_location="cpu")

    ## reload weights for training of the linear classifier
    if 'model' in checkpoint.keys():
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    ## reload weights and optimizers for continuing training
    if args.start_epoch > 0:
        print("Continuing training from epoch ", args.start_epoch)

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except KeyError:
            raise KeyError('Sry, no optimizer saved. Set start_epoch=0 to start from pretrained weights')

    return model, optimizer


def distribute_over_GPUs(opt, model):
    ## distribute over GPUs
    if opt.device.type != "cpu":
        model = nn.DataParallel(model)
        num_GPU = torch.cuda.device_count()
        opt.batch_size_multiGPU = opt.batch_size * num_GPU
    else:
        model = nn.DataParallel(model)
        opt.batch_size_multiGPU = opt.batch_size

    model = model.to(opt.device)
    print("Let's use", num_GPU, "GPUs!")

    return model, num_GPU

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    """Borrowed from MoCo implementation"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x




class FixedRandomRotation:
    """Rotate by one of the given angles."""
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

