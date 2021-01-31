import argparse
import os
import random 
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from simclr.utils import distribute_over_GPUs 
from model import Model
from get_dataloader import get_dataloader

import neptune

torch.backends.cudnn.benchmark=True

class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()

        # Load pre-trained model
        base_model = Model()
        base_model.load_state_dict(torch.load(opt.model_path, map_location=opt.device.type), strict=False)
        # print(base_model)
        # encoder
        self.f = base_model.f
        # classifier
        self.fc = nn.Linear(2048, opt.num_classes, bias=True)
        # self.load_state_dict(torch.load(pretrained_path, map_location='cuda'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val(net, data_loader, train_optimizer, exp):
    is_train = train_optimizer is not None
    net.eval() # train only the last layers. 
    #net.train() if is_train else net.eval()

    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)

    all_preds, all_labels, all_slides, all_outputs0, all_outputs1, all_patches  = [], [], [], [], [], []

    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target, patch_id, slide_id in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            _, preds = torch.max(out.data, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().data.numpy())
            all_patches.extend(patch_id)
            all_slides.extend(slide_id)

            probs = torch.nn.functional.softmax(out.data, dim=1).cpu().numpy()
            all_outputs0.extend(probs[:, 0])
            all_outputs1.extend(probs[:, 1])

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description(f'{"Train" if is_train else "Test"} Epoch: [{epoch}/{epochs}] Loss: {total_loss / total_num:.4f} ACC: {total_correct / total_num * 100:.2f}% ')


    df =  pd.DataFrame({
                'label': all_labels,
                'prediction': all_preds,
                'slide_id': all_slides,
                'patch_id': all_patches,
                'probabilities_0': all_outputs0,
                'probabilities_1': all_outputs1,
            })

    for label in df.label.unique():
        label_acc = len(df[(df.label == label) & (df.prediction == df.label)])/len(df[df.label == label])
        exp.send_metric(f'acc_{label}', label_acc)

    return total_loss / total_num, total_correct / total_num * 100, df


if __name__ == '__main__':
    neptune.init('k-stacke/self-supervised')

    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')

    parser.add_argument('--training_data_csv', required=True, type=str, help='Path to file to use to read training data')
    parser.add_argument('--test_data_csv', required=True, type=str, help='Path to file to use to read test data')
    # For validation set, need to specify either csv or train/val split ratio
    group_validationset = parser.add_mutually_exclusive_group(required=True)
    group_validationset.add_argument('--validation_data_csv', type=str, help='Path to file to use to read validation data')
    group_validationset.add_argument('--trainingset_split', type=float, help='If not none, training csv with be split in train/val. Value between 0-1')
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")

    parser.add_argument('--dataset', choices=['cam', 'patchcam'], default='cam', type=str, help='Dataset')
    parser.add_argument('--data_input_dir', type=str, help='Base folder for images (or h5 file)')
    parser.add_argument('--save_dir', type=str, help='Path to save log')
    parser.add_argument('--save_after', type=int, default=1, help='Save model after every Nth epoch, default every epoch')
    parser.add_argument("--validate", action="store_true", default=False,help="Boolean to decide whether to split train dataset into train/val and plot validation loss",)
    parser.add_argument("--balanced_validation_set", action="store_true", default=False, help="Equal size of classes in validation AND test set",)

    parser.add_argument("--finetune", action="store_true", default=False, help="If true, pre-trained model weights will not be frozen.")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument("--model_to_save", choices=['best', 'latest'], default='latest', type=str, help='Save latest or best (based on val acc)')
    parser.add_argument('--seed', type=int, default=44, help='seed')

    opt = parser.parse_args()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', opt.device)
    is_windows = True if os.name == 'nt' else False
    opt.num_workers = 0 if is_windows else 16

    opt.train_supervised = True
    opt.grayscale = False

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)
    opt.log_path = opt.save_dir

    exp = neptune.create_experiment(name='SimCLR linear classification', params=opt.__dict__,
                                    tags=['simclr', 'linear'])

    seed = opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_path, batch_size, epochs = opt.model_path, opt.batch_size, opt.epochs

    model = Net(opt)
    model, num_GPU = distribute_over_GPUs(opt, model)

    train_loader, train_data, val_loader, val_data, test_loader, test_data = get_dataloader(opt)

    if not opt.finetune:
        for param in model.module.f.parameters():
            param.requires_grad = False

    for name, param in model.named_parameters():
        print('Requires grad: ')
        if param.requires_grad:
            print(name)

    optimizer = optim.Adam(model.module.fc.parameters(), lr=opt.lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, opt.epochs)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=1e-6)

    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': []}

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        exp.log_metric('learning_rate', optimizer.param_groups[0]['lr'])

        train_loss, train_acc, _ = train_val(model, train_loader, optimizer, exp)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        exp.log_metric('train_acc', train_acc)
        val_loss, val_acc, _ = train_val(model, val_loader, None, exp)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        exp.log_metric('val_acc', val_acc)

        scheduler.step()

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(f'{opt.log_path}/linear_statistics.csv', index_label='epoch')

        if opt.model_to_save == 'best' and val_acc > best_acc:
            # Save only the if the accuracy exceeds previous accuracy
            best_acc = val_acc
            torch.save(model.state_dict(), f'{opt.log_path}/linear_model.pth')
        elif opt.model_to_save == 'latest':
            # Save latest model
            best_acc = val_acc
            torch.save(model.state_dict(), f'{opt.log_path}/linear_model.pth')

    # trainig finished, run test
    print('Training finished, testing started...')
    # Load saved model
    model.load_state_dict(torch.load(f'{opt.log_path}/linear_model.pth'))
    model.eval()
    test_loss, test_acc, df = train_val(model, test_loader, None, exp)

    df.to_csv(
       f"{opt.log_path}/inference_result_model.csv")





