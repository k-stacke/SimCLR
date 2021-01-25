import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda import amp

from torchvision import transforms
#from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

# from torchlars import LARS

from simclr.utils import distribute_over_GPUs
from simclr.get_dataloader import get_dataloader
from simclr.model import Model
from simclr.optimisers import LARS

import neptune

torch.backends.cudnn.benchmark=True

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, scaler, opt, exp):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target, _, _ in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

        with amp.autocast():
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)
            # [2*B, D]
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * opt.batch_size_multiGPU, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * opt.batch_size_multiGPU, -1)
    
            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        train_optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(train_optimizer)
        scaler.update()
        #train_optimizer.step()

        total_num += opt.batch_size_multiGPU
        total_loss += loss.item() * opt.batch_size_multiGPU
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        exp.log_metric('loss', total_loss / total_num)

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, val_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target, _, _ in tqdm(val_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(val_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target, _, _ in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description(f'Test Epoch: [{epoch}/{epochs}] Acc@1:{total_top1 / total_num * 100:.2f}%')

    return total_top1 / total_num * 100


if __name__ == '__main__':
    neptune.init('k-stacke/self-supervised')

    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')

    parser.add_argument('--training_data_csv', required=True, type=str, help='Path to file to use to read training data')
    parser.add_argument('--test_data_csv', required=True, type=str, help='Path to file to use to read test data')
    # For validation set, need to specify either csv or train/val split ratio
    group_validationset = parser.add_mutually_exclusive_group(required=True)
    group_validationset.add_argument('--validation_data_csv', type=str, help='Path to file to use to read validation data')
    group_validationset.add_argument('--trainingset_split', type=float, help='If not none, training csv with be split in train/val. Value between 0-1')

    parser.add_argument('--dataset', choices=['cam', 'patchcam'], default='cam', type=str, help='Dataset')
    parser.add_argument('--data_input_dir', type=str, help='Base folder for images (or h5 file)')
    parser.add_argument('--save_dir', type=str, help='Path to save log')
    parser.add_argument('--save_after', type=int, default=1, help='Save model after every Nth epoch, default every epoch')
    parser.add_argument("--validate", action="store_true", default=False,help="Boolean to decide whether to split train dataset into train/val and plot validation loss",)
    parser.add_argument("--balanced_validation_set", action="store_true", default=False, help="Equal size of classes in validation AND test set",)

    parser.add_argument("--optimizer", choices=['adam', 'lars'], default='adam', help="Optimizer to use",)


    # args parse
    opt = parser.parse_args()
    feature_dim, temperature, k = opt.feature_dim, opt.temperature, opt.k
    batch_size, epochs = opt.batch_size, opt.epochs


    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', opt.device)
    is_windows = True if os.name == 'nt' else False
    opt.num_workers = 0 if is_windows else 16

    opt.train_supervised = False
    opt.grayscale = False

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)
    opt.log_path = opt.save_dir

    exp = neptune.create_experiment(name='SimCLR', params=opt.__dict__, tags=['simclr'])

    # Write the parameters used to run experiment to file
    with open(f'{opt.log_path}/metadata_train.txt', 'w') as metadata_file:
        metadata_file.write(f'{opt}')

    # model setup and optimizer config
    scaler = amp.GradScaler()

    model = Model(feature_dim)
    model, num_GPU = distribute_over_GPUs(opt, model)

    if opt.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6)
    elif opt.optimizer == 'lars':
        params_models = []
        reduced_params = []
        removed_params = []

        skip_lists = ['bn', 'bias']

        m_skip = []
        m_noskip = []
        params_models += list(model.parameters())

        for name, param in model.named_parameters():
            if (any(skip_name in name for skip_name in skip_lists)):
                m_skip.append(param)
            else:
                m_noskip.append(param)
        reduced_params += list(m_noskip)
        removed_params += list(m_skip)
        print("reduced_params len: {}".format(len(reduced_params)))
        print("removed_params len: {}".format(len(removed_params)))
        optimizer = LARS(reduced_params+removed_params, lr=opt.lr,
                         weight_decay=1e-6, eta=0.001, use_nesterov=False, len_reduced=len(reduced_params))
        # base_optimizer = optim.SGD(model.parameters(), lr=0.1)
        # optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
    # c = 2

    train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset = get_dataloader(opt)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    # training loop
    results = {'train_loss': []}#, 'test_acc': []}
    save_name_pre = f'{feature_dim}_{temperature}_{k}_{batch_size}_{epochs}'
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        exp.log_metric('learning_rate', scheduler.get_last_lr()[0])

        train_loss = train(model, train_loader, optimizer, scaler, opt, exp)
        scheduler.step()
        results['train_loss'].append(train_loss)
        exp.log_metric('epoch_loss', train_loss)

        #test_acc= test(model, val_loader, test_loader)
        #results['test_acc'].append(test_acc)
        #exp.log_metric('epoch_accuracy', test_acc)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(f'{opt.save_dir}/{save_name_pre}_statistics.csv', index_label='epoch')

        ## Save model
        if epoch % opt.save_after == 0:
            torch.save(model.module.state_dict(), f'{opt.log_path}/{save_name_pre}_model_{epoch}.pth')
        # Delete old ones, save latest, keep every 10th
        if (epoch - 1) % 10 != 0:
            try:
                os.remove(f'{opt.log_path}/{save_name_pre}_model_{epoch - 1}.pth')
            except:
                print("not enough models there yet, nothing to delete")
