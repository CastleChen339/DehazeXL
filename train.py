import os
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import CloudRemovalDataset
from models.DehazeXL.decoders.decoder import DehazeXL


def freq_loss(res, clear, criterion):
    clear_fft = torch.fft.fft2(clear, dim=(-2, -1))
    clear_fft = torch.stack((clear_fft.real, clear_fft.imag), -1)
    pred_fft = torch.fft.fft2(res, dim=(-2, -1))
    pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
    loss = criterion(pred_fft, clear_fft)
    return loss


def cal_loss(res, clear_img, criterion):
    loss1 = criterion(res, clear_img)
    loss2 = freq_loss(res, clear_img, criterion)
    return loss1 + loss2


def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, device, scheduler, args):
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f'Checkpoint {args.resume} not found.')
        else:
            print(f'Resume from {args.resume}')
            model.load_state_dict(torch.load(args.resume, weights_only=True))
            evaluate(model, test_dataloader, criterion, device)
    else:
        print('Train from scratch...')
    print("Learning rate: ", args.lr)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    best_loss = float("inf")
    step = 0
    for epoch in range(args.epochs):
        model.train()

        losses = {'loss': []}
        for batch in tqdm(train_dataloader, desc='Training'):
            step += 1
            cloud_imgs = batch['cloud_img'].to(device)
            clear_imgs = batch['clear_img'].to(device)

            res = model(cloud_imgs)
            loss = cal_loss(res, clear_imgs, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            losses['loss'].append(loss.item())
            if step % int(args.save_cycle * len(train_dataloader)) == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, '{}.pth'.format(step)))
        print('Epoch:{}/{} | Loss:{:.4f}'.format(epoch + 1, args.epochs, np.mean(losses['loss'])))
        test_loss = evaluate(model, test_dataloader, criterion, device)
        if test_loss <= best_loss:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pth'))
            best_loss = test_loss
            print('Saving best model...')

    print('\nTrain Complete.\n')
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'last.pth'))


def evaluate(model, test_dataloader, criterion, device):
    model.eval()
    losses = {'loss': []}
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating'):
            cloud_imgs = batch['cloud_img'].to(device)
            clear_imgs = batch['clear_img'].to(device)
            res = model(cloud_imgs)
            loss = cal_loss(res, clear_imgs, criterion)
            losses['loss'].append(loss.item())
    print('Test | Loss:{:.4f}'.format(np.mean(losses['loss'])))
    return np.mean(losses['loss'])


def main(args):
    total_dataset = CloudRemovalDataset(os.path.join(args.data_dir), args.nor, crop_size=2048)

    generator = torch.Generator().manual_seed(22)
    train_set, test_set = random_split(total_dataset,
                                       [int(len(total_dataset) * 0.85),
                                        len(total_dataset) - int(len(total_dataset) * 0.85)],
                                      generator=generator)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size * 2,
                             shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f'Using device: {device}')

    model = DehazeXL().to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.AdamW(params=filter(lambda x: x.requires_grad, model.parameters()),
                                  lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                                           eta_min=1e-6)
    optimizer.zero_grad()
    train_model(model, train_loader, test_loader, optimizer, criterion, device, scheduler, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--data_dir', type=str, default=r'./datasets/8KDehaze',
                        help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default=r'./checkpoints/train',
                        help='Path to save checkpoints')
    parser.add_argument('--save_cycle', type=int, default=1,
                        help='Cycle of saving checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size of each data batch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Epochs')
    parser.add_argument('--nor', action='store_true',
                        help='Normalize the image')
    main(parser.parse_args())

