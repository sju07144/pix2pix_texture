import argparse
import datetime
import os
import sys
import time
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.utils import save_image

import matplotlib.pyplot as plt
import numpy as np

from pix2pix_dataset import ABODataset
from pix2pix_model import Generator, Discriminator

root_dir = "../../hoyeon/Capture_231113" # current directory is "/home/sju07144/pix2pix"
pix2pix_dir = "/home/sju07144/pix2pix"
root_512_dir = "../../hoyeon/Capture_512"
# root_dir = "..\\resources\\IBL_rendered_examples"
modes = [
  'Albedo',
  'AO',
  'Metallic',
  'Metallic-Roughness',
  'NormalMap',
  'Normal',
  'Roughness'
]

generator_losses = []
discriminator_losses = []
pixel_losses = []
GAN_losses = []

def get_args_parser():
    parser = argparse.ArgumentParser(description='Pytorch pix2pix Training')
    parser.add_argument("--generator_mode", type=int, default=0, help="Check which data for training")
    parser.add_argument("--start_epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--img_height", type=int, default=512, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    parser.add_argument("--channels", type=int, default=4, help="number of image channels")
    parser.add_argument(
        "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
    parser.add_argument('--rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--num_workers', type=int, default=16, help="number of cpu threads to use during batch generation")
    # parser.add_argument('--gpu_ids', nargs="+", default=['0'])
    # parser.add_argument('--gpu_ids', nargs="+", default=['0', '1'])
    # parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2'])
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2', '3'], help='ids of gpu')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes for distributed training')
    parser.add_argument('--port', type=int, default=23456, help='selection of port')
    parser.add_argument('--root', type=str, default='/home/sju07144/Capture_231113', help='dataset root directory')
    # parser.add_argument('--save_path', type=str, default='./save')
    # parser.add_argument('--save_file_name', type=str, default='vgg_cifar')
    parser.add_argument('--local_rank', type=int, help='choose local rank')
    parser.add_argument('--dist_url', default = 'tcp://127.0.0.1:23456', 
                        type=str, help='url used to set up distributed training')
    # usage : --gpu_ids 0, 1, 2, 3
    return parser

def init_for_distributed(rank, opts):
    # 1. setting for distributed training
    opts.rank = rank
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    torch.distributed.init_process_group(backend='nccl', init_method=opts.dist_url,
                                         world_size=opts.world_size, rank=opts.rank)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    # setup_for_distributed(opts.rank == 0)
    print(opts)
    return local_gpu_id


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    # import builtins as __builtin__
    # builtin_print = __builtin__.print
# 
    # def print(*args, **kwargs):
    #     force = kwargs.pop('force', False)
    #     if is_master or force:
    #         builtin_print(*args, **kwargs)
# 
    # __builtin__.print = print

def main_worker(rank, opts):
    # Tensor type
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    # init dist
    local_gpu_id = init_for_distributed(rank, opts)
    device = torch.device('cuda:{}'.format(local_gpu_id))
    
    # os.makedirs(pix2pix_dir + "/images/%s" % modes[opts.generator_mode], exist_ok=True)
    # os.makedirs(pix2pix_dir + "/saved_models/%s" % modes[opts.generator_mode], exist_ok=True)
    
    train_dataset = ABODataset(root=opts.root, mode=opts.generator_mode)
    val_dataset = ABODataset(root=opts.root, train=False, mode=opts.generator_mode)
    
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    val_sampler = DistributedSampler(dataset=val_dataset, shuffle=True)
    
    train_dataLoader = DataLoader(dataset=train_dataset,
                                  batch_size=opts.batch_size // opts.world_size,
                                  shuffle=False, # Choose False if you do distributed processing!!
                                  num_workers=opts.num_workers // opts.world_size,
                                  sampler=train_sampler,
                                  pin_memory=True)

    # sampler_train = DistributedSampler(train_dataset, shuffle=False)
    # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, opts.batch_size, drop_last=True)
    # train_dataLoader = DataLoader(dataset=train_dataset, 
    #                               batch_sampler=batch_sampler_train, 
    #                               num_workers=opts.num_workers)
    val_dataLoader = DataLoader(dataset=val_dataset,
                                batch_size=opts.batch_size // opts.world_size,
                                shuffle=False,
                                num_workers=opts.num_workers // opts.world_size,
                                sampler=val_sampler,
                                pin_memory=True)
    
    # model
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    generator = DDP(module=generator, device_ids=[local_gpu_id], broadcast_buffers=False)
    discriminator = DDP(module=discriminator, device_ids=[local_gpu_id], broadcast_buffers=False)
    
    if opts.start_epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load(pix2pix_dir + "/saved_models/%s/generator_%d.pth" 
                                             % (modes[opts.generator_mode], opts.start_epoch), map_location=device))
        discriminator.load_state_dict(torch.load(pix2pix_dir + "/saved_models/%s/discriminator_%d.pth" 
                                                 % (modes[opts.generator_mode], opts.start_epoch), map_location=device))
        if opts.rank == 0:
            print('\nLoaded checkpoint from epoch %d\n' % (int(opts.start_epoch) - 1))   
    
    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss().to(device) # only vanila and lsgan -> nn.MSELoss()
    criterion_pixelwise = nn.L1Loss().to(device)
    
    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100
    
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))
    
    # Calculate output of image discriminator (PatchGAN)
    # patch = (1, opts.img_height // 2 ** 4, opts.img_width // 2 ** 4)
        
    prev_time = time.time()
    
    

    for epoch in range(opts.start_epoch, opts.n_epochs):
        generator_losses_per_epoch = []
        discriminator_losses_per_epoch = []
        pixel_losses_per_epoch = []
        GAN_losses_per_epoch = []
        
        for i, batch in enumerate(train_dataLoader):
            # Model inputs
            input_image, real_image = batch
            input_image = Variable(input_image.type(Tensor)).to(device)
            real_image = Variable(real_image.type(Tensor)).to(device)
            
            # ------------------
            #  Train Generators
            # ------------------

            generator_optimizer.zero_grad()

            # GAN loss
            generated_image = generator(input_image)
            pred_fake = discriminator(generated_image, input_image)

            # Adversarial ground truths
            pred_fake_numpy = pred_fake.detach().cpu().numpy()
            valid = Variable(Tensor(np.ones_like(pred_fake_numpy)), requires_grad=False)
            fake = Variable(Tensor(np.zeros_like(pred_fake_numpy)), requires_grad=False)

            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(generated_image, real_image)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()
            generator_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            discriminator_optimizer.zero_grad()

            # Real loss
            pred_real = discriminator(real_image, input_image)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(generated_image.detach(), input_image)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            discriminator_optimizer.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(train_dataLoader) + i
            batches_left = opts.n_epochs * len(train_dataLoader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            if i % 1000 == 0 and opts.rank == 0:
                print(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                    % (
                        epoch + 1,
                        opts.n_epochs,
                        i,
                        len(train_dataLoader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_pixel.item(),
                        loss_GAN.item(),
                        time_left,
                    )
                )
                
                generator_losses_per_epoch.append(loss_G.item())
                discriminator_losses_per_epoch.append(loss_D.item())
                pixel_losses_per_epoch.append(loss_pixel.item())
                GAN_losses_per_epoch.append(loss_GAN.item())
                
            # If at sample interval save image
            if i != 0 and i % opts.sample_interval == 0 and opts.rank == 0:
                sample_images(opts, val_dataLoader, generator, epoch + 1, i, len(train_dataLoader))

        if opts.rank == 0 and opts.checkpoint_interval != -1 and epoch != 0 and (epoch + 1) % opts.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), pix2pix_dir + "/saved_models/%s/generator_%d.pth" 
                       % (modes[opts.generator_mode], epoch + 1))
            print("Save generator pth.tar {} epoch!".format(epoch + 1))
            torch.save(discriminator.state_dict(), pix2pix_dir + "/saved_models/%s/discriminator_%d.pth" 
                       % (modes[opts.generator_mode], epoch + 1))
            print("Save discriminator pth.tar {} epoch!".format(epoch + 1))
            
        generator_losses.append(np.mean(generator_losses_per_epoch))
        discriminator_losses.append(np.mean(discriminator_losses_per_epoch))
        pixel_losses.append(np.mean(pixel_losses_per_epoch))
        GAN_losses.append(np.mean(GAN_losses_per_epoch))

# Tensor type
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def sample_images(opts, val_dataLoader, generator, epoch, current_batch, total_batch_count):
    """Saves a generated sample from the validation set"""
    input_image, real_image = next(iter(val_dataLoader))
    input_image = Variable(input_image.type(Tensor))
    real_image = Variable(real_image.type(Tensor))
    generated_image = generator(input_image)
    img_sample = torch.cat((input_image[0] * 0.5 + 0.5, generated_image[0] * 0.5 + 0.5, real_image[0] * 0.5 + 0.5), -2)
    img_sample = img_sample.unsqueeze(0)
    image_dir = pix2pix_dir + "/images/{0}/{1}".format(modes[opts.generator_mode], epoch)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    save_image(img_sample, image_dir + "/%s_%s.png" % (current_batch, total_batch_count), nrow=5, normalize=False)
    print('Complete to save {0}_{1}.png sample images.'.format(current_batch, total_batch_count))
    
def save_losses(opts):
    epochs = np.arange(1, opts.n_epochs + 1)
    
    plt.plot(epochs, np.array(generator_losses), 'r', label='loss_G')
    plt.plot(epochs, np.array(discriminator_losses), 'b', label='loss_D')
    plt.plot(epochs, np.array(pixel_losses), 'g', label='loss_pixel')
    plt.plot(epochs, np.array(GAN_losses), color='violet', label='loss_GAN')
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    plt.legend(loc='upper right')
    
    file_name = './statistics/losses.png'
    if os.path.isfile(file_name):
        os.remove(file_name)
    
    plt.savefig(file_name, dpi=300, facecolor='white', edgecolor='black', 
                orientation='portrait', format='png', transparent=False,
                bbox_inches='tight', pad_inches=0.1)

if __name__ == '__main__':
    parser = get_args_parser()
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4
    
    mp.spawn(main_worker,
             args=(opts, ),
             nprocs=opts.world_size,
             join=True)
    
    save_losses(opts)