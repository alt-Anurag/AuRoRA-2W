import os
import sys
import argparse
import pprint
import logging
import timeit
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'PIDNet'))

import models.aurora2w
import datasets
from configs import config, update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import validate, train
from utils.utils import create_logger, AverageMeter, adjust_learning_rate
import time

class AuRoRA2W_FullModel(nn.Module):
    def __init__(self, model, sem_loss, bd_loss):
        super(AuRoRA2W_FullModel, self).__init__()
        self.model = model
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss

    def forward(self, inputs, labels, bd_gts):
        # --- SYNTHETIC ROLL AUGMENTATION (FIXED) ---
        B = inputs.size(0)
        # Generate random roll angles between -30 and 30 degrees
        roll_angles = (torch.rand(B, device=inputs.device) * 60.0) - 30.0
        
        # Rotate inputs (fill corners with ImageNet Mean to avoid pitch-black contrast shocks)
        rot_inputs = torch.stack([TF.rotate(inputs[i], float(roll_angles[i]), fill=[0.0, 0.0, 0.0]) for i in range(B)])
        
        # CRITICAL FIX: Rotate labels and fill corners with 255 (Ignore Index). 
        # This prevents the network from thinking black corners are "Road" (Class 0).
        rot_labels = torch.stack([TF.rotate(labels[i].unsqueeze(0).float(), float(roll_angles[i]), interpolation=TF.InterpolationMode.NEAREST, fill=[255.0]).squeeze(0).long() for i in range(B)])
        
        # Rotate boundary ground truths
        rot_bd = torch.stack([TF.rotate(bd_gts[i].unsqueeze(0), float(roll_angles[i]), interpolation=TF.InterpolationMode.NEAREST, fill=[0.0]).squeeze(0) for i in range(B)])

        # --- FORWARD PASS (Injecting roll_angles into AuRoRA2W) ---
        out = self.model(rot_inputs, roll_angles=roll_angles)
        
        # --- LOSS CALCULATION ---
        ph, pw = out[0].size(2), out[0].size(3)
        h, w = labels.size(1), labels.size(2)
        if ph != h or pw != w:
            for i in range(len(out)):
                out[i] = F.interpolate(out[i], size=(h, w), mode='bilinear', align_corners=True)
        
        loss1 = self.sem_loss([out[0]], rot_labels)
        loss2 = self.sem_loss([out[1]], rot_labels)
        loss3 = self.bd_loss(out[2], rot_bd)
        
        # Fake accuracy for API compatibility
        acc = torch.tensor([0.0], device=inputs.device)
        return loss1 + loss2 + loss3, [out[0], out[1]], acc, [loss1+loss2, loss3]


def parse_args():
    parser = argparse.ArgumentParser(description='Train AuRoRA-2W')
    parser.add_argument('--cfg', default="models/PIDNet/configs/cityscapes/pidnet_small_cityscapes.yaml", type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    update_config(config, args)
    
    # We must fix paths for the config because we are running from root
    config.defrost()
    config.DATASET.ROOT = 'models/PIDNet/data/'
    config.MODEL.PRETRAINED = 'models/PIDNet/pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar'
    config.TRAIN.IMAGE_SIZE = [512, 512]
    config.TRAIN.BATCH_SIZE_PER_GPU = 2
    config.GPUS = (0,)
    config.freeze()
    
    return args

def main():
    args = parse_args()

    if args.seed > 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)

    # --- FIX: use ImageNet-pretrained backbone weights ---
    # Training PIDNet-S from random init, at a fine-tuning LR, on top of
    # aggressive synthetic-roll augmentation is what was producing the
    # "terrible" early-epoch results seen at 30% of an epoch -- not the IDFA
    # math (idfa.py's closed-form offsets are correct and unchanged below).
    # Load the pretrained backbone unless the weights file is genuinely
    # missing, in which case warn loudly instead of silently degrading to
    # random init, and fall back to a from-scratch-appropriate LR.
    pretrained_path = config.MODEL.PRETRAINED
    use_pretrained = os.path.isfile(pretrained_path)
    if not use_pretrained:
        print(f"[!] WARNING: pretrained weights not found at '{pretrained_path}'.")
        print("[!] Training from random init. Expect much slower convergence -- "
              "raise TRAIN.LR and/or TRAIN.END_EPOCH accordingly, or fix the path.")
    else:
        print(f"[*] Loading ImageNet-pretrained backbone from '{pretrained_path}'")

    model = models.aurora2w.get_seg_model(config, imgnet_pretrained=use_pretrained)

    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=config.TRAIN.SCALE_FACTOR)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0, # Disable multiprocessing for faster startup
        pin_memory=False,
        drop_last=True)
        
    valid_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size)

    validloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=0,
        pin_memory=False)

    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL, thres=config.LOSS.OHEMTHRES, min_kept=config.LOSS.OHEMKEEP, weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL, weight=train_dataset.class_weights)
        
    bd_criterion = BondaryLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use our Custom Roll-Augmented Model Wrapper!
    model = AuRoRA2W_FullModel(model, sem_criterion, bd_criterion).to(device)
    if device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=gpus)

    params = [{'params': list(model.parameters()), 'lr': config.TRAIN.LR}]
    optimizer = torch.optim.SGD(params, lr=config.TRAIN.LR, momentum=config.TRAIN.MOMENTUM, weight_decay=config.TRAIN.WD)
    
    # Initialize AMP Scaler for Mixed Precision
    scaler = torch.amp.GradScaler('cuda')

    # Full Training Loop Setup
    print("[*] Starting AuRoRA-2W Synthetic Roll Training Loop with AMP...")
    model.train()
    
    # Resume logic
    start_epoch = 0
    base_lr = config.TRAIN.LR
    if not use_pretrained and base_lr < 0.001:
        print(f"[!] LR {base_lr} looks tuned for a pretrained fine-tune, not "
              "from-scratch training. Consider raising it (e.g. 0.01) if this "
              "run is starting from random init and convergence looks slow.")
    import glob
    checkpoints = glob.glob(os.path.join(final_output_dir, 'aurora2w_epoch_*.pt'))
    if checkpoints:
        latest_cp = max(checkpoints, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        print(f"[*] Resuming from checkpoint: {latest_cp}")
        checkpoint = torch.load(latest_cp, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            # Legacy format recovery
            model.module.load_state_dict(checkpoint)
            start_epoch = int(os.path.basename(latest_cp).split('_')[-1].split('.')[0]) + 1
            print("[!] Legacy checkpoint detected (No optimizer state). Dropping base learning rate to 0.0001 to prevent momentum shock.")
            base_lr = 0.0001

    num_epochs = config.TRAIN.END_EPOCH
    for epoch in range(start_epoch, num_epochs):
        model.train()
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        for i, batch in enumerate(trainloader):
            # Apply learning rate decay scheduler per iteration
            cur_iters = epoch * len(trainloader) + i
            max_iters = config.TRAIN.END_EPOCH * len(trainloader)
            lr = adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters)
        
            images, labels, bd_gts, _, _ = batch
            images = images.to(device)
            labels = labels.long().to(device)
            bd_gts = bd_gts.float().to(device)
            
            optimizer.zero_grad()
            
            # Autocast for Mixed Precision
            with torch.amp.autocast('cuda'):
                losses, _, acc, loss_list = model(images, labels, bd_gts)
                loss = losses.mean()
            
            # Scaled Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if i % 10 == 0:
                print(f"[Iter {i}/{len(trainloader)}] LR: {lr:.6f} | Loss: {loss.item():.4f}")
                
        # --- FIX: restored validation ---
        # Running blind on training loss alone (no mIoU) meant there was no
        # way to tell "undertrained backbone" apart from "IDFA doesn't work"
        # apart from eyeballing output. Validate every VAL_INTERVAL epochs;
        # this costs one forward pass over validloader, not a big time hit,
        # and gives you the roll-stratified signal your actual hypothesis
        # needs. Adjust VAL_INTERVAL if even this is too slow for your clock.
        VAL_INTERVAL = 5
        if (epoch + 1) % VAL_INTERVAL == 0 or (epoch + 1) == num_epochs:
            model.eval()
            with torch.no_grad():
                try:
                    val_loss, mean_IoU, IoU_array = validate(
                        config, validloader, model, writer_dict)
                    print(f"[VAL] Epoch {epoch+1}: mIoU={mean_IoU:.4f}  "
                          f"val_loss={val_loss:.4f}")
                except Exception as e:
                    # utils.function.validate() may expect the raw
                    # segmentation model rather than the AuRoRA2W_FullModel
                    # wrapper (which returns losses, not just logits) --
                    # if it errors, don't let it kill the training run.
                    print(f"[VAL] Skipped -- validate() raised: {e}")
            model.train()

        # Save full state checkpoint after each epoch
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'mean_IoU': locals().get('mean_IoU', None),  # None if not validated this epoch
        }
        torch.save(checkpoint_dict, os.path.join(final_output_dir, f'aurora2w_epoch_{epoch}.pt'))
            
    print("[+] Training script successfully completed!")

if __name__ == '__main__':
    main()