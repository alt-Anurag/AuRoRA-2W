import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'PIDNet'))
import models.aurora2w
import datasets
from configs import config
from utils.criterion import CrossEntropy, BondaryLoss
from utils.function import validate

class DummyWriter:
    def add_scalar(self, *args, **kwargs): pass

class AuRoRA2W_FullModel(nn.Module):
    def __init__(self, model, sem_loss, bd_loss):
        super(AuRoRA2W_FullModel, self).__init__()
        self.model = model
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss

    def forward(self, inputs, labels, bd_gts):
        B = inputs.size(0)
        roll_angles = torch.zeros(B, device=inputs.device)
        out = self.model(inputs, roll_angles=roll_angles)
        
        ph, pw = out[0].size(2), out[0].size(3)
        h, w = labels.size(1), labels.size(2)
        if ph != h or pw != w:
            for i in range(len(out)):
                out[i] = F.interpolate(out[i], size=(h, w), mode='bilinear', align_corners=True)
        
        loss1 = self.sem_loss([out[0]], labels)
        loss3 = self.bd_loss(out[2], bd_gts)
        acc = torch.tensor([0.0], device=inputs.device)
        return loss1 + loss3, [out[0], out[1]], acc, [loss1, loss3]

def main():
    import argparse
    from configs import update_config
    args = argparse.Namespace(cfg="models/PIDNet/configs/cityscapes/pidnet_small_cityscapes.yaml", opts=[])
    update_config(config, args)

    config.defrost()
    config.DATASET.ROOT = 'models/PIDNet/data/'
    config.MODEL.PRETRAINED = 'models/PIDNet/pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar'
    config.TRAIN.IMAGE_SIZE = [512, 512]
    config.TEST.BATCH_SIZE_PER_GPU = 2
    config.GPUS = (0,)
    config.freeze()

    valid_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=(512, 512))

    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.TEST.BATCH_SIZE_PER_GPU, shuffle=False, num_workers=0)

    sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL, weight=valid_dataset.class_weights)
    bd_criterion = BondaryLoss()

    model = models.aurora2w.get_seg_model(config, imgnet_pretrained=False)
    model = AuRoRA2W_FullModel(model, sem_criterion, bd_criterion).cuda()

    checkpoint_path = 'output/cityscapes/pidnet_small_cityscapes/aurora2w_epoch_116.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # We load into model directly (which is AuRoRA2W_FullModel)
    # The checkpoint might have module. prefix if saved with DataParallel, wait, our manual save had `model.module.state_dict()`.
    # Let's clean just in case.
    cleaned_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_dict, strict=False)

    writer_dict = {'writer': DummyWriter(), 'valid_global_steps': 0}
    
    print("[*] Running Validation on Epoch 116...")
    valid_loss, mean_IoU, IoU_array = validate(config, validloader, model, writer_dict)
    
    print(f"\n[+] Validation Complete!")
    print(f"Mean IoU: {mean_IoU:.4f}")
    print(f"Per-Class IoU: \n{np.round(IoU_array, 4)}")

if __name__ == '__main__':
    main()
