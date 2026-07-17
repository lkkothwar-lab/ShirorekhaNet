import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SanskritLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5,
                 perceptual_weight=0.0, device=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.perceptual_weight = perceptual_weight
        self.device = device if device else torch.device('cpu')
        self.bce = nn.BCEWithLogitsLoss()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg = vgg16.features[:16].to(self.device).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def dice_loss(self, logits, target, smooth=1e-6):
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = (2.*intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def perceptual_loss(self, logits, target):
        probs = torch.sigmoid(logits)
        if probs.shape[-2:] != target.shape[-2:]:
            probs = F.interpolate(probs, size=target.shape[-2:],
                                  mode='bilinear', align_corners=False)
        probs_3 = probs.repeat(1,3,1,1)
        target_3 = target.repeat(1,3,1,1)
        return F.l1_loss(self.vgg(probs_3), self.vgg(target_3))

    def forward(self, logits, target):
        dice_l = self.dice_loss(logits, target)
        bce_l = self.bce(logits, target)
        perc_l = torch.tensor(0.0, device=self.device)
        if self.perceptual_weight > 0:
            perc_l = self.perceptual_loss(logits, target)
        total = (self.dice_weight * dice_l +
                 self.bce_weight * bce_l +
                 self.perceptual_weight * perc_l)
        return total, dice_l.item(), bce_l.item(), perc_l.item()


class CosineLR_PID_Curriculum:
    """
    Cosine learning rate schedule + PID loss‑weight balancing from epoch 0.
    """
    def __init__(self, lr_max=1e-4, lr_min=1e-7, T_max=20,
                 initial_dice=0.5, initial_bce=0.5,
                 Kp=0.20, Ki=0.10, Kd=0.02,
                 iou_threshold=0.87, perceptual_max=0.0,
                 perceptual_ramp_epochs=4, epoch_offset=0):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max
        self.dice_weight = initial_dice
        self.bce_weight  = initial_bce
        self.perceptual_weight = 0.0
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.iou_threshold = iou_threshold
        self.perceptual_max = perceptual_max
        self.perceptual_ramp_epochs = perceptual_ramp_epochs
        self.perceptual_active = False
        self.activation_epoch = None
        self.epoch_offset = epoch_offset

    def update(self, epoch, metrics, optimizer):
        t = max(0, epoch - self.epoch_offset)
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * \
             (1 + math.cos(math.pi * t / self.T_max))

        precision = metrics.get('precision', 0.0)
        recall    = metrics.get('recall', 0.0)
        error = recall - precision
        self.integral += error
        derivative = error - self.prev_error
        adjustment = (self.Kp * error +
                      self.Ki * self.integral +
                      self.Kd * derivative)
        self.bce_weight = np.clip(self.bce_weight + adjustment, 0.35, 0.65)
        self.prev_error = error

        iou = metrics.get('iou', 0.0)
        if not self.perceptual_active and iou >= self.iou_threshold:
            self.perceptual_active = True
            self.activation_epoch = epoch
        if self.perceptual_active:
            progress = min(1.0, (epoch - self.activation_epoch) / self.perceptual_ramp_epochs)
            self.perceptual_weight = self.perceptual_max * \
                                     (1 / (1 + np.exp(-8 * (progress - 0.5))))
        else:
            self.perceptual_weight = 0.0
        self.dice_weight = 1.0 - self.bce_weight - self.perceptual_weight

        for g in optimizer.param_groups:
            g['lr'] = lr

        return {'lr': lr, 'dice_weight': self.dice_weight,
                'bce_weight': self.bce_weight,
                'perceptual_weight': self.perceptual_weight}

    def get_state(self):
        return {
            'dice_weight': self.dice_weight,
            'bce_weight': self.bce_weight,
            'perceptual_weight': self.perceptual_weight,
            'integral': self.integral,
            'prev_error': self.prev_error,
            'perceptual_active': self.perceptual_active,
            'activation_epoch': self.activation_epoch
        }

    def load_state(self, state):
        self.dice_weight = state['dice_weight']
        self.bce_weight = state['bce_weight']
        self.perceptual_weight = state['perceptual_weight']
        self.integral = state['integral']
        self.prev_error = state['prev_error']
        self.perceptual_active = state['perceptual_active']
        self.activation_epoch = state['activation_epoch']
