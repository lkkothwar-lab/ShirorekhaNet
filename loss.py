import torch
import torch.nn as nn

class SanskritLoss(nn.Module):
    def __init__(self, dice_weight=0.8, bce_weight=0.2, perceptual_weight=0.0, device=None):
        super().__init__()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.perceptual_weight = perceptual_weight
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # BCE with logits (stable version)
        self.bce = nn.BCEWithLogitsLoss()

        # Initialize VGG for perceptual loss unconditionally
        vgg16_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg = vgg16_model.features[:16].to(self.device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    # -----------------------------
    # Dice Loss (expects logits)
    # -----------------------------
    def dice_loss(self, logits, target, smooth=1e-6):
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    # -----------------------------
    # Perceptual Loss (optional)
    # -----------------------------
    def perceptual_loss(self, logits, target):
        probs = torch.sigmoid(logits)

        if probs.shape[-2:] != target.shape[-2:]:
            probs = F.interpolate(probs, size=target.shape[-2:], mode='bilinear', align_corners=False)

        probs_3 = probs.repeat(1, 3, 1, 1)
        target_3 = target.repeat(1, 3, 1, 1)

        pred_features = self.vgg(probs_3)
        target_features = self.vgg(target_3)

        return F.l1_loss(pred_features, target_features)
        pass
        
    def forward(self, logits, target):

        dice_l = self.dice_loss(logits, target)
        bce_l = self.bce(logits, target)

        perceptual_l = 0
        if self.perceptual_weight > 0:
            perceptual_l = self.perceptual_loss(logits, target)

        total_loss = (
            self.dice_weight * dice_l +
            self.bce_weight * bce_l +
            self.perceptual_weight * perceptual_l
        )
        return total_loss, dice_l.item(), bce_l.item(), perceptual_l.item() if self.perceptual_weight > 0 else 0
        pass

class HyperAggressiveAdaptiveMetaTuner:

    def __init__(self, initial_lr=2e-4):

        self.base_lr = initial_lr

        # Initial weights
        self.dice_weight = 0.8
        self.bce_weight = 0.2
        self.perceptual_weight = 0.0

        # Strong PID Gains (Aggressive)
        self.Kp = 0.20
        self.Ki = 0.10
        self.Kd = 0.02

        self.integral = 0.0
        self.prev_error = 0.0

        # Perceptual control
        self.perceptual_max = 0.12
        self.perceptual_ramp_epochs = 4
        self.perceptual_active = False
        self.activation_epoch = None
        self.iou_threshold_for_perceptual = 0.90 # New threshold for perceptual loss activation

    def update(self, epoch, metrics, optimizer):

        lr = 0 # Initialize lr here to ensure it's always defined

        # =========================
        # STAGE 1 \u2014 FIXED WARMUP (Epochs 0-4)
        # =========================
        if epoch == 0:
            lr = 2e-4
            self.dice_weight = 0.8
            self.bce_weight = 0.2
            self.perceptual_weight = 0.0

        elif epoch == 1:
            lr = 1.5e-4
            self.dice_weight = 0.7
            self.bce_weight = 0.3
            self.perceptual_weight = 0.0

        elif epoch == 2:
            lr = 1e-4
            self.dice_weight = 0.6
            self.bce_weight = 0.4
            self.perceptual_weight = 0.0

        elif epoch == 3:
            lr = 5e-5
            self.dice_weight = 0.5
            self.bce_weight = 0.5
            self.perceptual_weight = 0.0

        elif epoch == 4: # Fixed parameters for epoch 4
            lr = 4e-5
            self.dice_weight = 0.45
            self.bce_weight = 0.55
            self.perceptual_weight = 0.0

        # =========================
        # STAGE 2 \u2014 AGGRESSIVE ADAPTIVE TUNING (from epoch 5 onwards)
        # =========================
        elif epoch >= 5:

            iou = metrics.get('iou', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)

            # ---- Controlled LR schedule (no oscillation) ----

            if epoch == 5:
                lr = 3e-5
            elif epoch == 6:
                lr = 2e-5
            elif epoch == 7:
                lr = 1e-5
            elif epoch == 8:
                lr = 5e-6
            else:
                lr = 2.5e-6

            # ---- PID Precision-Recall Control ----
            error = recall - precision
            self.integral += error
            derivative = error - self.prev_error

            adjustment = (
                self.Kp * error +
                self.Ki * self.integral +
                self.Kd * derivative
            )

            # Encourage BCE dominance early for boundary sharpening
            if epoch <= 8:
                self.bce_weight = np.clip(0.55 + adjustment, 0.45, 0.7)
            else:
                self.bce_weight = np.clip(0.5 + adjustment, 0.3, 0.7)

            self.prev_error = error

            # ---- Perceptual loss activation based on IoU threshold ----
            if not self.perceptual_active and iou >= self.iou_threshold_for_perceptual:
                self.activation_epoch = epoch
                self.perceptual_active = True

            if self.perceptual_active:
                progress = min(
                    1.0,
                    (epoch - self.activation_epoch) /
                    self.perceptual_ramp_epochs
                )
                self.perceptual_weight = (
                    self.perceptual_max *
                    (1 / (1 + np.exp(-8 * (progress - 0.5))))
                )
            else:
                self.perceptual_weight = 0.0 # Ensure it's 0 if not active yet

            self.dice_weight = 1.0 - self.bce_weight - self.perceptual_weight

        # Apply LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return {
            'lr': lr,
            'dice_weight': float(self.dice_weight),
            'bce_weight': float(self.bce_weight),
            'perceptual_weight': float(self.perceptual_weight)
        }

    def get_state(self):
        return {
            'dice_weight': self.dice_weight,
            'bce_weight': self.bce_weight,
            'perceptual_weight': self.perceptual_weight,
            'integral': self.integral,
            'prev_error': self.prev_error,
            'perceptual_active': self.perceptual_active,
            'activation_epoch': self.activation_epoch,
            'iou_threshold_for_perceptual': self.iou_threshold_for_perceptual,
            'base_lr': self.base_lr # Also save base_lr for consistency
        }

    def load_state(self, state_dict):
        self.dice_weight = state_dict['dice_weight']
        self.bce_weight = state_dict['bce_weight']
        self.perceptual_weight = state_dict['perceptual_weight']
        self.integral = state_dict['integral']
        self.prev_error = state_dict['prev_error']
        self.perceptual_active = state_dict['perceptual_active']
        self.activation_epoch = state_dict['activation_epoch']
        self.iou_threshold_for_perceptual = state_dict['iou_threshold_for_perceptual']
        self.base_lr = state_dict['base_lr']
    pass