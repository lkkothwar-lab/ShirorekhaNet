import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import your custom modules
from dataset import SanskritLineDataset, get_transforms
from model import ShirorekhaNet
from loss import SanskritLoss, HyperAggressiveAdaptiveMetaTuner
from utils import calculate_segmentation_metrics, ModelSaver, Visualizer

def main():
    # --- Configuration ---
    DEVICE = torch.cuda.is_available() and "cuda" or "cpu"
    IMG_SIZE = 512
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 2e-4
    
    # Paths (Relative to where train.py is run)
    TRAIN_IMG_DIR = "./"
    TRAIN_MASK_DIR = "."
    VAL_IMG_DIR = "./"
    VAL_MASK_DIR = "./"
    
    print(f"Starting training on {DEVICE}...")

    # --- Data Loading ---
    train_transform, val_transform = get_transforms(IMG_SIZE)
    
    train_dataset = SanskritLineDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    val_dataset = SanskritLineDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Initialization ---
    model = ShirorekhaNet(in_channels=4).to(DEVICE)
    loss_fn = SanskritLoss(device=DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)
    tuner = HyperAggressiveAdaptiveMetaTuner(initial_lr=LR)
    saver = ModelSaver(save_dir="./checkpoints")
    
    # --- Training Loop ---
    for epoch in range(start_epoch, TOTAL_NUM_EPOCHS):

    print(f"\n========== STARTING EPOCH {epoch} ==========")

    # -------------------------
    # ADAPTIVE TUNING - Apply parameters for the CURRENT epoch BEFORE training
    # -------------------------
    current_metrics = {'iou': 0, 'precision': 0, 'recall': 0}
    if epoch > 0: # For subsequent epochs, use metrics from the *previous* epoch
        if history['val_iou']:
            current_metrics = {
                'iou': history['val_iou'][-1],
                'precision': val_precision_prev, # Use stored previous values
                'recall': val_recall_prev
            }

    params = tuner.update(
        epoch,
        metrics=current_metrics,
        optimizer=optimizer
    )

    loss_fn.dice_weight = params['dice_weight']
    loss_fn.bce_weight = params['bce_weight']
    loss_fn.perceptual_weight = params['perceptual_weight']

    # -------------------------
    # TRAIN
    # -------------------------
    train_loss = train_epoch(
        train_loader,
        epoch,
        generator,
        optimizer,
        loss_fn,
        train_dataset,
        current_strategy={
            'name': 'Warmup Phase' if epoch < 4 else 'Adaptive Strategy',
            'g_lr': optimizer.param_groups[0]['lr'], # LR is already updated by tuner
            'dice_weight': loss_fn.dice_weight,
            'bce_weight': loss_fn.bce_weight,
            'perceptual_weight': loss_fn.perceptual_weight,
            'augmentation_preset_key': 'no_aug'
        }
    )

    history['train_loss'].append(train_loss)

    # -------------------------
    # VALIDATE
    # -------------------------
    val_iou, val_dice, val_precision, val_recall, val_f1 = \
        validate_epoch(val_loader, epoch, generator)

    history['val_iou'].append(val_iou)
    val_precision_prev = val_precision # Store for next epoch's tuner update
    val_recall_prev = val_recall       # Store for next epoch's tuner update

    # -------------------------
    # SAVE (ONLY ONCE)
    # -------------------------
    saver.save_checkpoint(epoch, generator, optimizer)
    saver.save_full_model(epoch, generator)

    # -------------------------
    # VISUALIZE (ONLY ONCE)
    # -------------------------
    visualizer.visualize_and_save(
        epoch,
        generator,
        val_loader,
        device
    )

    print(f"\nEpoch {epoch} Updated Parameters (for next epoch):")
    print(f"LR: {params['lr']:.6f}")
    print(f"Dice: {params['dice_weight']:.3f}")
    print(f"BCE: {params['bce_weight']:.3f}")
    print(f"Perceptual: {params['perceptual_weight']:.3f}")

    # -------------------------
    # TRACK BEST (NO EXTRA SAVE)
    # -------------------------
    if val_iou > best_iou:
        best_iou = val_iou
        print(f"🔥 New Best IoU: {best_iou:.4f}")

    # Early aggressive stop
    if val_iou >= 0.92:
        print(f"\n🎉 TARGET IoU \u2265 0.95 ACHIEVED at Epoch {epoch}")
        break

    print("="*60)

print("\nTraining Complete.")
print(f"Best IoU Achieved: {best_iou:.4f}")

if __name__ == "__main__":
    # Optimize CUDA memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()