import os, glob, csv, math, time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ShirorekhaDataset
from model import ShirorekhaNet
from loss import SanskritLoss, CosineLR_PID_Curriculum
from utils import compute_all_metrics, visualize_epoch

def train_one_epoch(model, loader, opt, loss_fn, device, model_name):
    model.train()
    total_loss = dice_t = bce_t = perc_t = 0.0
    pbar = tqdm(loader, desc=f'Train {model_name}')
    for inputs, masks in pbar:
        inputs, masks = inputs.to(device), masks.to(device)
        opt.zero_grad()
        logits = model(inputs)
        loss, d, b, p = loss_fn(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item(); dice_t += d; bce_t += b; perc_t += p
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    n = len(loader)
    return total_loss/n, dice_t/n, bce_t/n, perc_t/n

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    metrics_sum = {'iou':0,'dice':0,'f1':0,'precision':0,'recall':0,
                   'detection_rate':0,'shirorekha_accuracy':0}
    count = 0
    for imgs, masks in tqdm(loader, desc='Val'):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        batch_metrics = compute_all_metrics(logits, masks)
        for k in metrics_sum: metrics_sum[k] += batch_metrics[k]
        count += 1
    return {k: v/count for k,v in metrics_sum.items()}

def main():
    # --- Configuration (modify paths as needed) ---
    BASE_PATH = "/content/drive/MyDrive/ShirorekhaNet"   ##### Adjust as per your requirement
    TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_data/Trn_img")
    TRAIN_MASK_DIR = os.path.join(BASE_PATH, "train_data/Trn_lmask")
    VAL_IMG_DIR   = os.path.join(BASE_PATH, "val_data/Vld_img")
    VAL_MASK_DIR  = os.path.join(BASE_PATH, "val_data/Vld_lmask")
    OUTPUT_DIR    = os.path.join(BASE_PATH, 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    IMG_SIZE = 512
    BATCH_SIZE = 2
    NUM_EPOCHS = 20
    NUM_WORKERS = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- Datasets ---
    train_ds = ShirorekhaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMG_SIZE, True)
    val_ds   = ShirorekhaDataset(VAL_IMG_DIR, VAL_MASK_DIR, IMG_SIZE, False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Fixed validation images for visualization
    val_img_files = sorted(os.listdir(VAL_IMG_DIR))[:4]
    val_img_paths = [os.path.join(VAL_IMG_DIR, f) for f in val_img_files]
    val_mask_paths = [os.path.join(VAL_MASK_DIR, f) for f in val_img_files]

    # --- Model, Loss, Optimizer, Curriculum ---
    model = ShirorekhaNet(in_channels=4).to(device)
    loss_fn = SanskritLoss(dice_weight=0.6, bce_weight=0.4,
                           perceptual_weight=0.0, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    curriculum = CosineLR_PID_Curriculum(
        lr_max=2.54e-6, lr_min=1e-7, T_max=10,
        initial_dice=0.6, initial_bce=0.4,
        epoch_offset=10
    )

    # --- Resume from checkpoint if exists ---
    start_epoch = 0
    ckpt_files = glob.glob(os.path.join(OUTPUT_DIR, "ShirorekhaNet_epoch_*.pth"))
    if ckpt_files:
        latest = sorted(ckpt_files, key=lambda x: int(x.split('_epoch_')[-1].split('.')[0]))[-1]
        print(f'Resuming from checkpoint: {latest}')
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        curriculum.load_state(ckpt['curriculum_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed at epoch {start_epoch+1}')

    # --- Logging ---
    log_csv = os.path.join(OUTPUT_DIR, 'training_log_shirorekha.csv')
    csv_file = open(log_csv, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(log_csv) == 0:
        csv_writer.writerow(['epoch','lr','dice_weight','bce_weight','perceptual_weight',
                             'train_loss','train_dice','train_bce','train_perceptual',
                             'iou','dice','f1','precision','recall',
                             'detection_rate','shirorekha_accuracy'])

    # --- Training Loop ---
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Update loss weights & LR
        metrics_prev = {'iou':0.0, 'precision':0.0, 'recall':0.0}
        # (if not first epoch, we could pass last val metrics – here we start with zero)
        params = curriculum.update(epoch, metrics_prev, optimizer)
        loss_fn.dice_weight = params['dice_weight']
        loss_fn.bce_weight = params['bce_weight']
        loss_fn.perceptual_weight = params['perceptual_weight']

        # Train
        avg_loss, avg_d, avg_b, avg_p = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, 'ShirorekhaNet')
        # Validate
        val_metrics = validate(model, val_loader, device)

        # Log
        csv_writer.writerow([
            epoch, params['lr'], params['dice_weight'], params['bce_weight'],
            params['perceptual_weight'],
            avg_loss, avg_d, avg_b, avg_p,
            val_metrics['iou'], val_metrics['dice'], val_metrics['f1'],
            val_metrics['precision'], val_metrics['recall'],
            val_metrics['detection_rate'], val_metrics['shirorekha_accuracy']
        ])
        csv_file.flush()

        # Save checkpoint
        ckpt_path = os.path.join(OUTPUT_DIR, f'ShirorekhaNet_epoch_{epoch:03d}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'curriculum_state': curriculum.get_state()
        }, ckpt_path)

        # Visualize every epoch (optional)
        visualize_epoch(epoch, model, val_img_paths, val_mask_paths,
                        device, OUTPUT_DIR)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} – IoU: {val_metrics['iou']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, F1: {val_metrics['f1']:.4f}")

    csv_file.close()
    print('Training complete.')

if __name__ == '__main__':
    main()
