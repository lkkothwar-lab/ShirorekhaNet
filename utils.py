import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_shirorekha_accuracy(pred, target, thickness=3):
    pred_img = (pred[0,0].cpu().numpy()*255).astype(np.uint8)
    target_img = (target[0,0].cpu().numpy()*255).astype(np.uint8)

    pred_lines = cv2.HoughLinesP(pred_img, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=5)
    target_lines = cv2.HoughLinesP(target_img, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=5)

    if target_lines is None or len(target_lines) == 0:
        return 1.0

    if pred_lines is None or len(pred_lines) == 0:
        return 0.0

    matched = 0
    for t_line in target_lines:
        if any(abs(t_line[0][1] - p_line[0][1]) < thickness for p_line in pred_lines):
            matched += 1
    return matched / len(target_lines)
    pass

def calculate_segmentation_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_flat = target.view(-1)
    pred_flat = pred_bin.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    iou = (intersection / (union + 1e-8)).item()

    dice = (2. * intersection / (pred_flat.sum() + target_flat.sum() + 1e-8)).item()

    tp = intersection
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()

    precision = (tp / (tp + fp + 1e-8)).item()

    recall = (tp / (tp + fn + 1e-8)).item()

    f1 = (2 * precision * recall / (precision + recall + 1e-8))

    return iou, dice, precision, recall, f1
    pass

def calculate_detection_rate(pred_mask, gt_mask, iou_threshold=0.5):
    if pred_mask.dim() == 4 and pred_mask.shape[0] > 1:
        pred_mask_np = (pred_mask[0,0].cpu().numpy() * 255).astype(np.uint8)
        gt_mask_np = (gt_mask[0,0].cpu().numpy() * 255).astype(np.uint8)
    else:
        pred_mask_np = (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        gt_mask_np = (gt_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

    gt_contours, _ = cv2.findContours(gt_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gt_bboxes = [cv2.boundingRect(c) for c in gt_contours]
    pred_bboxes = [cv2.boundingRect(c) for c in pred_contours]

    if not gt_bboxes:
        return 1.0

    if not pred_bboxes:
        return 0.0

    detected_gt_count = 0

    for gt_bbox in gt_bboxes:
        gt_x, gt_y, gt_w, gt_h = gt_bbox
        gt_area = gt_w * gt_h

        is_detected = False
        for pred_bbox in pred_bboxes:
            pr_x, pr_y, pr_w, pr_h = pred_bbox
            pr_area = pr_w * pr_h

            x_left = max(gt_x, pr_x)
            y_top = max(gt_y, pr_y)
            x_right = min(gt_x + gt_w, pr_x + pr_w)
            y_bottom = min(gt_y + gt_h, pr_y + pr_h)

            if x_right < x_left or y_bottom < y_top:
                intersection_area = 0
            else:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)

            union_area = gt_area + pr_area - intersection_area

            if union_area > 0:
                iou = intersection_area / union_area
                if iou >= iou_threshold:
                    is_detected = True
                    break
        if is_detected:
            detected_gt_count += 1

    return detected_gt_count / len(gt_bboxes)

def train_epoch(dataloader, epoch, generator, optimizer, loss_fn, train_dataset, current_strategy):
    generator.train()

    # Apply current augmentation
    # train_dataset.augmentations = augmentation_presets[current_strategy['augmentation_preset_key']]

    # Update loss weights
    loss_fn.dice_weight = current_strategy['dice_weight']
    loss_fn.bce_weight = current_strategy['bce_weight']
    loss_fn.perceptual_weight = current_strategy['perceptual_weight']

    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_strategy['g_lr']

    # Print to both log and console
    epoch_start_msg = f"\n--- Epoch {epoch} ({current_strategy['name']}) ---"
    print(epoch_start_msg)

    epoch_params_msg = (f"Epoch {epoch} parameters: LR: {current_strategy['g_lr']:.6f}, Dice Weight: {current_strategy['dice_weight']:.1f}, "
                        f"BCE Weight: {current_strategy['bce_weight']:.1f}, Perceptual Weight: {current_strategy['perceptual_weight']:.1f}")
    print(epoch_params_msg)

    total_loss = 0
    dice_loss_total = 0
    bce_loss_total = 0
    perceptual_loss_total = 0

    # Redirect tqdm output to original_stderr so it doesn't interfere with batch prints on original_stdout
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=True)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = generator(inputs)

        # Calculate loss
        loss, dice_l, bce_l, perceptual_l = loss_fn(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        dice_loss_total += dice_l
        bce_loss_total += bce_l
        perceptual_loss_total += perceptual_l

        # Explicitly print losses for each batch to both log and console
        batch_log_msg = f"Batch {batch_idx+1}/{len(dataloader)} - Total Loss: {loss.item():.4f}, Dice Loss: {dice_l:.4f}, BCE Loss: {bce_l:.4f}, Perceptual Loss: {perceptual_l:.4f}"
        print(batch_log_msg) # removed original_stdout

        # Update progress bar (removed loss display here as it's now explicitly printed)
        progress_bar.set_postfix({'Current Avg Loss': f'{total_loss/(batch_idx+1):.4f}'}, refresh=False)

    # Average losses
    avg_total_loss = total_loss / len(dataloader)
    avg_dice_loss = dice_loss_total / len(dataloader)
    avg_bce_loss = bce_loss_total / len(dataloader)
    avg_perceptual_loss = perceptual_loss_total / len(dataloader)

    avg_loss_msg = (f"Epoch {epoch} Training: Total Loss: {avg_total_loss:.4f} | "
                    f"Dice: {avg_dice_loss:.4f} | BCE: {avg_bce_loss:.4f} | "
                    f"Perceptual: {avg_perceptual_loss:.4f}")
    print(avg_loss_msg)

    return avg_total_loss

# --- Validation Function (Logits Version) ---
def validate_epoch(dataloader, epoch, generator, threshold=0.5):
    generator.eval()

    shirorekha_acc_total = 0
    iou_total = 0
    dice_total = 0
    precision_total = 0
    recall_total = 0
    f1_total = 0
    detection_rate_total = 0
    count = 0

    with torch.no_grad():
        # Redirect tqdm output to original_stderr
        progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch}", leave=True)

        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # \u25c6 Model now returns LOGITS
            logits = generator(inputs)

            # \u25c6 Convert logits \u2192 probabilities
            probs = torch.sigmoid(logits)

            # \u25c6 Apply threshold
            preds_thresholded = (probs > threshold).float()

            # -----------------------------
            # Calculate metrics
            # -----------------------------
            shirorekha_acc_total += calculate_shirorekha_accuracy(
                preds_thresholded, targets
            )

            iou, dice, precision, recall, f1 = calculate_segmentation_metrics(
                preds_thresholded, targets
            )

            detection_rate_total += calculate_detection_rate(
                preds_thresholded, targets
            )

            iou_total += iou
            dice_total += dice
            precision_total += precision
            recall_total += recall
            f1_total += f1
            count += 1

            progress_bar.set_postfix({'IoU': f'{iou_total/count:.4f}'}, refresh=False)

    if count > 0:
        shirorekha_acc_avg = shirorekha_acc_total / count
        iou_avg = iou_total / count
        dice_avg = dice_total / count
        precision_avg = precision_total / count
        recall_avg = recall_total / count
        f1_avg = f1_total / count
        detection_rate_avg = detection_rate_total / count

        # Print to both log and console
        val_summary_msg = f"\nEpoch {epoch} Validation:"
        print(val_summary_msg)

        metrics_msg_1 = f"  Shirorekha Acc: {shirorekha_acc_avg:.2%}"
        print(metrics_msg_1)

        metrics_msg_2 = f"  IoU: {iou_avg:.4f} | Dice: {dice_avg:.4f}"
        print(metrics_msg_2)

        metrics_msg_3 = f"  Precision: {precision_avg:.4f} | Recall: {recall_avg:.4f} | F1: {f1_avg:.4f}"
        print(metrics_msg_3)

        metrics_msg_4 = f"  Detection Rate: {detection_rate_avg:.4f}"
        print(metrics_msg_4)

        return iou_avg, dice_avg, precision_avg, recall_avg, f1_avg
    else:
        warning_msg = "Warning: No validation batches processed."
        print(warning_msg)
        return 0, 0, 0, 0, 0

# --- Visualization Function ---
def visualize_results(generator, dataloader, epoch=None, num_samples=3):
    generator.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_samples:
                break

            inputs = inputs.to(device)
            preds = generator(inputs)

            input_img = inputs[0,:3].cpu().permute(1,2,0).numpy()
            target_img = targets[0,0].cpu().numpy()
            pred_img = preds[0,0].cpu().numpy()
            pred_thresholded = (pred_img > 0.5).astype(np.float32)

            fig, axs = plt.subplots(1, 4, figsize=(20, 5))

            axs[0].imshow(input_img)
            axs[0].set_title("Original")
            axs[0].axis('off')

            axs[1].imshow(target_img, cmap='gray')
            axs[1].set_title("Ground Truth")
            axs[1].axis('off')

            axs[2].imshow(pred_img, cmap='gray')
            axs[2].set_title("Raw Output")
            axs[2].axis('off')

            axs[3].imshow(pred_thresholded, cmap='gray')
            axs[3].set_title("Threshold Output")
            axs[3].axis('off')

            plt.tight_layout()
            if epoch is not None:
                filename = f"epoch_{epoch:03d}_sample_{i:02d}.png"
            else:
                filename = f"sample_{i:02d}.png"
            filepath = os.path.join(VISUALIZATION_PATH_TRAIN, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.show()
            plt.close(fig)

class PathManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir

        self.checkpoint_dir = os.path.join(base_dir, "Cp")
        self.model_dir = os.path.join(base_dir, "Hm")
        self.visual_dir = os.path.join(base_dir, "Visuals_Train")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.visual_dir, exist_ok=True)

        print("Checkpoint Dir:", self.checkpoint_dir)
        print("Model Dir:", self.model_dir)
        print("Visual Dir:", self.visual_dir)

class ModelSaver:

    def __init__(self, path_manager):
        self.paths = path_manager

    def save_checkpoint(self, epoch, model, optimizer, tuner_state, val_iou=0, val_precision=0, val_recall=0):
        checkpoint_path = os.path.join(
            self.paths.checkpoint_dir,
            f"checkpoint_epoch_{epoch:03d}.pth"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'tuner_state_dict': tuner_state, # Save tuner state
            'val_iou': val_iou,
            'val_precision': val_precision,
            'val_recall': val_recall
        }, checkpoint_path)

        print(f"✅ Checkpoint saved → {checkpoint_path}")

    def save_full_model(self, epoch, model):
        model_path = os.path.join(
            self.paths.model_dir,
            f"model_epoch_{epoch:03d}.pth"
        )

        torch.save(model.state_dict(), model_path)

        print(f"✅ Model saved → {model_path}")
    pass

class Visualizer:

    def __init__(self, path_manager):
        self.paths = path_manager

    def visualize_and_save(self, epoch, model, val_loader, device):

        model.eval()

        epoch_dir = os.path.join(
            self.paths.visual_dir,
            f"Epoch_{epoch:03d}"
        )
        os.makedirs(epoch_dir, exist_ok=True)

        print("\n📸 Visualizing first 2 validation images...\n")

        count = 0

        with torch.no_grad():
            for images, masks in val_loader:

                images = images.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs)

                for i in range(images.size(0)):

                    if count >= 2:
                        break

                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    gt = masks[i].cpu().squeeze().numpy()
                    pred_prob = preds[i].cpu().squeeze().numpy()

                    # Normalize image
                    img_vis = (img - img.min()) / (img.max() - img.min() + 1e-8)

                    # Convert to uint8
                    pred_uint8 = (pred_prob * 255).astype(np.uint8)

                    # OTSU Threshold
                    threshold_val, pred_bin = cv2.threshold(
                        pred_uint8,
                        0,
                        255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )

                    print(f"Otsu Threshold (Image {count}): {threshold_val}")

                    gt_bin = (gt > 0.5).astype(np.uint8) * 255

                    # Save
                    cv2.imwrite(os.path.join(epoch_dir, f"{count}_original.png"),
                                (img_vis * 255).astype(np.uint8))

                    cv2.imwrite(os.path.join(epoch_dir, f"{count}_groundtruth.png"),
                                gt_bin)

                    cv2.imwrite(os.path.join(epoch_dir, f"{count}_prediction_prob.png"),
                                pred_uint8)

                    cv2.imwrite(os.path.join(epoch_dir, f"{count}_prediction_binary.png"),
                                pred_bin)

                    # Display
                    plt.figure(figsize=(12,4))

                    plt.subplot(1,4,1)
                    plt.imshow(img_vis)
                    plt.title("Original")
                    plt.axis("off")

                    plt.subplot(1,4,2)
                    plt.imshow(gt_bin, cmap='gray')
                    plt.title("Ground Truth")
                    plt.axis("off")

                    plt.subplot(1,4,3)
                    plt.imshow(pred_prob, cmap='gray')
                    plt.title("Prediction (Prob)")
                    plt.axis("off")

                    plt.subplot(1,4,4)
                    plt.imshow(pred_bin, cmap='gray')
                    plt.title("Prediction (Binary)")
                    plt.axis("off")

                    plt.show()

                    count += 1

                if count >= 2:
                    break

        print(f"✅ Visuals saved to → {epoch_dir}")
    pass