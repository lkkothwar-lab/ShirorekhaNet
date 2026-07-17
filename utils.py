import os, cv2, numpy as np, torch, matplotlib.pyplot as plt

def calculate_iou(pred, gt, smooth=1e-8):
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()

def calculate_dice(pred, gt, smooth=1e-8):
    intersection = (pred * gt).sum()
    return ((2.*intersection + smooth) / (pred.sum() + gt.sum() + smooth)).item()

def calculate_precision_recall(pred, gt, smooth=1e-8):
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision.item(), recall.item()

def calculate_shirorekha_accuracy(pred, target, thickness=3):
    pred_img = (pred[0,0].cpu().numpy()*255).astype(np.uint8)
    target_img = (target[0,0].cpu().numpy()*255).astype(np.uint8)
    pred_lines = cv2.HoughLinesP(pred_img, 1, np.pi/180, threshold=50,
                                 minLineLength=100, maxLineGap=5)
    target_lines = cv2.HoughLinesP(target_img, 1, np.pi/180, threshold=50,
                                   minLineLength=100, maxLineGap=5)
    if target_lines is None or len(target_lines) == 0:
        return 1.0 if pred_lines is None or len(pred_lines) == 0 else 0.0
    if pred_lines is None or len(pred_lines) == 0:
        return 0.0
    matched = 0
    for t_line in target_lines:
        if any(abs(t_line[0][1] - p_line[0][1]) < thickness for p_line in pred_lines):
            matched += 1
    return matched / len(target_lines)

def calculate_detection_rate(pred, gt, iou_threshold=0.5):
    def to_uint8(t):
        if t.dim()==4: t = t[0,0]
        elif t.dim()==3: t = t[0]
        arr = t.cpu().numpy(); arr = (arr>0.5).astype(np.uint8)*255
        return arr.squeeze()
    pred_np = to_uint8(pred); gt_np = to_uint8(gt)
    if pred_np.ndim!=2: pred_np = pred_np.squeeze()
    if gt_np.ndim!=2: gt_np = gt_np.squeeze()
    pred_np = pred_np.astype(np.uint8); gt_np = gt_np.astype(np.uint8)
    gt_contours, _ = cv2.findContours(gt_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not gt_contours: return 1.0 if not pred_contours else 0.0
    if not pred_contours: return 0.0
    gt_bboxes = [cv2.boundingRect(c) for c in gt_contours]
    pred_bboxes = [cv2.boundingRect(c) for c in pred_contours]
    detected = 0
    for gx,gy,gw,gh in gt_bboxes:
        for px,py,pw,ph in pred_bboxes:
            ix = max(gx,px); iy = max(gy,py)
            iw = min(gx+gw, px+pw) - ix; ih = min(gy+gh, py+ph) - iy
            if iw<=0 or ih<=0: continue
            inter = iw*ih; union = gw*gh + pw*ph - inter
            if inter/union >= iou_threshold:
                detected += 1
                break
    return detected / len(gt_bboxes)

def compute_all_metrics(logits, gt, threshold=0.5):
    probs = torch.sigmoid(logits)
    pred_bin = (probs > threshold).float()
    iou = calculate_iou(pred_bin, gt)
    dice = calculate_dice(pred_bin, gt)
    precision, recall = calculate_precision_recall(pred_bin, gt)
    f1 = dice
    det_rate = calculate_detection_rate(pred_bin, gt)
    shiro_acc = calculate_shirorekha_accuracy(pred_bin, gt)
    return {'iou': iou, 'dice': dice, 'f1': f1,
            'precision': precision, 'recall': recall,
            'detection_rate': det_rate,
            'shirorekha_accuracy': shiro_acc}

def visualize_epoch(epoch, model, val_img_paths, val_mask_paths,
                    device, output_dir):
    try:
        num_images = len(val_img_paths)
        colour_map = []
        for i in range(50):
            hue = (i * 137.5) % 360
            h = hue / 60.0
            c = 1.0
            x = c * (1 - abs(h % 2 - 1))
            if h < 1:   r, g, b = c, x, 0
            elif h < 2: r, g, b = x, c, 0
            elif h < 3: r, g, b = 0, c, x
            elif h < 4: r, g, b = 0, x, c
            elif h < 5: r, g, b = x, 0, c
            else:       r, g, b = c, 0, x
            colour_map.append((int(r*255), int(g*255), int(b*255)))

        fig, axes = plt.subplots(3, num_images, figsize=(4*num_images, 12))
        if num_images == 1:
            axes = np.expand_dims(axes, axis=1)

        model.eval()
        for j, (img_path, mask_path) in enumerate(zip(val_img_paths, val_mask_paths)):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))

            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            gt_mask_bin = (gt_mask > 127).astype(np.uint8)

            axes[0, j].imshow(img)
            axes[0, j].set_title('Original', fontsize=10)
            axes[0, j].axis('off')

            gt_coloured = np.zeros_like(img, dtype=np.uint8)
            gt_contours, _ = cv2.findContours(gt_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for k, cnt in enumerate(gt_contours):
                if cv2.contourArea(cnt) > 50:
                    colour = colour_map[k % len(colour_map)]
                    cv2.drawContours(gt_coloured, [cnt], -1, colour, thickness=cv2.FILLED)
            axes[1, j].imshow(cv2.addWeighted(img, 0.5, gt_coloured, 0.5, 0))
            axes[1, j].set_title('Ground Truth (coloured lines)', fontsize=10)
            axes[1, j].axis('off')

            img_t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            bin_t = torch.from_numpy(binary).unsqueeze(0).float() / 255.0
            input_tensor = torch.cat([img_t, bin_t], dim=0).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits)
                pred_mask = (probs > 0.5).float().cpu().numpy()[0, 0]
            pred_mask = (pred_mask * 255).astype(np.uint8)
            pred_bin = (pred_mask > 127).astype(np.uint8)
            pred_contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            pred_coloured = np.zeros_like(img, dtype=np.uint8)
            for k, cnt in enumerate(pred_contours):
                if cv2.contourArea(cnt) > 50:
                    colour = colour_map[k % len(colour_map)]
                    cv2.drawContours(pred_coloured, [cnt], -1, colour, thickness=cv2.FILLED)
            axes[2, j].imshow(cv2.addWeighted(img, 0.5, pred_coloured, 0.5, 0))
            axes[2, j].set_title('ShirorekhaNet (PID+Cosine)', fontsize=10)
            axes[2, j].axis('off')

        plt.suptitle(f'Epoch {epoch+1} Validation Predictions', fontsize=14)
        plt.tight_layout()
        vis_path = os.path.join(output_dir, f'comparison_epoch_v5{epoch+1:03d}.png')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Visualisation saved to {vis_path}')
    except Exception as e:
        print(f'Visualization error (non‑critical): {e}')
