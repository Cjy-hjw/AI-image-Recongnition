import cv2
import numpy as np

def detect_copy_move(img_np, block_size=16, stride=16, similarity_threshold=15.0, min_dist=30):
    """
    Classical block-matching copy-move detection.
    Divides the image into overlapping blocks, extracts simple block features,
    and finds matching block pairs with large spatial separation.
    
    Returns:
        is_tampered (bool)
        tamper_mask (np.ndarray): Binary mask highlighting tampered areas.
        bbox (list or None): [x1, y1, x2, y2] bounding box of detected tampering.
    """
    h, w = img_np.shape
    blocks = []
    positions = []
    
    # 1. Slide window to extract blocks and features
    for y in range(0, h - block_size + 1, stride):
        for x in range(0, w - block_size + 1, stride):
            block = img_np[y:y+block_size, x:x+block_size]
            
            # Simple features: mean, std, and a 4x4 downscaled version of the block
            mean = np.mean(block)
            std = np.std(block)
            
            # Filter out uniform background blocks (e.g. flat black scanning background)
            if std < 5.0:
                continue
                
            downscaled = cv2.resize(block, (4, 4), interpolation=cv2.INTER_AREA).flatten()
            
            # Concatenate features
            feat = np.concatenate([[mean, std], downscaled])
            
            blocks.append(feat)
            positions.append((x, y))
            
    blocks = np.array(blocks, dtype=np.float32)
    positions = np.array(positions)
    
    # Normalize features for distance comparison
    feat_mean = np.mean(blocks, axis=0)
    feat_std = np.std(blocks, axis=0) + 1e-6
    normalized_blocks = (blocks - feat_mean) / feat_std
    
    # 2. Match blocks
    num_blocks = len(normalized_blocks)
    match_mask = np.zeros_like(img_np, dtype=np.uint8)
    matched_pairs = 0
    
    # Lexicographical sort or simple nearest neighbor check
    # For small image size (e.g. 256x256), a nested loop with early stopping is fast
    # We sort by the first few feature dimensions to group similar blocks
    sorted_indices = np.lexsort((normalized_blocks[:, 1], normalized_blocks[:, 0]))
    
    # Search window for matching sorted blocks
    search_window = 20
    
    for i in range(num_blocks):
        idx_i = sorted_indices[i]
        pos_i = positions[idx_i]
        
        for j in range(max(0, i - search_window), min(num_blocks, i + search_window)):
            if i == j:
                continue
            idx_j = sorted_indices[j]
            pos_j = positions[idx_j]
            
            # Spatial distance
            dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
            if dist < min_dist:
                continue
                
            # Feature distance (L2 norm)
            feat_dist = np.linalg.norm(normalized_blocks[idx_i] - normalized_blocks[idx_j])
            if feat_dist < similarity_threshold:
                # Highlight matching blocks on mask
                cv2.rectangle(match_mask, pos_i, (pos_i[0] + block_size, pos_i[1] + block_size), 255, -1)
                cv2.rectangle(match_mask, pos_j, (pos_j[0] + block_size, pos_j[1] + block_size), 255, -1)
                matched_pairs += 1
                
    # 3. Post-process and extract bounding box
    is_tampered = False
    bbox = None
    
    # Filter matches: noise will produce isolated random matches;
    # actual tampering will have clustered overlapping matches.
    if matched_pairs > 4:
        # Find contours of matched regions
        contours, _ = cv2.findContours(match_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > (block_size * block_size)]
        
        if len(valid_contours) >= 2:
            is_tampered = True
            # Find the bounding box of the largest contour (typically the pasted/tampered region)
            areas = [cv2.contourArea(c) for c in valid_contours]
            target_contour = valid_contours[np.argmax(areas)]
            tx, ty, tw, th = cv2.boundingRect(target_contour)
            bbox = [tx, ty, tx + tw, ty + th]
            
    return is_tampered, match_mask, bbox

def detect_splicing(img_np, threshold=40):
    """
    Detects splicing tampering by checking for unnatural edges and local variance inconsistencies.
    Ultrasound images have smooth speckle gradients; spliced patches introduce artificial sharp boundaries.
    """
    # 1. Edge detection using Sobel operator
    sobelx = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # 2. Local variance inconsistency check
    # Spliced regions often come from different scans with distinct noise levels/gains
    kernel = np.ones((9, 9), np.float32) / 81.0
    local_mean = cv2.filter2D(img_np.astype(float), -1, kernel)
    local_sq_mean = cv2.filter2D(img_np.astype(float)**2, -1, kernel)
    local_variance = np.clip(local_sq_mean - local_mean**2, 0, None)
    
    # Calculate gradients of variance map to spot sharp noise transition boundaries
    var_sobelx = cv2.Sobel(local_variance, cv2.CV_64F, 1, 0, ksize=3)
    var_sobely = cv2.Sobel(local_variance, cv2.CV_64F, 0, 1, ksize=3)
    var_edge = np.sqrt(var_sobelx**2 + var_sobely**2)
    
    # Detect high anomaly regions using a robust self-calibrating threshold
    # local variance maps can have very large gradients (0-16000), so 40 is too low.
    dynamic_threshold = max(800.0, np.mean(var_edge) + 4.5 * np.std(var_edge))
    anomaly_map = (var_edge > dynamic_threshold).astype(np.uint8) * 255
    
    # Clean up small noise points
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    anomaly_map = cv2.morphologyEx(anomaly_map, cv2.MORPH_OPEN, kernel_morph)
    
    contours, _ = cv2.findContours(anomaly_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    is_spliced = False
    bbox = None
    
    # If we find a significant anomalous border, flag it
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
    if len(valid_contours) > 0:
        is_spliced = True
        # Bounding box of the largest anomaly
        areas = [cv2.contourArea(c) for c in valid_contours]
        target_contour = valid_contours[np.argmax(areas)]
        tx, ty, tw, th = cv2.boundingRect(target_contour)
        bbox = [tx, ty, tx + tw, ty + th]
        
    return is_spliced, anomaly_map, bbox
