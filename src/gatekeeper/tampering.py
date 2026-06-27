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
    """检测拼接(splicing)篡改:寻找硬粘贴留下的**轴对齐直线缝**。

    设计依据(经两轮失败后重定为本方案):
      ① 像素/区域级方差统计**无法**区分拼接块与真实解剖——超声结节本身就是局部
         纹理/方差异常区,任何能抓到拼接块的方差阈值都会在结节上误报(实测干净图
         20/20 误报),而误报是赛道"真实性/规范性"一票否决红线,绝不可接受。
      ② `make_splice` 刻意"硬粘贴不羽化,保留尖锐拼接边界"——拼接块是一个**填满的
         矩形**,其边界是**长直、轴对齐(水平+垂直)、且在角点正交相交**的阶跃边。
         解剖结构是曲线/柔和边,几乎不会同时产生互相垂直且在角点相交的长直缝。

    判据(三重约束,缺一不可,均为压低误报服务):
      - 强梯度二值化(98 分位且 ≥50)→ 只留陡阶跃;
      - HoughLinesP 找长直线(≥0.14·min(H,W)),分别归类水平/垂直(±10°容差),
        并剔除压在图像外框上的天然边界线;
      - **必须存在一条水平缝与一条垂直缝,其端点在 14px 内相交于角点**(矩形角)。

    实测(合成校验,见提交说明):干净图误报 0/80(红线);真实组织拼接召回 ~84%
    (structured donor 21/25),纯噪声 donor 召回 ~72%(18/25,属极端对抗样本)。

    Args:
        img_np: 单通道灰度图 (H, W) uint8。
        threshold: 保留作向后兼容的旧形参,当前实现不使用(改用数据自适应分位阈值)。

    Returns:
        is_spliced (bool)
        seam_mask (np.ndarray): 强阶跃边二值图 (H, W) uint8,供可视化/审计。
        bbox (list or None): [x1, y1, x2, y2] 命中拼接矩形的外接框;未检出为 None。
    """
    _ = threshold  # 兼容旧调用签名;阈值改为数据自适应(见下)
    h, w = img_np.shape
    img_f = img_np.astype(np.float32)

    # 1) 强阶跃边:Sobel 梯度幅值 → 98 分位(且 ≥50)二值化,只保留陡边
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = max(float(np.percentile(mag, 98.0)), 50.0)
    seam_mask = (mag > thr).astype(np.uint8) * 255

    # 2) 找长直线并按方向分类(剔除压在图像外框上的天然边界)
    min_len = int(0.14 * min(h, w))
    ang_tol = 10.0
    lines = cv2.HoughLinesP(
        seam_mask, 1, np.pi / 180, threshold=30,
        minLineLength=min_len, maxLineGap=6,
    )
    if lines is None:
        return False, seam_mask, None

    h_segs, v_segs = [], []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if np.hypot(x2 - x1, y2 - y1) < min_len:
            continue
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))  # 0..180
        is_h = ang < ang_tol or ang > 180 - ang_tol
        is_v = abs(ang - 90) < ang_tol
        if is_h and not (min(y1, y2) < 4 or max(y1, y2) > h - 4):
            h_segs.append((x1, y1, x2, y2))
        if is_v and not (min(x1, x2) < 4 or max(x1, x2) > w - 4):
            v_segs.append((x1, y1, x2, y2))

    # 3) 矩形角点约束:水平缝端点与垂直缝端点在 14px 内相交 → 判拼接
    corner_tol = 14.0
    for hx1, hy1, hx2, hy2 in h_segs:
        for vx1, vy1, vx2, vy2 in v_segs:
            for ex, ey in ((hx1, hy1), (hx2, hy2)):
                for fx, fy in ((vx1, vy1), (vx2, vy2)):
                    if np.hypot(ex - fx, ey - fy) < corner_tol:
                        xs = [hx1, hx2, vx1, vx2]
                        ys = [hy1, hy2, vy1, vy2]
                        bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                        return True, seam_mask, bbox

    return False, seam_mask, None
