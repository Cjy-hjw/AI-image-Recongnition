import os
import csv
import json
import shutil
import cv2
import numpy as np

def extract_features_from_mask_and_image(img_path, mask_path):
    """Extract image-based TI-RADS pseudo-labels from ultrasound image and segmentation mask.

    Returns (shape_label, foci_label, foci_coords, echo_label, comp_label, margin_label).
    All labels are derived from physical image features — NOT correlated with malignancy.
    """
    img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    shape_label  = 0
    foci_label   = [0, 0, 0, 0]
    foci_coords  = []
    echo_label   = 1
    comp_label   = 3
    margin_label = 0

    if img is None or mask is None:
        return shape_label, foci_label, foci_coords, echo_label, comp_label, margin_label

    bin_mask = (mask > 127).astype(np.uint8)
    if bin_mask.sum() < 50:
        return shape_label, foci_label, foci_coords, echo_label, comp_label, margin_label

    ys, xs = np.where(bin_mask)

    # 1. SHAPE: bounding-box aspect ratio
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    shape_label = 1 if h > w else 0  # 1=taller-than-wide, 0=wider-than-tall

    # 2. FOCI: white top-hat on nodule ROI
    roi    = img * bin_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    tophat = cv2.subtract(roi, opened)
    bright_px = (tophat > 40) & (bin_mask == 1)
    if bright_px.sum() > 10:
        foci_label[3] = 1  # punctate echogenic foci (micro-calcification)
        for y, x in zip(*np.where(bright_px)):
            foci_coords.append([int(x), int(y)])

    # 3. ECHOGENICITY: nodule vs. surrounding parenchyma
    dilated  = cv2.dilate(bin_mask, np.ones((31, 31), np.uint8), iterations=1)
    surround = (dilated - bin_mask).astype(bool)
    mu_nod   = float(img[bin_mask == 1].mean())
    sur_vals = img[surround]
    if sur_vals.size == 0:
        echo_label = 2
    else:
        mu_par = float(sur_vals.mean())
        mu_mus = float(np.percentile(sur_vals, 20))  # dark band ≈ muscle
        if mu_nod < 15:
            echo_label = 0   # anechoic
        elif mu_nod >= mu_par:
            echo_label = 1   # hyperechoic / isoechoic
        elif mu_nod <= mu_mus:
            echo_label = 3   # very hypoechoic
        else:
            echo_label = 2   # hypoechoic

    # 4. COMPOSITION: cystic pixel fraction
    nod_pixels  = img[bin_mask == 1].astype(float)
    cystic_ratio = float((nod_pixels < 40).sum()) / nod_pixels.size
    if cystic_ratio > 0.90:
        comp_label = 0   # cystic
    elif cystic_ratio > 0.50:
        comp_label = 2   # mixed
    elif float(nod_pixels.std()) > 35:
        comp_label = 1   # spongiform (heterogeneous solid)
    else:
        comp_label = 3   # solid

    # 5. MARGIN: convexity defect ratio + border touch
    H, W = img.shape
    border_touch = (ys.min() < 5 or ys.max() > H - 5 or xs.min() < 5 or xs.max() > W - 5)
    if border_touch:
        margin_label = 3   # extrathyroidal extension
    else:
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt       = max(contours, key=cv2.contourArea)
            hull      = cv2.convexHull(cnt)
            cnt_area  = cv2.contourArea(cnt)
            hull_area = cv2.contourArea(hull)
            if cnt_area > 0:
                ratio = hull_area / cnt_area
                if ratio < 1.05:
                    margin_label = 0   # smooth
                elif ratio < 1.20:
                    margin_label = 1   # ill-defined
                else:
                    margin_label = 2   # lobulated / spiculated

    return shape_label, foci_label, foci_coords, echo_label, comp_label, margin_label

