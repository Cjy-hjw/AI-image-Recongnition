"""反例:ROI 裁剪的原始 bbox 框 vs 官方保底 holdout 框,loc 头输入差多少"""
import sys, numpy as np
sys.path.insert(0,'.')
from src.inference.roi_pipeline import crop_roi_for_instance, _roi_instance_bbox
# 官方风格实例(xywh),ROI 裁剪的 bbox 就是原始框
inst={"bbox":(100,80,30,20)}     # x=100,y=80,w=30,h=20
bbox=_roi_instance_bbox(inst)
frame=np.zeros((300,300),np.uint8)
patch, cropbox = crop_roi_for_instance(frame, bbox)
print("原始实例框(xyxy) =",bbox,"-> loc 头 bbox_xyxy")
print("crop_nodule 实际裁剪框 =",cropbox)
print("差异: 左",bbox[0]-cropbox[0],"上",bbox[1]-cropbox[1],"右",cropbox[2]-bbox[2],"下",cropbox[3]-bbox[3],"(像素)")
print("归一化差(orig 300): dx0=%.4f dy0=%.4f dx1=%.4f dy1=%.4f"%(
    (bbox[0]-cropbox[0])/300,(bbox[1]-cropbox[1])/300,(cropbox[2]-bbox[2])/300,(cropbox[3]-bbox[3])/300))
print("patch shape",patch.shape,"crop box size",(cropbox[2]-cropbox[0],cropbox[3]-cropbox[1]))
