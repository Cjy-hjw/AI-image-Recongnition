"""反例:一个 [100,200] 的连通块在两个消费点的 shape 是否一致(裁剪框 vs 实例框)"""
import sys, numpy as np
sys.path.insert(0,'.')
from src.inference.nodule_split import crop_nodule
from src.inference.roi_pipeline import crop_roi_for_instance

frame=np.zeros((300,300),np.uint8); frame[50:150,60:160]=1
m=(frame>0).astype(np.uint8)
# 实例检测:2D 实例 mask 是全图尺寸的 m;官方 ROI 域 mask 应由「同样是该实例」的 mask 与图像同管线裁剪
x0,y0,x1,y1=60,50,160,150     # 半开
print("实例 bbox (xyxy)=",(x0,y0,x1,y1))
p_roi, cbox = crop_roi_for_instance(frame,(x0,y0,x1,y1))
print("ROI 图像 patch shape =",p_roi.shape,"实际裁剪框 =",cbox)
sub=m[y0:y1,x0:x1]; print("若按实例 bbox 裁 mask ->",sub.shape)
sub_c=m[cbox[1]:cbox[3],cbox[0]:cbox[2]]; print("若按裁剪框(含 pad)裁 mask ->",sub_c.shape)
print("=> 图像与 mask 是否同栅格(同裁剪框)?", p_roi.shape==sub_c.shape, "| 与实例框:",p_roi.shape==sub.shape)
