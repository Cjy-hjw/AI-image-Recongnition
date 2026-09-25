"""反例:2D 实例 mask(全图)在 ROI 域裁剪时,若拿 mask 当图像会得什么"""
import sys, numpy as np
sys.path.insert(0,'.')
from src.inference.nodule_split import crop_nodule, _split_2d
# 真实实例:512x512 图上 100x60 的椭圆状病灶
rng=np.random.default_rng(0)
binm=np.zeros((512,512),np.uint8)
binm[200:260,150:250]=1        # bbox=(150,200,100,60)
insts=_split_2d(binm,0.0001,20)
inst=insts[0]; x,y,w,h=inst["bbox"]
print("实例 bbox(xywh)=",inst["bbox"],"-> xyxy=",(x,y,x+w,y+h))
# 官方正确做法:mask 与图像走同一条裁剪管线(crop_nodule),则 mask_numel == H'*W'
mask_patch,_=crop_nodule(inst["mask"], inst["bbox"], pad_ratio=0.25)
print("mask 经 crop_nodule -> shape",mask_patch.shape,"前景素",int(mask_patch.sum()))
print("=> 与图像同栅格(150x150@512 图上 roi=150)?", mask_patch.shape)
