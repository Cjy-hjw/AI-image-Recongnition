# Z1 推理侧 tiling 本地实现设计(新架构线第一波,零训练)

> plan `swift-whistling-pearl.md` 任务 1 Z1。基线 ep16(线上 69.51,静态匹配 0.8478/AP 0.8250)。
> **零训练**:现有 ep16 权重不动,只改推理前向。判死线:离线诊断 Δ匹配≥+0.012 且 ΔAP≥+0.008(vs ep16);零救回或 AP 负涨即死。

## 预检结论(2026-09-21 本地,零上云)

1. **压缩比空间存在但中等**:官方数据图幅跨度 6.5×(订正史:685),代理域 ThyroidXL 主流 727×532 → 长边压缩 30%、短边仅 4%。tiling 收益集中在**长边 ≥700 的图**(像素保留率 1.42×)与**小病灶**(面积随分辨率平方缩)。
2. **疑过分割信号(09-17 r5 探针,未定案)**:matched 成对 Σpred/Σgt ≈2.05(图幅假设下),若 ep16 确实过分割,tiling 增分辨率**可能加重过分割**——Z1 实测时须同时看 Dice 方向。
3. **挂钩点已核**(子 agent 探查):`case_pipeline.diagnose_static_case` 内 `preprocess_image`+`run_forward` 两行(:199-201);`_write_static` 的 resize/split 边界不碰。
4. 🔴 **TTA 覆辙禁令(plan 明文)**:禁止跨 tile **平均**概率。掩膜融合用逐像素 max;conf 在融合图(已 max)的实例区域取均值——不是多 tile conf 再平均。Gate2 2026-09-21 冻结:实现=融合图均值,不再做「面积最大来源 tile 的单 tile 软图」(那条预注册未落地,与 max 融合重复记账)。

## 设计

### 修改文件
- `src/inference/case_pipeline.py`:`diagnose_static_case` 增 tiling 分支(env 开关,默认关=逐位零回归)
- `src/inference/model_runner.py`:新增 `tile_image(img_np, tile, stride)` 几何函数
- 测试:`tests/test_tiling_inference.py` 新增

### 算法规格(预注册)
- **tile=512, stride=384**(重叠 128)。图短边≤512 且长边≤512 时不 tiling(恒等路径,逐位=现状)。
- 只对**静态**做 tiling;动态逐帧已按原帧分辨率走 512,不碰(动态池占比 4.7%,收益上限 +0.43 分,plan 已定动态不立项)。
- 前向:每 tile 独立 `preprocess_image`(恒等,tile 本身 512)→ `run_forward` → sigmoid 概率图。
- **掩膜融合**:概率域 max(不是平均——平均压缩分布;max 保尖峰,与 hysteresis 的 seed 语义兼容)。
- **置信度**:`detection_confidence` 消费融合后 `soft_fullres`(逐像素 max 的原图域概率),实例区域内取均值。单 tile 内实例与该 tile 前向逐位一致;跨 tile 实例反映峰值响应,禁止再对多 tile conf 做平均。
- 分类四头:取**融合图全局 argmax 像素所属 tile** 的 outputs(单一前向,零融合,位置头绝不吃混合特征)。所属=半开区间 `[oy,oy+tile)×[ox,ox+tile)`;重叠时取起点更近者。

### 置信度红线
禁止多 tile conf 平均。跨 tile 实例:融合图(max)上的实例均值,不是「面积最大来源 tile」单源(预注册未落地,2026-09-21 改跟实现)。

## 测试规格
1. 恒等:图 ≤512×512 时 tiling 路径输出与现行逐位一致(含 env 开与关两种)
2. 融合正确性:合成 727×532 图放已知位置病灶,tiling 后实例 bbox 坐标落在原图坐标系正确位置
3. conf 解耦:跨 tile 实例 conf 不等于两 tile conf 的平均(max 语义,非单 tile 源)
4. 零回归:env 关(默认)时 diagnose_static_case 全链路逐位不变(现有 26 测试为准)
5. 分类头单源:四头输出全部来自单一 tile(无融合痕迹)

## 执行顺序
1. 本地实现+测试(Gate1+Gate2:推理侧改动 commit 前触发)
2. 云端零上传诊断(diagnose_detection 走 workspace 树,sidecar v2,与 ep16 同源对拍)
3. 判死线裁决:Δ匹配/ΔAP vs ep16
