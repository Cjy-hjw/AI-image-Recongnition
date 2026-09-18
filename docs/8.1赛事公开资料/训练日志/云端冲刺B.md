[load] sidecar 行=1042 跳过 v1 行=3 漏检 GT=241
==== unclassified 分层 ====
[static] 漏检=198 unclassified=96
  分层: 表征=0 (0.0%) / 阈值=80 (83.3%) / 形状=0 (0.0%) / 分散=16 (16.7%)
  主/次: {'main': 73, 'secondary': 23}
  soft_max 分布: {'0.5': 89, '0.2': 3, '0.35': 2, '0.1': 2, 'invalid': 0}
  soft_p95 分布: {'0.75': 78, '0.2': 6, '0.0': 7, '0.5': 2, '0.1': 3, 'invalid': 0}
  max_iou 分布: {'0.35': 43, '0.1': 37, '0.0': 16, 'invalid': 0}
  area_ratio 分布: {'0.005': 15, '0.01': 37, '0.001': 11, '0.05': 12, '0.002': 15, '0.0005': 4, '0.0': 2, 'invalid': 0}
  交叉(主桶×分层): {'mech1': {'repr': 0, 'threshold': 17, 'shape': 0, 'dispersed': 0}, 'mech2': {'repr': 0, 'threshold': 35, 'shape': 0, 'dispersed': 0}, 'mech3': {'repr': 0, 'threshold': 2, 'shape': 0, 'dispersed': 0}, 'mech5': {'repr': 48, 'threshold': 0, 'shape': 0, 'dispersed': 0}, 'unclassified': {'repr': 0, 'threshold': 80, 'shape': 0, 'dispersed': 16}}
[dynamic] 漏检=43 unclassified=28
  分层: 表征=0 (0.0%) / 阈值=27 (96.4%) / 形状=0 (0.0%) / 分散=1 (3.6%)
  主/次: {'main': 20, 'secondary': 8}
  soft_max 分布: {'0.5': 27, '0.1': 1, 'invalid': 0}
  soft_p95 分布: {'0.75': 27, '0.0': 1, 'invalid': 0}
  max_iou 分布: {'0.1': 18, '0.35': 9, '0.0': 1, 'invalid': 0}
  area_ratio 分布: {'0.01': 5, '0.002': 7, '0.05': 2, '0.001': 3, '0.005': 9, '0.0005': 2, 'invalid': 0}
  交叉(主桶×分层): {'mech1': {'repr': 0, 'threshold': 1, 'shape': 0, 'dispersed': 0}, 'mech2': {'repr': 0, 'threshold': 12, 'shape': 0, 'dispersed': 0}, 'mech5': {'repr': 2, 'threshold': 0, 'shape': 0, 'dispersed': 0}, 'unclassified': {'repr': 0, 'threshold': 27, 'shape': 0, 'dispersed': 1}}
[verdict] 预注册线: 表征线=False 阈值线=True 形状线=False 分散主导=False
[verdict] 预注册: 任一线过 40% 即定向;dispersed>=50% 维持 r5;其余组合按三段占比排序取最大者报人工裁决
[out] /2026aicompetition/workspace/arch_push/mech_layers.json