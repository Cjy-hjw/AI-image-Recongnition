# -*- coding: utf-8 -*-
"""R-A 探针21:roi_meta 的 `pad_hw=(0,0)` 声称"键集对齐训练侧"。
训练侧 RoiDataset.__getitem__ 的 pad_hw 是**真实 pad 量**(右/下补),推理侧恒 (0,0)。
消费方若用 pad_hw 还原 ROI→原图坐标,推理侧会**静默错**(差 pad 量)。
检查:谁消费 pad_hw?全仓 grep。
"""
import subprocess, sys
out = subprocess.run([sys.executable, "-c", """
import subprocess, os, re
root = r'C:/Users/cjy/Desktop/AI image Recongnition'
hits = []
for d, _, fs in os.walk(root):
    if any(x in d for x in ('\\.git', 'worktrees', '__pycache__')): continue
    for f in fs:
        if not f.endswith(('.py',)): continue
        p = os.path.join(d, f)
        try: t = open(p, encoding='utf-8').read()
        except Exception: continue
        for i, l in enumerate(t.splitlines(), 1):
            if 'pad_hw' in l:
                hits.append((os.path.relpath(p, root), i, l.strip()[:110]))
for h in hits: print(h)
"""], capture_output=True, text=True)
print(out.stdout or out.stderr)
