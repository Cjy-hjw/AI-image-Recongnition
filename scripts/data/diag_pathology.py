"""诊断batch1文件名不匹配原因。"""
import csv, os

ROOT = "data/27021604_extract/batch1_image"
img_dir = os.path.join(ROOT, "dataset")

# 实际目录文件名集合
actual_files = set(os.listdir(img_dir))
print(f"实际文件数: {len(actual_files)}")
print(f"样本: {sorted(actual_files)[:5]}")

# CSV里患者341的条目
with open(os.path.join(ROOT, "batch1_image.csv"), "r", encoding="utf-8-sig") as f:
    rows = list(csv.reader(f))[1:]

p341 = [(r[0].strip(), r[1].strip()) for r in rows if r[0].strip() == "341"]
print(f"\n患者341共{len(p341)}张:")
for p,f in p341:
    exists = f in actual_files
    print(f"  {f} -> {'OK' if exists else 'MISS'}")

# 统计:CSV里的fname存不存在
miss_files = set()
hit_files = set()
with open(os.path.join(ROOT, "batch1_image.csv"), "r", encoding="utf-8-sig") as f:
    rows = list(csv.reader(f))[1:]
for r in rows:
    fname = r[1].strip()
    if fname in actual_files:
        hit_files.add(fname)
    else:
        miss_files.add(fname)

print(f"\nCSV引用文件: hit={len(hit_files)} miss={len(miss_files)}")
print(f"实际有但CSV未引用: {len(actual_files - hit_files)}")
# 看几个miss的文件名规律
print(f"\nmiss示例: {sorted(miss_files)[:5]}")
# 看几个miss的patient对应的实际文件
for fname in sorted(miss_files)[:3]:
    pat = fname.split('_')[0]
    actual_for_pat = sorted(f for f in actual_files if f.startswith(pat+'_'))
    print(f"  miss={fname}; 实际有{len(actual_for_pat)}张: {actual_for_pat[:4]}")
