#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""动态 .nii.gz 完整性诊断（云端专用，官方数据禁本地）。

## 为什么需要这个脚本

2026-08-14 云 agent 报某动态文件 nibabel 读 header 成功、shape 正确，但解压
报 `invalid block type`，并据「174MB 压缩率极低」推断文件损坏。

🔴 **该推断的基准是错的**：1618×1216×360 = 7.08e8 voxel，int16 未压缩约
1.35GB，压到 174MB 是 7.8:1，对超声视频完全正常。「正常应 5-10MB」是二值
**mask** 的量级（大片全零、压缩率 100:1+），两者本就差两个数量级。
→ 文件大小不能作为损坏证据，唯一实证只有 `invalid block type` 一条。

## 本脚本要区分的两种根因（决定修法完全不同）

- **A 文件真损坏**：gzip 流本身坏 → 加载时跳过这些样本即可。
- **B 读取被打断**：文件完好，是解码侧（内存压力/并发/nibabel ArrayProxy）
  失败 → 跳过样本无效，必须改 `official_loader` 的读取方式。
  ⚠️ B 有独立的先验证据：`official_loader._read_nifti_frame_gray` 对**每个
  样本**全量 `np.asarray(img.dataobj)` 一次（该模块 docstring 自述的已知雷），
  1.35GB × DataLoader worker 数，在 64G/14vCPU 上是明确的 OOM 路径。

**判据**：纯 gzip 层（不经 nibabel）能完整解压 → B；在某字节处失败 → A，
且失败偏移量直接给出损坏位置与截断与否。

## 输出

只打印原始数字，不下结论（结论归主会话）。逐文件一行 + 末尾汇总。
"""
from __future__ import annotations

import argparse
import gzip
import os
import struct
import sys
import zlib

# NIfTI-1 header 里我们需要的字段的字节偏移（小端/大端由 sizeof_hdr 判定）。
# 🔴 偏移量已与 nibabel 的 `Nifti1Header.template_dtype` 逐字段实测核对（2026-08-14 Gate2）。
_OFF_SIZEOF_HDR = 0
_OFF_VOX_OFFSET = 108  # float32：数据段起始偏移。**不是恒等于 352**——带 extension
                       # 的文件会更大。硬编码 352 会把 extension 的字节算成"多出来的"，
                       # 或把合法文件误判成 TRUNCATED（Gate2 实测发现）。
_OFF_DIM = 40          # short dim[8]，dim[0]=维数
_OFF_DATATYPE = 70     # short datatype
_OFF_BITPIX = 72       # short bitpix
_OFF_MAGIC = 344       # char magic[4]：'n+1\0'=单文件，'ni1\0'=对偶文件(数据在 .img)

_MIN_VOX_OFFSET = 352  # NIfTI-1 单文件的最小合法值（348 头 + 4 字节对齐）

# NIfTI datatype code → 人类可读名（只列超声可能出现的）
_DT_NAME = {
    2: "uint8", 4: "int16", 8: "int32", 16: "float32",
    64: "float64", 256: "int8", 512: "uint16", 768: "uint32",
}


def _parse_nifti_header(head: bytes):
    """从已解压的前 352 字节解析 (dims, datatype, bitpix, endian, vox_offset, magic)。

    不依赖 nibabel —— 本脚本的全部意义就是绕开它独立取证。
    偏移量已与 nibabel 的 header dtype 逐字段实测核对。
    """
    if len(head) < 348:
        return None
    # sizeof_hdr 必须是 348；若小端读出来不是，就试大端
    for endian, tag in ((("<"), "little"), ((">"), "big")):
        (sizeof_hdr,) = struct.unpack_from(endian + "i", head, _OFF_SIZEOF_HDR)
        if sizeof_hdr == 348:
            dim = struct.unpack_from(endian + "8h", head, _OFF_DIM)
            (datatype,) = struct.unpack_from(endian + "h", head, _OFF_DATATYPE)
            (bitpix,) = struct.unpack_from(endian + "h", head, _OFF_BITPIX)
            # dim[0] <= 0 是非法/未设置：按 1 维处理（与 nibabel 同口径），
            # 不要静默降级成"无 dims"——那会让诊断输出丢掉 shape，无法区分
            # "合法的只有头的文件"和"被截断只剩 352B"（Gate2 实测发现）。
            ndim = dim[0]
            ndim = 1 if ndim <= 0 else min(7, ndim)
            vox_offset = _MIN_VOX_OFFSET
            magic = b""
            if len(head) >= 352:
                (vo,) = struct.unpack_from(endian + "f", head, _OFF_VOX_OFFSET)
                # vox_offset=0 是"未设置"的常见写法，按最小合法值兜底
                vox_offset = int(vo) if vo >= _MIN_VOX_OFFSET else _MIN_VOX_OFFSET
                magic = bytes(head[_OFF_MAGIC:_OFF_MAGIC + 4]).rstrip(b"\x00")
            return (list(dim[1:1 + ndim]), datatype, bitpix, tag, vox_offset, magic)
    return None


def _probe_gzip(path: str, chunk: int = 8 << 20):
    """流式解压整个 .gz，返回 (ok, n_bytes_decompressed, err_str, head_352)。

    关键：**不 seek、不随机访问、不经 nibabel**，纯顺序读。这样若失败，
    失败原因只可能是 gzip 流本身，排除掉一切上层读取方式的干扰。
    内存不随文件大小增长（逐块丢弃）—— 实测 180MB 文件峰值 RSS 约 76MB，
    343 个文件串行扫在 64G 机器上无风险。

    ⚠️ `n_bytes_decompressed` 的语义（Gate2 实测澄清，别误读）：它是**成功解压
    出的字节数**，不总等于"损坏发生的位置"。gzip 在成员末尾才校验 CRC，故：
      - 大文件（>1 个 chunk）：损坏点前的块已累计，**有定位价值**
        （实测 54MB 压缩流在 90% 处损坏 → n_decomp=58,720,256）
      - 小文件（< 1 chunk）：一次解完才发现错，n_decomp 恒为 0
    官方动态文件是 1.35GB 量级，属前者。
    """
    total = 0
    head = b""
    try:
        with gzip.open(path, "rb") as f:
            while True:
                buf = f.read(chunk)
                if not buf:
                    break
                if len(head) < 352:
                    head += buf[: 352 - len(head)]
                total += len(buf)
        return True, total, None, head
    except (OSError, EOFError, zlib.error) as e:
        # gzip 把 zlib.error 包成 BadGzipFile(OSError)；两者都接住。
        # 🔴 特例：**完好的 gzip 成员后追加了非 gzip 字节**时，gzip 读完第一个
        # 成员会尝试解析下一个成员并抛 "Not a gzipped file"，但文件主体其实完好。
        # 直接判 FAIL 会把好文件误报成损坏 → "跳过其实能用的样本"（Gate2 实测）。
        # 故回退验证：只解第一个成员，成功即算好。
        if total == 0 and "Not a gzipped file" in str(e):
            ok2, n2, head2 = _probe_first_member(path, chunk)
            if ok2:
                return True, n2, None, head2
        return False, total, f"{type(e).__name__}: {e}", head


def _probe_first_member(path: str, chunk: int):
    """只解压**第一个** gzip 成员，忽略其后的任何追加字节。

    `zlib.decompressobj(16 + MAX_WBITS)`（16 = 期待 gzip 头）在成员结束时置
    `eof=True`，剩余字节进 `unused_data` —— 正是要忽略的部分。
    """
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    total, head = 0, b""
    try:
        with open(path, "rb") as f:
            while True:
                raw = f.read(chunk)
                if not raw:
                    break
                out = d.decompress(raw)
                if len(head) < 352:
                    head += out[: 352 - len(head)]
                total += len(out)
                if d.eof:            # 第一个成员正常结束，其后字节一律不管
                    return True, total, head
        return False, total, head   # 读完都没等到成员结束 = 真截断
    except (OSError, zlib.error):
        return False, total, head


def diagnose(path: str) -> dict:
    r = {"path": path, "file_size": -1, "ok": False, "n_decomp": 0,
         "err": None, "dims": None, "dtype": None, "expect": 0, "ratio": 0.0,
         "vox_offset": 0, "magic": ""}
    try:
        r["file_size"] = os.path.getsize(path)
    except OSError as e:
        r["err"] = f"stat failed: {e}"
        return r

    ok, n, err, head = _probe_gzip(path)
    r["ok"], r["n_decomp"], r["err"] = ok, n, err

    parsed = _parse_nifti_header(head)
    if parsed:
        dims, dt, bitpix, endian, vox_offset, magic = parsed
        nvox = 1
        for d in dims:
            nvox *= max(1, d)
        # 期望总字节 = 数据段起始偏移 + 数据。
        # 🔴 用 header 真实的 `vox_offset` 而非硬编码 352 —— 带 extension 的文件
        # vox_offset > 352，硬编码会把合法文件误判成 TRUNCATED（Gate2 实测）。
        expect = vox_offset + nvox * (bitpix // 8 if bitpix else 0)
        r["dims"] = dims
        r["dtype"] = f"{_DT_NAME.get(dt, dt)}/{bitpix}bit/{endian}"
        r["expect"] = expect
        r["vox_offset"] = vox_offset
        r["magic"] = magic.decode("ascii", "replace") if magic else ""
        if r["file_size"] > 0:
            r["ratio"] = expect / r["file_size"]
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="动态 .nii.gz 完整性诊断（纯 gzip 层，绕开 nibabel）")
    ap.add_argument("--root", default="/2026aicompetition/datasets/training/annotation",
                    help="官方 annotation 一级目录")
    ap.add_argument("--limit", type=int, default=0, help="只查前 N 个（0=全部）")
    ap.add_argument("--masks", action="store_true",
                    help="同时检查 *_mask.nii.gz（默认只查动态影像 {pid}/{pid}/{pid}.nii.gz）")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"FATAL: root 不存在: {args.root}")
        return 2

    targets = []
    # 单个目录的权限/IO 异常不得终止整场扫描（云端 NFS/ACL 下 listdir 才会抛，
    # isdir 先返回 True 骗过检查）—— 打印 warning 继续，别让 342 个好文件陪葬。
    try:
        pids = sorted(os.listdir(args.root))
    except OSError as e:
        print(f"FATAL: 无法列出 root: {e}")
        return 2
    for pid in pids:
        sub = os.path.join(args.root, pid, pid)
        if not os.path.isdir(sub):
            continue  # 静态病例（二级目录是 SeriesUID）
        dyn = os.path.join(sub, f"{pid}.nii.gz")
        if os.path.isfile(dyn):
            targets.append(dyn)
        if args.masks:
            try:
                for fn in sorted(os.listdir(sub)):
                    if fn.endswith("_mask.nii.gz"):
                        targets.append(os.path.join(sub, fn))
            except OSError as e:
                print(f"  [warn] 跳过不可读目录 {sub}: {e}", flush=True)

    if args.limit > 0:
        targets = targets[: args.limit]

    print(f"=== 待检 {len(targets)} 个文件 (root={args.root}) ===", flush=True)
    bad, trunc_list, good, total_bytes = [], [], 0, 0
    for i, p in enumerate(targets, 1):
        r = diagnose(p)
        total_bytes += max(0, r["file_size"])
        name = os.path.basename(p)
        # magic='ni1' = 对偶文件（数据在同名 .img 里），本文件只有头，
        # 此时拿 expect 判截断没有意义，单独标注而不误报 TRUNCATED。
        dual = (r["magic"] == "ni1")
        vo = f" vox_offset={r['vox_offset']}" if r["vox_offset"] not in (0, 352) else ""
        if r["ok"]:
            good += 1
            flag = ""
            if dual:
                flag = "  [ni1对偶文件:数据在.img,不判截断]"
            elif r["expect"] and r["n_decomp"] < r["expect"]:
                flag = f"  🔴TRUNCATED short_by={r['expect'] - r['n_decomp']}"
                trunc_list.append(r)
            elif r["expect"] and r["n_decomp"] > r["expect"]:
                # 多出来的通常是 extension；vox_offset 已计入，仍多说明有额外段
                flag = f"  [注]decomp 比 expect 多 {r['n_decomp'] - r['expect']} 字节"
            print(f"[{i}/{len(targets)}] OK   {name}  size={r['file_size']}"
                  f" decomp={r['n_decomp']} dims={r['dims']} dt={r['dtype']}"
                  f"{vo} ratio={r['ratio']:.1f}:1{flag}", flush=True)
        else:
            bad.append(r)
            print(f"[{i}/{len(targets)}] FAIL {name}  size={r['file_size']}"
                  f" decomp_ok={r['n_decomp']} dims={r['dims']}"
                  f" dt={r['dtype']} expect={r['expect']}{vo}  ERR={r['err']}", flush=True)

    print("\n=== 汇总 ===")
    print(f"总文件数        : {len(targets)}")
    print(f"完整解压成功    : {good}")
    print(f"解压失败        : {len(bad)}")
    print(f"完整但内容截断  : {len(trunc_list)}   <- 第三种情况,A/B 方案都不覆盖")
    print(f"总字节          : {total_bytes}")
    if bad:
        print("\n失败清单（路径 / 成功解压字节数 / 错误）:")
        print("  ⚠️ 「成功解压字节数」对**大文件**才有定位价值(损坏点前的块已累计);")
        print("     小文件(<8MB)恒为 0,那不代表损坏在 0 字节处。")
        for r in bad:
            print(f"  {r['path']}  decomp_ok={r['n_decomp']}  {r['err']}")
    if trunc_list:
        print("\n截断清单（gzip 流完好但数据不足 header 声明的量）:")
        for r in trunc_list:
            print(f"  {r['path']}  decomp={r['n_decomp']} expect={r['expect']} "
                  f"short_by={r['expect'] - r['n_decomp']} dims={r['dims']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
