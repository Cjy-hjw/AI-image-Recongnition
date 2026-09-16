# -*- coding: utf-8 -*-
"""tests/ 共享 fixture。

阶段1(2026-09-13)新增:`split_nodules` 的 `max_nodules` 默认参改为
None→env 哨兵后,任何不显式传参的既有测试在本机/CI 残留
`AICOMP_MAX_NODULES` 时会假红(Gate2 审查实测 env=1 → 14 failed,反过来
也会掩盖真回归)。统一在此隔离;专测 env 路径的测试类
(TestMaxNodulesEnv/TestMinAreaEnv)用 monkeypatch.setenv 在测试体内自设,
autouse 删除发生在 setup 期、先于测试体,互不冲突。
"""
import pytest


@pytest.fixture(autouse=True)
def _isolate_split_envs(monkeypatch):
    """删掉可能影响 split_nodules 默认行为的 AICOMP_* env(含 min_area 同款哨兵)。"""
    monkeypatch.delenv("AICOMP_MAX_NODULES", raising=False)
    monkeypatch.delenv("AICOMP_MIN_AREA_RATIO", raising=False)
    monkeypatch.delenv("AICOMP_SEG_GROW", raising=False)
    monkeypatch.delenv("AICOMP_SEG_GROW_DYNAMIC", raising=False)
