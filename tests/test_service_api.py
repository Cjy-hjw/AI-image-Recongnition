"""`src/service/` HTTP 推理服务回归测试(规范 §二/§三)。

**`TestHealthNeverBlocks` 是本文件的核心防线** —— 平台探活探针每 30 秒打一次
`/health`,连续失败 3 次即重启测评容器。torch 推理是同步阻塞代码,若跑在 event loop
里会堵死 `/health`,90 秒后整场评测被容器重启作废。该测试锁死"推理进行中 /health
仍然快速返回 200"这一不变量。
"""

import importlib
import os
import threading
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def svc(tmp_path, monkeypatch):
    """每个用例一套隔离的 answer 根目录与全新模块状态。

    必须 reload:`config` 在 import 时就从环境变量读定了路径常量,`tasks` 又在模块级
    建了线程池和任务表 —— 不 reload 会让用例间互相看到对方的任务与结果目录。
    """
    monkeypatch.setenv("AICOMP_ANSWER_ROOT", str(tmp_path / "answer"))
    # callback 默认基址已改为平台内网 IP(config.py);测试环境连不上会触发 3 次重试
    # 退避、阻塞单 worker,拖垮后续任务(test_service_survives_failed_task 曾因此超时)。
    # 显式设空串禁用回调——测试不该依赖真实网络;需要测回调的用例自行 setenv 覆盖。
    monkeypatch.setenv("AICOMP_CALLBACK_BASE_URL", "")

    from src.service import app as app_mod
    from src.service import config as config_mod
    from src.service import tasks as tasks_mod

    importlib.reload(config_mod)
    importlib.reload(tasks_mod)
    importlib.reload(app_mod)

    yield app_mod, config_mod, tasks_mod
    tasks_mod.shutdown(wait=True)


def _post(client, evaluation_id=1, request_id="req-1", dataset_path="/data/x"):
    body = {"input": {"evaluation_id": evaluation_id, "dataset_path": dataset_path}}
    if request_id is not None:
        body["request_id"] = request_id
    return client.post("/call", json=body)


def _wait_status(client, request_id, target, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/task/{request_id}")
        if r.status_code == 200 and r.json().get("status") == target:
            return r.json()
        time.sleep(0.02)
    raise AssertionError(
        f"任务 {request_id} 未在 {timeout}s 内达到 {target};"
        f"当前={client.get(f'/task/{request_id}').json()}")


class TestHealthNeverBlocks:
    """探活不能被推理阻塞 —— 否则容器被平台重启,整场评测作废。"""

    def test_health_responds_while_inference_blocks(self, svc):
        app_mod, _, _ = svc
        started = threading.Event()
        release = threading.Event()

        def blocking_runner(dataset_path, out_dir):
            started.set()
            release.wait(timeout=10)  # 模拟长时间 torch 推理

        app_mod.set_runner(blocking_runner)
        client = TestClient(app_mod.app)

        _post(client)
        assert started.wait(timeout=5), "任务未进入执行"

        # 推理正阻塞着,此刻探针必须仍能拿到 200 且足够快
        for _ in range(3):
            t0 = time.time()
            resp = client.get("/health")
            elapsed = time.time() - t0
            assert resp.status_code == 200
            assert elapsed < 1.0, f"/health 耗时 {elapsed:.2f}s,探针会超时"

        release.set()

    def test_health_reports_task_counts(self, svc):
        app_mod, _, _ = svc
        client = TestClient(app_mod.app)
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["tasks"] == {"total": 0, "by_status": {}}


class TestCallEndpoint:
    def test_returns_200_immediately(self, svc):
        """规范:接收请求后须在 5 秒内返回 200,超时判定请求异常。"""
        app_mod, _, _ = svc
        release = threading.Event()
        app_mod.set_runner(lambda ds, out: release.wait(timeout=10))
        client = TestClient(app_mod.app)

        t0 = time.time()
        resp = _post(client)
        elapsed = time.time() - t0

        assert resp.status_code == 200
        assert elapsed < 5.0, f"/call 耗时 {elapsed:.2f}s,超出规范 5 秒上限"
        assert resp.json()["request_id"] == "req-1"
        release.set()

    def test_creates_evaluation_dir_and_runs(self, svc):
        """规范 §二.2:{evaluation_id} 目录必须由参赛者代码在推理时动态创建。"""
        app_mod, config_mod, _ = svc
        app_mod.set_runner(
            lambda ds, out: open(os.path.join(out, "done.txt"), "w").close())
        client = TestClient(app_mod.app)

        _post(client, evaluation_id=999)
        _wait_status(client, "req-1", "DONE")

        assert os.path.exists(os.path.join(config_mod.answer_dir(999), "done.txt"))

    def test_dataset_path_forwarded(self, svc):
        app_mod, _, _ = svc
        seen = {}
        app_mod.set_runner(lambda ds, out: seen.update(ds=ds, out=out))
        client = TestClient(app_mod.app)

        _post(client, dataset_path="/mnt/testset/v1")
        _wait_status(client, "req-1", "DONE")
        assert seen["ds"] == "/mnt/testset/v1"


class TestPayloadTolerance:
    """规范未给完整 JSON Schema,且 5 秒内没返 200 即判异常、只有一次提交机会 ——
    入参解析必须尽可能宽容,不能因字段位置不同就 422 掉整个请求。"""

    def test_accepts_flat_payload(self, svc):
        app_mod, _, _ = svc
        app_mod.set_runner(lambda ds, out: None)
        client = TestClient(app_mod.app)
        resp = client.post("/call", json={
            "request_id": "r", "evaluation_id": 5, "dataset_path": "/d"})
        assert resp.status_code == 200

    def test_accepts_camel_case(self, svc):
        app_mod, _, _ = svc
        app_mod.set_runner(lambda ds, out: None)
        client = TestClient(app_mod.app)
        resp = client.post("/call", json={
            "requestId": "r", "input": {"evaluationId": 5, "datasetPath": "/d"}})
        assert resp.status_code == 200

    def test_missing_request_id_autogenerated(self, svc):
        """宁可用我方生成的 id 回调,也不要因缺字段 422 掉整个请求。"""
        app_mod, _, _ = svc
        app_mod.set_runner(lambda ds, out: None)
        client = TestClient(app_mod.app)
        resp = _post(client, request_id=None)
        assert resp.status_code == 200
        assert resp.json()["request_id"]

    def test_missing_evaluation_id_rejected(self, svc):
        """唯一的硬性拒绝:没有 evaluation_id 就无处写结果,也无法回调闭环。"""
        app_mod, _, _ = svc
        client = TestClient(app_mod.app)
        resp = client.post("/call", json={"request_id": "r"})
        assert resp.status_code == 400

    def test_invalid_json_gives_400_not_500(self, svc):
        app_mod, _, _ = svc
        client = TestClient(app_mod.app)
        resp = client.post(
            "/call", content=b"not json",
            headers={"Content-Type": "application/json"})
        assert resp.status_code == 400


class TestFailureIsolation:
    def test_runner_exception_writes_error_json(self, svc):
        """推理失败也要留下可诊断痕迹,而非空目录 —— 空目录与"还没跑"无法区分。"""
        app_mod, config_mod, _ = svc
        app_mod.set_runner(lambda ds, out: (_ for _ in ()).throw(RuntimeError("boom")))
        client = TestClient(app_mod.app)

        _post(client, evaluation_id=888)
        entry = _wait_status(client, "req-1", "FAILED")
        assert "RuntimeError: boom" in entry["error"]

        err_path = os.path.join(config_mod.answer_dir(888), "_error.json")
        assert os.path.exists(err_path)

    def test_service_survives_failed_task(self, svc):
        """一个任务崩溃不得影响后续任务 —— 线程池 worker 必须能继续接活。"""
        app_mod, _, _ = svc
        client = TestClient(app_mod.app)

        app_mod.set_runner(lambda ds, out: (_ for _ in ()).throw(ValueError("x")))
        _post(client, request_id="bad", evaluation_id=1)
        _wait_status(client, "bad", "FAILED")

        app_mod.set_runner(
            lambda ds, out: open(os.path.join(out, "ok.txt"), "w").close())
        _post(client, request_id="good", evaluation_id=2)
        _wait_status(client, "good", "DONE")
        assert client.get("/health").status_code == 200


class TestCallback:
    def test_skipped_when_base_url_unset(self, svc):
        """未配置回调基址时跳过并记日志,**不抛异常** —— 回调失败不该反过来毁掉
        已经算好的结果文件。"""
        _, _, tasks_mod = svc
        assert tasks_mod.post_callback("r", 1) is False

    def test_body_contains_three_required_fields(self, svc, monkeypatch):
        """规范:必须回传 request_id 与 evaluation_id,否则评测无法闭环、校验不通过。"""
        _, config_mod, tasks_mod = svc
        monkeypatch.setenv("AICOMP_CALLBACK_BASE_URL", "http://platform:8080")
        importlib.reload(config_mod)
        importlib.reload(tasks_mod)

        captured = {}

        class _Resp:
            status_code = 200

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["body"] = json
            return _Resp()

        import requests
        monkeypatch.setattr(requests, "post", fake_post)

        assert tasks_mod.post_callback("req-9", 4321) is True
        assert captured["url"] == "http://platform:8080/api/competition/inference/callback/"
        assert captured["body"] == {
            "request_id": "req-9",
            "evaluationId": 4321,
            "predPath": config_mod.pred_path(4321),
        }

    def test_callback_fires_even_on_failure(self, svc, monkeypatch):
        """失败时不回调会让平台一直等到评测超时,比回调一个错误结果更糟。"""
        app_mod, config_mod, tasks_mod = svc
        monkeypatch.setenv("AICOMP_CALLBACK_BASE_URL", "http://platform:8080")
        importlib.reload(config_mod)
        importlib.reload(tasks_mod)
        importlib.reload(app_mod)

        calls = []
        monkeypatch.setattr(
            tasks_mod, "post_callback",
            lambda rid, eid, pred_path=None: calls.append((rid, eid)) or True)

        app_mod.set_runner(lambda ds, out: (_ for _ in ()).throw(RuntimeError("x")))
        client = TestClient(app_mod.app)
        _post(client, request_id="rf", evaluation_id=77)
        _wait_status(client, "rf", "FAILED")

        assert ("rf", 77) in calls


class TestConfigPaths:
    def test_answer_dir_layout(self, svc):
        _, config_mod, _ = svc
        assert config_mod.answer_dir(123).endswith(os.path.join("answer", "123"))

    def test_pred_path_is_absolute_answer_dir(self, svc):
        """predPath 改为**绝对路径**(按官方推理服务代码示例 `/2026aicompetition/
        workspace/answer/{id}`);相对路径平台定位不到结果→零分。`AICOMP_PRED_PATH_PREFIX`
        旧开关已随之移除。`relative_pred_path` 保留为 `pred_path` 的向后兼容别名。"""
        _, config_mod, _ = svc
        expect = config_mod.answer_dir(9).replace(os.sep, "/")
        assert config_mod.pred_path(9) == expect
        assert config_mod.pred_path(9).endswith("answer/9")
        assert config_mod.relative_pred_path is config_mod.pred_path

    def test_port_is_8000(self, svc):
        """规范 §三:健康检查接口必须在 8000 端口。"""
        _, config_mod, _ = svc
        assert config_mod.SERVICE_PORT == 8000

    def test_single_inference_worker(self, svc):
        """单卡 32G 显存(官方答疑 Q33)并发推理会 OOM,且模型非线程安全。"""
        _, config_mod, _ = svc
        assert config_mod.INFERENCE_WORKERS == 1


class TestPlaceholderRunner:
    def test_placeholder_marks_output_clearly(self, svc, tmp_path):
        """骨架模式必须留下明确痕迹 —— 静默产出空目录会被误认为"推理成功但阴性"。"""
        app_mod, _, _ = svc
        out = tmp_path / "ph"
        app_mod._placeholder_runner("/data/x", str(out))
        assert (out / "_placeholder.txt").exists()
