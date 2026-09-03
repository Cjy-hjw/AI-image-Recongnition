import cv2
import os
import numpy as np
import threading
import torch
import torchvision.models as models
import torchvision.transforms as transforms

def compute_dhash(img_np, hash_size=8):
    """
    Computes difference hash (dHash) for fast image similarity screening.
    Resizes image to (hash_size + 1) x hash_size, then compares horizontal gradients.
    """
    if img_np.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img_np.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img_np = cv2.cvtColor(img_np, code)
    resized = cv2.resize(img_np, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    # Compare left and right pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # Convert boolean array to a 64-bit integer hash
    decimal_value = 0
    hash_string = ""
    for idx, value in enumerate(diff.flatten()):
        if value:
            decimal_value += 2**(idx % 8)
        if (idx % 8) == 7:
            hash_string += hex(decimal_value)[2:].rjust(2, '0')
            decimal_value = 0
    return hash_string

def hamming_distance(hash1, hash2):
    """Computes Hamming distance between two hex hashes."""
    # Convert hex to binary strings
    bin1 = bin(int(hash1, 16))[2:].zfill(64)
    bin2 = bin(int(hash2, 16))[2:].zfill(64)
    return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))

# SSCD 离线 TorchScript 权重的候选路径(任一存在即启用 SSCD 后端)。
# 来源:facebookresearch/sscd-copy-detection,Apache-2.0/MIT,95MB,512 维 GeM。
# 比赛交付为商用离线容器,SSCD 协议合规;权重需赛前手动下载放入 checkpoints/。
_SSCD_WEIGHT_CANDIDATES = (
    os.path.join("checkpoints", "sscd_disc_mixup.torchscript.pt"),
    os.path.expanduser("~/.cache/torch/hub/checkpoints/sscd_disc_mixup.torchscript.pt"),
)


def _find_sscd_weights(explicit_path=None):
    """返回首个存在的 SSCD 权重路径,均不存在返回 None(触发 ResNet18 降级)。"""
    candidates = ([explicit_path] if explicit_path else []) + list(_SSCD_WEIGHT_CANDIDATES)
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


class FeatureExtractor:
    """可插拔深度描述符提取器。

    优先 SSCD(近重复检测 SOTA,对 ≤80% 局部裁切/水印/色调偏移近似不变);
    SSCD TorchScript 权重缺失时**降级** torchvision ResNet18(行为与历史一致,
    保证离线无权重环境下测试与推理不破)。两后端均输出 512 维 L2 归一向量,
    与 FAISS `IndexFlatIP(512)` 及余弦判据完全兼容。
    """

    # 各后端推荐的余弦近重复阈值(Deduplicator cosine_threshold=None 时自适应取用)。
    # SSCD 0.86:官方 README 给 0.75(90% 精度),本项目取更严的 0.86 以贴合医保
    #   一票否决红线「宁可漏检不可误杀真实影像」(阈值越高→误报越少)。
    # ResNet18 0.95:类默认;生产 config 另以 0.995 进一步调严(见 thyroid_config.yaml)。
    _RECOMMENDED_COSINE = {"sscd": 0.86, "resnet18": 0.95}

    def __init__(self, device='cpu', sscd_weights=None):
        self.device = device
        sscd_path = _find_sscd_weights(sscd_weights)

        if sscd_path is not None:
            # SSCD: 自带 GeM 池化,TorchScript 直接 jit.load,零外部框架依赖
            self.backend_name = "sscd"
            self.model = torch.jit.load(sscd_path, map_location=device)
            self.model.eval()
            resize = 288  # SSCD disc 官方推荐输入边长
        else:
            # 降级:torchvision ResNet18(ImageNet),去分类头取 512 维 pooled 特征。
            # 权重查找顺序:① 环境变量 AICOMP_RESNET18_WEIGHTS(云端持久盘部署用,
            # 2026-08-31 补:平台容器重建会清容器层,~/.cache 下的拷贝/软链接都会
            # 丢 —— 让权重直接从 workspace 挂载盘读,免疫重建);② 官方 torchvision
            # 缓存位(本地开发/CI 默认);③ 项目 checkpoints/(与 SSCD 候选位一致)。
            # 三处都找不到才走下面的联网下载分支。
            self.backend_name = "resnet18"
            self.model = models.resnet18(pretrained=False)
            _resnet18_candidates = [
                os.environ.get("AICOMP_RESNET18_WEIGHTS"),
                os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"),
                os.path.join("checkpoints", "resnet18-f37072fd.pth"),
            ]
            local_weights = next(
                (p for p in _resnet18_candidates if p and os.path.exists(p)), None
            )
            if local_weights:
                # weights_only=True:纯张量 state_dict,torch≥2.6 下安全加载(拒绝
                # 任意代码执行)且消除 FutureWarning;旧版 torch 同样支持该参数。
                self.model.load_state_dict(
                    torch.load(local_weights, map_location='cpu', weights_only=True))
            else:
                # 2026-07-11 修复(v1→v2,v1 曾直接 raise 挡掉整个联网尝试,结果把 CI/开发
                # 机等本身有网络的环境也一并堵死——这些环境本不需要预置本地缓存,联网下载
                # 本来就能成功,是本项目自己的回归,已用真实 CI 跑红复现锁定,详见
                # git log 里 v1 commit 的教训)。
                # 真正要防的不是"这次没有本地缓存"，而是"离线交付容器场景(答疑Q26禁联网)
                # 下联网尝试失败/挂起"——两者的正确处理不同:CI/开发机应该让 pretrained=True
                # 照常联网下载成功;只有当联网这一步本身失败(DNS解析失败/连接被拒/超时等)
                # 才需要转成清晰的 RuntimeError,而不是任由原始网络异常类型裸传播,或者
                # (更糟)完全不给联网尝试的机会就一刀切拒绝。
                try:
                    self.model = models.resnet18(pretrained=True)
                except Exception as e:
                    raise RuntimeError(
                        "[FATAL] Deduplicator resnet18 降级后端权重缺失且联网下载失败: "
                        f"{local_weights} 不存在,尝试 torchvision pretrained=True 联网"
                        f"下载时报错({type(e).__name__}: {e})。若在离线交付容器场景"
                        "(答疑Q26禁联网)下触发,说明打包流程漏了预置"
                        "resnet18-f37072fd.pth——去重是 gatekeeper 一票否决链路的一环,"
                        "不能在权重链路完全断裂且联网也失败时继续静默运行。"
                        "请确认打包流程已把权重放入 ~/.cache/torch/hub/checkpoints/,"
                        "或修复 SSCD 主后端权重缺失(sscd_weights 缺失是本次降级到 "
                        "resnet18 的根因)。"
                    ) from e
            self.model.fc = torch.nn.Identity()
            self.model.to(device)
            self.model.eval()
            resize = 224

        self.recommended_cosine_threshold = self._RECOMMENDED_COSINE[self.backend_name]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        # Fail-fast 输出维度探测(reviewer 必修2):跑一次 dummy 前向实测描述符维度。
        # 防止误传非 512 维 SSCD 权重(如 disc_large=1024)时,FAISS 晚抛错或
        # NumPy 回退路径静默算出错误相似度。Deduplicator 据此动态建索引,不再硬编码。
        self.feat_dim = int(self.extract(np.zeros((64, 64), dtype=np.uint8)).shape[0])

    def extract(self, img_np):
        # 灰度超声补成 RGB(两后端均吃 3 通道 ImageNet 归一化输入)
        if len(img_np.shape) == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_np

        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = feat.squeeze(0).cpu().numpy().reshape(-1)

        # L2 归一化 → 内积即余弦相似度
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

class Deduplicator:
    """
    Main deduplication module.
    Performs fast dHash screening first, then verifies using cosine similarity of deep embeddings.
    特征后端可插拔(SSCD 优先,降级 ResNet18);均为 512 维,支持 FAISS / NumPy 双路。

    线程安全(2026-09-03):`_lock` 串行化 `add_image`/`is_duplicate` 对共享库
    (hashes / embeddings / FAISS index)的读写。背景:推理服务 worker 1→2
    (config.INFERENCE_WORKERS + runner 病例级并行)后,两个线程共享同一
    Deduplicator 实例(runner 闭包 state["dedup"] 跨病例复用),无锁时两方法并发
    会竞态——`add_image` 是「append list + index.add」两步非原子,is_duplicate
    并发读时 NumPy 回退路径 `np.array(self.embeddings)` 可能读到与 FAISS index
    不同步的撕裂状态,search 结果不可预测。**锁只圈共享结构的读写**:dHash 与
    深度特征提取(~0.3-0.5s torch 前向)在锁外——只读输入不碰共享状态,圈进
    等于把两线程的审计全程序串行化、提速归零。宁可串行写库、不可并发互踩
    (「宁可漏检不可误杀」红线在去重语义上同样成立——误判重复在 hard 模式
    会硬拒真图)。
    """
    def __init__(self, device='cpu', phash_threshold=5, cosine_threshold=None,
                 sscd_weights=None):
        self.phash_threshold = phash_threshold
        self._lock = threading.Lock()
        self.extractor = FeatureExtractor(device=device, sscd_weights=sscd_weights)
        # 阈值解析(按激活后端选值,守红线):
        #   None         → 取后端推荐(SSCD 0.86 / ResNet18 0.95);
        #   dict         → 按 backend_name 取值(如 {sscd:0.86, resnet18:0.995}),
        #                  缺失键回退后端推荐。单一标量无法同时服务两后端——SSCD 与
        #                  ResNet18 相似度分布不同(SSCD 治近重复漏检需 0.86,ResNet18
        #                  守一票否决红线「宁可漏检不可误杀」需 0.995),故支持映射;
        #   标量(float)  → 尊重调用方,两后端同值(向后兼容旧 config 硬编码阈值)。
        backend = self.extractor.backend_name
        if cosine_threshold is None:
            self.cosine_threshold = self.extractor.recommended_cosine_threshold
        elif isinstance(cosine_threshold, dict):
            self.cosine_threshold = float(cosine_threshold.get(
                backend, self.extractor.recommended_cosine_threshold))
        else:
            self.cosine_threshold = float(cosine_threshold)
        
        # In-memory database of hashes and embeddings
        self.hashes = []      # list of dHash strings
        self.embeddings = []  # list of feature vectors
        
        # Try to import FAISS
        try:
            import faiss
            self.use_faiss = True
            # 用后端实测描述符维度(SSCD/ResNet18 均 512;防误传 1024 维权重静默错算)
            self.index = faiss.IndexFlatIP(self.extractor.feat_dim)
        except ImportError:
            self.use_faiss = False
            
    def add_image(self, img_np):
        dhash = compute_dhash(img_np)
        feat = self.extractor.extract(img_np)

        # 锁内完成「append + index.add」两步,保证 embeddings 与 FAISS index
        # 始终同步(见类 docstring 的线程安全说明)。
        with self._lock:
            self.hashes.append(dhash)
            self.embeddings.append(feat)

            if self.use_faiss:
                # FAISS IndexFlatIP expects float32 array
                self.index.add(np.expand_dims(feat, axis=0).astype(np.float32))

    def is_duplicate(self, img_np):
        # 线程安全:重活(dHash + 深度特征提取,~0.3-0.5s 的 torch 前向)在**锁外**
        # 算——只读输入不触碰共享状态;锁只圈住对 hashes/embeddings/index 的读。
        # 若把 extract 圈进锁,worker=2 的两病例审计会全程序串行化,提速归零
        # (2026-09-03 审查 P1-2)。add_image 同款策略:extract 锁外、写库锁内。
        query_hash = compute_dhash(img_np)
        query_feat = self.extractor.extract(img_np)
        with self._lock:
            if not self.hashes:
                return False, None

            # 1. Fast dHash screen
            for idx, stored_hash in enumerate(self.hashes):
                dist = hamming_distance(query_hash, stored_hash)
                if dist <= self.phash_threshold:
                    return True, f"dHash match (distance: {dist})"

            # 2. ResNet feature screening
            if self.use_faiss:
                # Query FAISS
                distances, indices = self.index.search(
                    np.expand_dims(query_feat, axis=0).astype(np.float32), 1)
                best_sim = distances[0][0]
                best_idx = indices[0][0]
                if best_sim >= self.cosine_threshold and best_idx != -1:
                    return True, f"Deep feature match (Cosine Similarity: {best_sim:.4f})"
            else:
                # NumPy fallback
                embeddings_matrix = np.array(self.embeddings)
                similarities = np.dot(embeddings_matrix, query_feat)
                best_idx = np.argmax(similarities)
                best_sim = similarities[best_idx]
                if best_sim >= self.cosine_threshold:
                    return True, f"Deep feature match (Cosine Similarity: {best_sim:.4f})"

            return False, None
