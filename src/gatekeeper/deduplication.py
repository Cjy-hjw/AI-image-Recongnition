import cv2
import os
import numpy as np
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
            # 降级:torchvision ResNet18(ImageNet),去分类头取 512 维 pooled 特征
            self.backend_name = "resnet18"
            self.model = models.resnet18(pretrained=False)
            local_weights = os.path.expanduser(
                "~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
            )
            if os.path.exists(local_weights):
                # weights_only=True:纯张量 state_dict,torch≥2.6 下安全加载(拒绝
                # 任意代码执行)且消除 FutureWarning;旧版 torch 同样支持该参数。
                self.model.load_state_dict(
                    torch.load(local_weights, map_location='cpu', weights_only=True))
            else:
                # 2026-07-11 修复:此前 else 分支是 `models.resnet18(pretrained=True)`,
                # 本地缓存缺失时会触发 torchvision 联网下载官方 ImageNet 权重——离线
                # 交付容器场景(答疑Q26「不能联网」)下这条路径会直接挂死/超时,而不是
                # 像预期那样"降级但至少能跑"。SSCD 主后端缺失已经是一次降级(→resnet18),
                # 这里再对 resnet18 自身的权重缺失做二次静默降级(→联网下载)，会让整条
                # 去重链路在完全没有任何权重的环境下看似"优雅降级"、实则卡死，且卡死点
                # 发生在 gatekeeper 审计阶段、报错信息只有网络超时，排障成本极高。
                # 按本项目"硬失败不静默降级"哲学(authenticity.py 同类修复先例)，改为
                # 显式 raise——调用方(run_gatekeeper_audit → predict.py/video_predict.py
                # 的 except Exception)会把它转成 status=ERROR 的 result.json，而不是让
                # 进程在无网环境下无限期挂起。
                raise RuntimeError(
                    "[FATAL] Deduplicator resnet18 降级后端权重缺失: "
                    f"{local_weights} 不存在,且离线容器不允许联网下载("
                    "torchvision pretrained=True 会尝试联网,官方平台无网络访问)。"
                    "去重是 gatekeeper 一票否决链路的一环,不能在权重链路完全断裂时"
                    "继续尝试联网兜底——请确认打包流程已把 resnet18-f37072fd.pth "
                    "放入 ~/.cache/torch/hub/checkpoints/,或修复 SSCD 主后端权重缺失"
                    "(sscd_weights 缺失是本次降级到 resnet18 的根因)。"
                )
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
    """
    def __init__(self, device='cpu', phash_threshold=5, cosine_threshold=None,
                 sscd_weights=None):
        self.phash_threshold = phash_threshold
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
        
        self.hashes.append(dhash)
        self.embeddings.append(feat)
        
        if self.use_faiss:
            # FAISS IndexFlatIP expects float32 array
            self.index.add(np.expand_dims(feat, axis=0).astype(np.float32))
            
    def is_duplicate(self, img_np):
        if not self.hashes:
            return False, None
            
        query_hash = compute_dhash(img_np)
        query_feat = self.extractor.extract(img_np)
        
        # 1. Fast dHash screen
        for idx, stored_hash in enumerate(self.hashes):
            dist = hamming_distance(query_hash, stored_hash)
            if dist <= self.phash_threshold:
                return True, f"dHash match (distance: {dist})"
                
        # 2. ResNet feature screening
        if self.use_faiss:
            # Query FAISS
            distances, indices = self.index.search(np.expand_dims(query_feat, axis=0).astype(np.float32), 1)
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
