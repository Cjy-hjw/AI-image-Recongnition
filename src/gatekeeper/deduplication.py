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

class FeatureExtractor:
    """Extracts deep features from images using a pre-trained ResNet18."""
    def __init__(self, device='cpu'):
        self.device = device
        # Use torchvision ResNet18 pre-trained on ImageNet
        self.model = models.resnet18(pretrained=False)
        local_weights = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
        if os.path.exists(local_weights):
            self.model.load_state_dict(torch.load(local_weights, map_location='cpu'))
        else:
            self.model = models.resnet18(pretrained=True)
        # Remove the classification head (fc layer)
        self.model.fc = torch.nn.Identity()
        self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def extract(self, img_np):
        # Convert grayscale to RGB for ResNet
        if len(img_np.shape) == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_np
            
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = feat.squeeze(0).cpu().numpy()
            
        # L2 Normalize feature vector
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

class Deduplicator:
    """
    Main deduplication module.
    Performs fast dHash screening first, then verifies using cosine similarity of ResNet embeddings.
    Supports FAISS with standard NumPy fallback.
    """
    def __init__(self, device='cpu', phash_threshold=5, cosine_threshold=0.95):
        self.phash_threshold = phash_threshold
        self.cosine_threshold = cosine_threshold
        self.extractor = FeatureExtractor(device=device)
        
        # In-memory database of hashes and embeddings
        self.hashes = []      # list of dHash strings
        self.embeddings = []  # list of feature vectors
        
        # Try to import FAISS
        try:
            import faiss
            self.use_faiss = True
            # ResNet18 output dimension is 512
            self.index = faiss.IndexFlatIP(512)
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
            D, I = self.index.search(np.expand_dims(query_feat, axis=0).astype(np.float32), 1)
            best_sim = D[0][0]
            best_idx = I[0][0]
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
