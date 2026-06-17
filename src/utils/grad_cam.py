import cv2
import numpy as np
import torch

class GradCAM:
    """
    Grad-CAM (Class Activation Mapping) implementation for highlighting
    salient features contributing to malignancy classification.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_tensor, target_class=1):
        """
        Generates a 2D Grad-CAM heatmap normalized to [0, 1].
        """
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(input_tensor)
        mal_logits = outputs["malignancy"]
        
        # Target score
        score = mal_logits[0, target_class]
        
        # Backward pass
        score.backward()
        
        # Extract gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0] # [C, H, W]
        activations = self.activations.cpu().data.numpy()[0] # [C, H, W]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2)) # [C]
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to retain only features with positive influence
        cam = np.maximum(cam, 0)
        
        # Resize to input tensor dimensions
        h, w = input_tensor.shape[2:]
        cam = cv2.resize(cam, (w, h))
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam
        
    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()

def overlay_heatmap(img_np, heatmap, alpha=0.4):
    """
    Overlays a heat map on the original grayscale/RGB image.
    """
    # Create colored heatmap from normalized 1D heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
    # Convert grayscale image to BGR if needed
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_np
        
    # Resize heatmap color to match image
    heatmap_colored = cv2.resize(heatmap_colored, (img_bgr.shape[1], img_bgr.shape[0]))
    
    # Blend images
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_colored, alpha, 0)
    return overlay
