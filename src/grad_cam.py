import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models import get_model

class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        outputs = self.model(input_tensor)
        probs = F.softmax(outputs, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(outputs, dim=1).item())

        confidence = float(probs[0, target_class].item())
        score = outputs[0, target_class]
        score.backward()

        weights = torch.mean(
            self.gradients, 
            dim=(2, 3), 
            keepdim=True
        )
        cam = torch.sum(
            weights * self.activations, 
            dim=1, 
            keepdim=False
        )  
        cam = F.relu(cam).squeeze(0)

        _, _, h, w = input_tensor.shape
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros((h, w), dtype=np.float32)

        return cam.astype(np.float32), target_class, confidence

def get_target_layer(model, model_name):
    name = model_name.lower()
    if name == "custom_cnn":
        conv_layers = [m for m in model.features.modules() if isinstance(m, torch.nn.Conv2d)]
        return conv_layers[-1]
    elif name == "resnet18":
        return model.layer4[-1]
    elif name == "efficientnet_b0":
        return model.features[-1]
    else:
        raise ValueError(f"Automatic selection of target_layer for model `{model_name}` is not yet supported.")

def overlay_heatmap_on_image(rgb_image, cam, alpha=0.45):
    h, w, _ = rgb_image.shape
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(rgb_image, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return overlay

def run_grad_cam(checkpoint_path, image_path, output_path="./logs/grad_cam_result.png", target_class=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get(
        "class_names",
        ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    )

    file_name = os.path.basename(checkpoint_path).lower()
    if "resnet18" in file_name:
        model_name = "resnet18"
    elif "efficientnet_b0" in file_name:
        model_name = "efficientnet_b0"
    else:
        model_name = "custom_cnn"

    model = get_model(model_name=model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    pil_img = Image.open(image_path).convert("RGB")
    rgb_img = np.array(pil_img.resize((224, 224), Image.BILINEAR))

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    cam, pred_idx, conf = grad_cam.generate_cam(input_tensor, target_class=target_class)
    overlay = overlay_heatmap_on_image(rgb_img, cam, alpha=0.45)

    heatmap_colored = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB
    )

    combined = np.hstack([rgb_img, heatmap_colored, overlay])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    label_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    print(f"[Grad-CAM] Model: {model_name} | Class: {label_name} ({conf * 100:.2f}%) | Saved: {output_path}")

if __name__ == "__main__":
    CKPT = "./checkpoints/resnet18_adamw_best.pth"
    SAMPLE_IMG = "./data/val/scratches/scratches_241.jpg"
    if os.path.exists(CKPT) and os.path.exists(SAMPLE_IMG):
        run_grad_cam(CKPT, SAMPLE_IMG, output_path="./logs/grad_cam_crack.png")
    else:
        print("Please update the paths to the checkpoint and the sample image with valid ones in order to run `src/grad_cam.py`.")