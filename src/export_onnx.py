import os
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

from src.models import get_model


def infer_model_name(checkpoint_path: str) -> str:
    name = os.path.basename(checkpoint_path).lower()
    if "resnet18" in name:
        return "resnet18"
    if "efficientnet_b0" in name:
        return "efficientnet_b0"
    return "custom_cnn"


def export_to_onnx(
    checkpoint_path: str,
    onnx_path: str,
    img_size: int = 224,
    opset_version: int = 17,
    atol: float = 1e-4,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device("cpu")  
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint.get(
        "class_names",
        ["crazing", "iclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
    )
    num_classes = len(class_names)
    model_name = infer_model_name(checkpoint_path)

    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    with torch.no_grad():
        torch_logits = model(dummy_input).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_logits = session.run(None, ort_inputs)[0]

    max_diff = float(np.max(np.abs(torch_logits - ort_logits)))
    is_close = np.allclose(torch_logits, ort_logits, rtol=1e-4, atol=atol)
    print(f"Maximum error between PyTorch and ONNX Runtime: {max_diff:.2e} | Joint: {is_close}")

    if not is_close:
        raise RuntimeError("Warning: The discrepancy between PyTorch and ONNX exceeds the allowable threshold.!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch defect model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/resnet18_adamw_best.pth",
        help="Path to the .pth checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./checkpoints/model.onnx",
        help="Path to the exported .onnx file",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        onnx_path=args.output,
        img_size=args.img_size,
        opset_version=args.opset,
    )