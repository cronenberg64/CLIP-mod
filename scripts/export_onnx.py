import torch
import open_clip
import onnxruntime
import numpy as np
import argparse

def export_onnx(model_name, output_file="model.onnx", opset_version=14):
    print(f"Exporting {model_name} to ONNX...")
    
    # 1. Load Model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=False,
        force_custom_text=True if 'convnext' in model_name else False
    )
    model.eval()
    
    # We only export the visual tower for now as it's the main component for robotics
    visual_model = model.visual
    
    # 2. Create Dummy Input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 3. Export
    torch.onnx.export(
        visual_model,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {output_file}")
    
    # 4. Verify
    print("Verifying ONNX model...")
    ort_session = onnxruntime.InferenceSession(output_file)
    
    # Compute PyTorch output
    with torch.no_grad():
        torch_out = visual_model(dummy_input).numpy()
        
    # Compute ONNX output
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Compare
    np.testing.assert_allclose(torch_out, ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Verification Successful! PyTorch and ONNX outputs match.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="convnext_v2_tiny", help="Model name")
    parser.add_argument("--output", type=str, default="convnext_v2_tiny.onnx", help="Output filename")
    args = parser.parse_args()
    
    export_onnx(args.model, args.output)
