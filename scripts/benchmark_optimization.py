import torch
import open_clip
import onnxruntime
import time
import numpy as np
import pandas as pd
from tabulate import tabulate
import argparse

def benchmark_pytorch(model, input_tensor, device, fp16=False, num_runs=100):
    model.eval()
    if fp16:
        model.half()
        input_tensor = input_tensor.half()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event) / num_runs

def benchmark_onnx(onnx_path, input_array, num_runs=100):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: input_array})
        
    start_time = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: input_array})
    end_time = time.time()
    
    return (end_time - start_time) * 1000 / num_runs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="convnext_v2_tiny")
    parser.add_argument("--onnx", type=str, default="convnext_v2_tiny.onnx")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking {args.model} on {device} with Batch Size {args.batch_size}")
    
    # Load PyTorch Model
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, 
        pretrained=False,
        force_custom_text=True if 'convnext' in args.model else False
    )
    model = model.visual.to(device)
    
    input_tensor = torch.randn(args.batch_size, 3, 224, 224).to(device)
    input_array = input_tensor.cpu().numpy()
    
    results = []
    
    # 1. PyTorch FP32
    print("Running PyTorch FP32...")
    latency_fp32 = benchmark_pytorch(model, input_tensor, device, fp16=False)
    results.append({"Method": "PyTorch FP32", "Latency (ms)": latency_fp32})
    
    # 2. PyTorch FP16
    print("Running PyTorch FP16...")
    # Reload model to reset weights/dtype
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, 
        pretrained=False,
        force_custom_text=True if 'convnext' in args.model else False
    )
    model = model.visual.to(device)
    latency_fp16 = benchmark_pytorch(model, input_tensor, device, fp16=True)
    results.append({"Method": "PyTorch FP16", "Latency (ms)": latency_fp16})
    
    # 3. ONNX
    print("Running ONNX Runtime...")
    try:
        latency_onnx = benchmark_onnx(args.onnx, input_array)
        results.append({"Method": "ONNX Runtime", "Latency (ms)": latency_onnx})
    except Exception as e:
        print(f"ONNX Benchmark failed: {e}")
        
    df = pd.DataFrame(results)
    df["FPS"] = (1000 / df["Latency (ms)"]) * args.batch_size
    
    print("\nOptimization Benchmark Results:")
    print(tabulate(df, headers='keys', tablefmt='psql'))
    
    df.to_csv("benchmark_optimization.csv", index=False)

if __name__ == "__main__":
    main()
