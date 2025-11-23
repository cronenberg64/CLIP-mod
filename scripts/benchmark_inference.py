import torch
import open_clip
import time
import pandas as pd
from tabulate import tabulate

def benchmark_model(model_name, pretrained=False, batch_size=1, device='cuda'):
    print(f"Benchmarking {model_name} (Batch Size: {batch_size})...")
    
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            force_custom_text=True if 'convnext' in model_name else False
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    # Dummy Input
    image = torch.randn(batch_size, 3, 224, 224).to(device)
    text = open_clip.tokenize(["a photo of a cat"] * batch_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(image, text)
    
    torch.cuda.synchronize()
    
    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.reset_peak_memory_stats()
    start_event.record()
    
    num_runs = 100
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(image, text)
            
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / num_runs
    fps = (batch_size * num_runs) / (total_time_ms / 1000)
    
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    
    return {
        "Model": model_name,
        "Batch Size": batch_size,
        "Latency (ms)": f"{avg_time_ms:.2f}",
        "FPS": f"{fps:.2f}",
        "Peak Mem (MB)": f"{peak_mem_mb:.2f}",
        "Params (M)": f"{params_m:.2f}"
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    models_to_test = [
        ("ViT-B-32", None), # Standard CLIP
        ("convnext_v2_tiny", None) # Our model
    ]
    
    results = []
    
    for model_name, pretrained in models_to_test:
        for bs in [1, 32]:
            res = benchmark_model(model_name, pretrained=pretrained, batch_size=bs, device=device)
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(tabulate(df, headers='keys', tablefmt='psql'))
    
    # Save to CSV
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
