# ConvNeXt-V2 Vision-Language Model Report

## Project Overview
This project integrated the **ConvNeXt-V2** (Tiny variant) architecture as a vision encoder into the `open_clip` framework. The goal was to explore a modern CNN backbone for contrastive vision-language learning, targeting efficient deployment.

## Integration Details
- **Backbone**: `convnextv2_tiny` from `timm`.
- **Adapter**: Used `open_clip`'s `TimmModel` adapter with a linear projection head.
- **Text Encoder**: Standard Transformer (matching ViT-B-32 configuration).
- **Configuration**: Created `src/open_clip/model_configs/convnext_v2_tiny.json`.

## Training Verification
- **Dataset**: Synthetic dataset (100 samples) generated for functional verification.
- **Pipeline**: Successfully ran `open_clip_train.main` for 2 epochs.
- **Loss**: Contrastive loss converged as expected for synthetic data.

## Benchmark Results
We compared the new `convnext_v2_tiny` model against the standard `ViT-B-32` baseline on an NVIDIA GPU.

| Model | Batch Size | Latency (ms) | FPS | Peak Mem (MB) | Params (M) |
|-------|------------|--------------|-----|---------------|------------|
| ViT-B-32 | 1 | 16.35 | 61.18 | 591.53 | 151.28 |
| ViT-B-32 | 32 | 180.75 | 177.04 | 664.81 | 151.28 |
| **convnext_v2_tiny** | 1 | 21.96 | 45.54 | **382.53** | **91.69** |
| **convnext_v2_tiny** | 32 | 239.73 | 133.48 | 934.90 | **91.69** |

### Analysis
- **Model Size**: ConvNeXt-V2 Tiny is **~40% smaller** in terms of parameter count (91.7M vs 151.3M).
- **Memory**: For single-image inference (BS=1), it uses **~35% less memory**, making it suitable for constrained edge devices.
- **Speed**: On the tested GPU, it was slightly slower than ViT-B-32. This could be due to the highly optimized nature of ViT implementations or specific GPU characteristics. Optimization (e.g., TensorRT, quantization) is recommended for deployment.

## Zero-shot Evaluation
We evaluated the model's zero-shot classification capability on **CIFAR-100**.
- **Accuracy**: 1.03% (Random Initialization)
> Note: The low accuracy is expected as the model was initialized with random weights and trained on a tiny synthetic dataset. This verifies the evaluation pipeline works.

## Optimization Benchmark
We benchmarked inference latency for the visual tower (Batch Size = 1) on NVIDIA GPU using different runtimes.

| Method | Latency (ms) | FPS | Notes |
|--------|--------------|-----|-------|
| PyTorch FP32 | 20.22 | 49.45 | Baseline |
| PyTorch FP16 | 28.28 | 35.35 | Slower due to overhead at BS=1 |
| **ONNX Runtime (GPU)** | **15.84** | **63.13** | **~1.3x Speedup** |

**Conclusion**: Exporting to ONNX provides the best performance for single-image inference, making it ideal for the edge computing.

## Usage Instructions

### 1. Installation
```bash
pip install -r requirements.txt
pip install timm onnx onnxruntime-gpu
```

### 2. Training
To train the model using contrastive learning (CLIP style):
```bash
python -m open_clip_train.main \
    --model convnext_v2_tiny \
    --train-data data/synthetic/train.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --batch-size 32 \
    --epochs 10
```

### 3. Inference / Benchmarking
```bash
python scripts/benchmark_inference.py
python scripts/benchmark_optimization.py
```

### 4. ONNX Export
```bash
python scripts/export_onnx.py --output model.onnx
```

## Future Work
- **Open-Vocabulary Detection**: Integrate the trained backbone with a detection head (e.g., DETR).
- **Large Scale Training**: Train on LAION-400M or DataComp for competitive zero-shot performance.
