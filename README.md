# CLIP-mod: Modular ConvNeXt-V2 Vision-Language Model

**An efficient, CNN-based VLM built on OpenCLIP for Edge Robotics.**

This project integrates the **ConvNeXt-V2** architecture (specifically the Tiny variant) into the OpenCLIP framework to create a VLM optimized for embedded systems like the NVIDIA Jetson. It replaces the standard ViT backbone with a modern CNN to achieve lower memory usage while maintaining semantic alignment capabilities. The project is a personal project that I decided to tackle for my own personal interest in future robotics use.

## Key Features
- **Efficient Backbone**: Uses `convnextv2_tiny` (~28M params) as the vision encoder.
- **Low Memory Footprint**: ~35% reduction in peak memory usage compared to ViT-B-32 during inference.
- **OpenCLIP Integration**: Fully compatible with OpenCLIP's training and inference pipelines.
- **Edge-Ready**: Designed for deployment on robotics platforms (RiOne).

## Performance Benchmark

| Model | Batch Size | Latency (ms) | FPS | Peak Mem (MB) | Params (M) |
|-------|------------|--------------|-----|---------------|------------|
| ViT-B-32 (Baseline) | 1 | 16.35 | 61.18 | 591.53 | 151.28 |
| **ConvNeXt-V2 Tiny** | 1 | 21.96 | 45.54 | **382.53** | **91.69** |

*Benchmarks run on NVIDIA GPU. See [REPORT.md](REPORT.md) for full details.*

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/cronenberg64/CLIP-mod.git
cd CLIP-mod

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Training
To train the model using contrastive learning (CLIP style):

```bash
python -m open_clip_train.main \
    --model convnext_v2_tiny \
    --train-data data/synthetic/train.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --batch-size 32 \
    --epochs 10 \
    --save-frequency 1
```

### Inference & Benchmarking
```bash
# Run inference benchmark (Latency, Memory, FPS)
python scripts/benchmark_inference.py

# Run optimization benchmark (FP32 vs FP16 vs ONNX)
python scripts/benchmark_optimization.py

# Evaluate Zero-shot Accuracy (CIFAR-100)
python scripts/evaluate_zeroshot.py
```

### Deployment (ONNX)
Export the vision tower to ONNX for edge deployment:
```bash
python scripts/export_onnx.py --output convnext_v2_tiny.onnx
```

## Project Structure
- `src/open_clip/`: Modified OpenCLIP source code.
- `src/open_clip/model_configs/`: Configuration files (includes `convnext_v2_tiny.json`).
- `scripts/`: Utility scripts (benchmarking, data generation).
- `REPORT.md`: Detailed project report and analysis.

## Acknowledgements
This project is a fork of [OpenCLIP](https://github.com/mlfoundations/open_clip). I would like to thank the OpenCLIP authors and contributors for their excellent framework.

## License
MIT License. See [LICENSE](LICENSE) for details.
