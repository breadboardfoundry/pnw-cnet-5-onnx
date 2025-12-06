# PNW-Cnet-5 ONNX Conversion Tutorial

This repository documents converting the [PNW-Cnet-5](https://github.com/zjruff/PNW-Cnet-5) bioacoustic classifier from Keras/TensorFlow to ONNX format, achieving **12x faster inference** on Apple Silicon using CoreML acceleration.

## Results Summary

| Metric | Keras (CPU) | ONNX (CoreML + ANE) |
|--------|-------------|---------------------|
| Time per image | 18.2ms | 1.5ms |
| 3300 images | 60s | 5s |
| Speedup | 1x | **12x** |
| Classification accuracy | baseline | 100% match |

### Benchmark Environment

- **Hardware:** Apple M2 (8-core CPU, 10-core GPU, 16-core Neural Engine)
- **OS:** macOS 15 (Sequoia)
- **Python:** 3.12
- **ONNX Runtime:** 1.20+ with CoreMLExecutionProvider
- **Dataset:** 3300 spectrogram images (257x1000 grayscale)

## Quick Start

### Prerequisites

- Python 3.10-3.12
- [sox](http://sox.sourceforge.net/) for audio processing
- macOS with Apple Silicon (for CoreML acceleration)

```bash
# Install sox
brew install sox

# Install Python dependencies
uv sync
```

### Convert Audio to Spectrograms

```bash
# Single file
uv run python wav_to_spectrogram.py recording.wav output_spectrograms/

# Entire directory
uv run python wav_to_spectrogram.py recordings/ output_spectrograms/
```

Audio files are split into 12-second clips, each producing a 257x1000 grayscale spectrogram.

### Convert Model to ONNX

```bash
uv run python convert_to_onnx.py
```

This converts `model/Final_Model.h5` to `model/Final_Model.onnx`.

### Run Inference

**With ONNX (fast):**
```bash
uv run python run_inference_onnx.py output_spectrograms/

# Export to CSV
uv run python run_inference_onnx.py output_spectrograms/ --output predictions.csv
```

**With Keras (for comparison):**
```bash
uv run python run_inference.py output_spectrograms/
```

### Compare Models

Verify the ONNX conversion produces identical results:

```bash
uv run python compare_models.py output_spectrograms/
```

Output shows numerical differences, classification agreement, and performance comparison.

## Command Reference

### wav_to_spectrogram.py
```bash
uv run python wav_to_spectrogram.py <input> <output_dir> [--clip-duration 12]
```

### convert_to_onnx.py
```bash
uv run python convert_to_onnx.py [--input model.h5] [--output model.onnx] [--opset 13]
```

### run_inference_onnx.py
```bash
uv run python run_inference_onnx.py <input_dir> [options]
  --model PATH        ONNX model path (default: model/Final_Model.onnx)
  --threshold FLOAT   Confidence threshold (default: 0.5)
  --top-k INT         Top predictions to show (default: 5)
  --batch-size INT    Batch size (default: 32)
  --output PATH       Export results to CSV
```

### compare_models.py
```bash
uv run python compare_models.py <input_dir> [options]
  --h5-model PATH     Keras model path
  --onnx-model PATH   ONNX model path
  --threshold FLOAT   Classification threshold
```

## References

- **Dataset:** https://zenodo.org/records/10895837
- **Original Model:** https://github.com/zjruff/PNW-Cnet-5
- **BirdNET ONNX Conversion:** https://github.com/birdnet-team/BirdNET-Analyzer/issues/177

## Reference Scripts

The `reference_scripts/` directory contains the original model architecture and training scripts from [PNW-Cnet-5](https://github.com/zjruff/PNW-Cnet-5).
