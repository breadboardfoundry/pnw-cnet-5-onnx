# pnw-cnet-5-onnx
A repo testing performance improvements of the PNW CNET 5 Model in ONNX. 

Status: very much a work in progress

# Basic Steps

1. Download the sample dataset: https://zenodo.org/records/10895837
2. Convert the audio files to the image format used by the model.
3. Run the inference using the exisitng .h5 model
4. Run the inference using the converted ONNX model
5. Compare the performance and accuracy of both models

# Usage

## Prerequisites

- Python 3.13+
- [sox](http://sox.sourceforge.net/) command-line tool for audio processing

Install sox on macOS:
```bash
brew install sox
```

Install Python dependencies:
```bash
uv sync
```

## Convert WAV to Spectrograms

The model requires 257x1000 grayscale spectrogram images. Use `wav_to_spectrogram.py` to convert audio files:

```bash
# Single file
uv run wav_to_spectrogram.py recordings/Site_001_Rep_B.wav output_spectrograms/

# Entire directory (recursive)
uv run wav_to_spectrogram.py recordings/ output_spectrograms/

# Custom clip duration (default is 12 seconds)
uv run wav_to_spectrogram.py recordings/ output_spectrograms/ --clip-duration 10
```

Audio files are split into 12-second clips, and each clip produces one spectrogram image named `{filename}_part_001.png`, `_part_002.png`, etc.

## Run Inference with H5 Model

Once you have spectrograms, run inference using the pre-trained Keras model:

```bash
# Basic usage
uv run python run_inference.py output_spectrograms/

# Adjust confidence threshold (default: 0.5)
uv run python run_inference.py output_spectrograms/ --threshold 0.3

# Show more top predictions per image (default: 5)
uv run python run_inference.py output_spectrograms/ --top-k 10

# Export results to CSV
uv run python run_inference.py output_spectrograms/ --output predictions.csv

# Use a different model file
uv run python run_inference.py output_spectrograms/ --model path/to/model.h5
```

The script outputs:
- Detections above the confidence threshold for each spectrogram
- Top-k predictions regardless of threshold
- Optional CSV with raw prediction scores for all 135 classes

## Convert H5 Model to ONNX

Convert the Keras H5 model to ONNX format for optimized inference:

```bash
# Default paths (model/Final_Model.h5 -> model/Final_Model.onnx)
uv run python convert_to_onnx.py

# Custom paths
uv run python convert_to_onnx.py --input model/Final_Model.h5 --output model/Final_Model.onnx

# Specify ONNX opset version
uv run python convert_to_onnx.py --opset 15
```

The conversion script will:
- Load the Keras model architecture and weights
- Convert to ONNX format using tf2onnx
- Verify the conversion by comparing outputs between Keras and ONNX models

## Run Inference with ONNX Model

Once you have the ONNX model, run inference using ONNX Runtime:

```bash
# Basic usage
uv run python run_inference_onnx.py output_spectrograms/

# Adjust confidence threshold (default: 0.5)
uv run python run_inference_onnx.py output_spectrograms/ --threshold 0.3

# Show more top predictions per image (default: 5)
uv run python run_inference_onnx.py output_spectrograms/ --top-k 10

# Export results to CSV
uv run python run_inference_onnx.py output_spectrograms/ --output predictions_onnx.csv

# Use a different model file
uv run python run_inference_onnx.py output_spectrograms/ --model path/to/model.onnx

# Adjust batch size for inference (default: 32)
uv run python run_inference_onnx.py output_spectrograms/ --batch-size 64
```

The ONNX inference script uses CoreML acceleration on macOS (CPU + GPU + Neural Engine) for ~12x speedup over the Keras model.

## Compare Models

Compare inference results between Keras and ONNX models to verify conversion accuracy:

```bash
# Basic comparison
uv run python compare_models.py output_spectrograms/

# Custom model paths
uv run python compare_models.py output_spectrograms/ --h5-model model/Final_Model.h5 --onnx-model model/Final_Model.onnx

# Adjust classification threshold
uv run python compare_models.py output_spectrograms/ --threshold 0.3
```

The comparison script outputs:
- Numerical differences (max, mean, std absolute difference)
- Classification decision agreement
- Performance timing comparison (Keras vs ONNX speedup)
- Per-image top prediction comparison

# References

Dataset: https://zenodo.org/records/10895837

Model: https://github.com/zjruff/PNW-Cnet-5/tree/main

Precedent for conversion to ONNX: https://github.com/birdnet-team/BirdNET-Analyzer/issues/177#issuecomment-3549538447


## Reference Scripts

The original model architecture and training scripts are in the `reference_scripts/` directory here. These are from
the original PNW-Cnet-5 repository: https://github.com/zjruff/PNW-Cnet-5
