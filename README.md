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

# References

Dataset: https://zenodo.org/records/10895837

Model: https://github.com/zjruff/PNW-Cnet-5/tree/main

Precedent for conversion to ONNX: https://github.com/birdnet-team/BirdNET-Analyzer/issues/177#issuecomment-3549538447
