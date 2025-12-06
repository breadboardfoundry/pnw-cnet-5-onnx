# pnw-cnet-5-onnx
A repo testing performance improvements of the PNW CNET 5 Model in ONNX. 

Status: very much a work in progress

# Steps to run

1. Download the sample dataset: https://zenodo.org/records/10895837
2. Convert the audio files to the image format used by the model.
3. Run the inference using the exisitng .h5 model
4. Run the inference using the converted ONNX model
5. Compare the performance and accuracy of both models

# References

Dataset: https://zenodo.org/records/10895837

Model: https://github.com/zjruff/PNW-Cnet-5/tree/main

Precedent for conversion to ONNX: https://github.com/birdnet-team/BirdNET-Analyzer/issues/177#issuecomment-3549538447
