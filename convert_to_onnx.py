"""
Convert the PNW-Cnet-5 H5 model to ONNX format.

This script loads the Keras model architecture, loads the weights from the H5 file,
saves as a TensorFlow SavedModel, then converts to ONNX using the tf2onnx CLI.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential


# Number of output classes (same as in run_inference.py)
NUM_CLASSES = 135


def build_model(nclasses: int) -> Sequential:
    """
    Build the PNW-Cnet-5 CNN architecture.

    Architecture matches Train_Model.py with:
    - 6 Conv2D blocks with 32->32->64->64->128->128 filters
    - Dropout of 0.30 throughout
    - Dense layer with 512 units

    Args:
        nclasses: Number of output classes

    Returns:
        Keras Sequential model
    """
    dropout_prop = 0.30
    n_fc_nodes = 512

    model = Sequential()

    # Block 1
    model.add(
        Conv2D(
            32, (5, 5),
            input_shape=(257, 1000, 1),
            data_format="channels_last",
            activation="relu",
            padding="same",
        )
    )
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    # Block 2
    model.add(Conv2D(32, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    # Block 3
    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    # Block 4
    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    # Block 5
    model.add(Conv2D(128, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    # Block 6
    model.add(Conv2D(128, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    model.add(Flatten())

    # Dense layers
    model.add(Dense(n_fc_nodes, activation="relu"))
    model.add(Dropout(dropout_prop))

    model.add(Dense(nclasses, activation="sigmoid"))

    return model


def convert_to_onnx(h5_path: str, onnx_path: str, opset: int = 13) -> None:
    """
    Convert a Keras H5 model to ONNX format via SavedModel.

    Args:
        h5_path: Path to the H5 weights file
        onnx_path: Output path for the ONNX model
        opset: ONNX opset version (default: 13)
    """
    print("Building model architecture...")
    model = build_model(NUM_CLASSES)

    print(f"Loading weights from: {h5_path}")
    model.load_weights(h5_path)

    # Build model with concrete input shape
    model.build(input_shape=(None, 257, 1000, 1))

    print("Model summary:")
    model.summary()

    # Create temporary directory for SavedModel
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_path = Path(tmpdir) / "saved_model"

        print(f"\nExporting to TensorFlow SavedModel format...")
        model.export(str(saved_model_path))

        print(f"\nConverting to ONNX (opset {opset}) using tf2onnx CLI...")

        # Use tf2onnx command-line tool to avoid Keras 3 API incompatibilities
        cmd = [
            "python", "-m", "tf2onnx.convert",
            "--saved-model", str(saved_model_path),
            "--output", onnx_path,
            "--opset", str(opset),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(f"tf2onnx conversion failed with code {result.returncode}")

    print(f"\nONNX model saved to: {onnx_path}")

    # Verify the conversion
    print("\nVerifying conversion...")
    verify_conversion(model, onnx_path)


def verify_conversion(keras_model: Sequential, onnx_path: str) -> None:
    """
    Verify that the ONNX model produces the same outputs as the Keras model.

    Args:
        keras_model: The original Keras model
        onnx_path: Path to the converted ONNX model
    """
    import onnxruntime as ort

    # Create random test input
    test_input = np.random.rand(1, 257, 1000, 1).astype(np.float32)

    # Get Keras prediction
    keras_output = keras_model.predict(test_input, verbose=0)

    # Get ONNX prediction
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: test_input})[0]

    # Compare outputs
    max_diff = np.max(np.abs(keras_output - onnx_output))
    mean_diff = np.mean(np.abs(keras_output - onnx_output))

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")

    if max_diff < 1e-4:
        print("  Verification PASSED: Outputs match within tolerance")
    else:
        print("  WARNING: Outputs differ more than expected")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PNW-Cnet-5 H5 model to ONNX format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="model/Final_Model.h5",
        help="Path to the H5 model file (default: model/Final_Model.h5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model/Final_Model.onnx",
        help="Output path for ONNX model (default: model/Final_Model.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert
    convert_to_onnx(args.input, args.output, args.opset)

    return 0


if __name__ == "__main__":
    exit(main())
