"""
Compare inference results between Keras H5 and ONNX models.

This script runs inference on the same images using both models and compares
the outputs to verify the ONNX conversion is accurate.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from PIL import Image


# Class labels for the 135 bioacoustic categories
CLASSES = [
    "ACCO1", "ACGE1", "ACGE2", "ACST1", "AEAC1", "AEAC2", "Airplane", "ANCA1",
    "ASOT1", "BOUM1", "BRCA1", "BRMA1", "BRMA2", "BUJA1", "BUJA2", "Bullfrog",
    "BUVI1", "BUVI2", "CACA1", "CAGU1", "CAGU2", "CAGU3", "CALA1", "CALU1",
    "CAPU1", "CAUS1", "CAUS2", "CCOO1", "CCOO2", "CECA1", "Chainsaw", "CHFA1",
    "Chicken", "CHMI1", "CHMI2", "COAU1", "COAU2", "COBR1", "COCO1", "COSO1",
    "Cow", "Creek", "Cricket", "CYST1", "CYST2", "DEFU1", "DEFU2", "Dog",
    "DRPU1", "Drum", "EMDI1", "EMOB1", "FACO1", "FASP1", "Fly", "Frog",
    "GADE1", "GLGN1", "Growler", "Gunshot", "HALE1", "HAPU1", "HEVE1",
    "Highway", "Horn", "Human", "HYPI1", "IXNA1", "IXNA2", "JUHY1", "LEAL1",
    "LECE1", "LEVI1", "LEVI2", "LOCU1", "MEFO1", "MEGA1", "MEKE1", "MEKE2",
    "MEKE3", "MYTO1", "NUCO1", "OCPR1", "ODOC1", "ORPI1", "ORPI2", "PAFA1",
    "PAFA2", "PAHA1", "PECA1", "PHME1", "PHNU1", "PILU1", "PILU2", "PIMA1",
    "PIMA2", "POEC1", "POEC2", "PSFL1", "Rain", "Raptor", "SICU1", "SITT1",
    "SITT2", "SPHY1", "SPHY2", "SPPA1", "SPPI1", "SPTH1", "STDE1", "STNE1",
    "STNE2", "STOC_4Note", "STOC_Series", "Strix_Bark", "Strix_Whistle",
    "STVA_8Note", "STVA_Insp", "STVA_Series", "Survey_Tone", "TADO1", "TADO2",
    "TAMI1", "Thunder", "TRAE1", "Train", "Tree", "TUMI1", "TUMI2", "URAM1",
    "VIHU1", "Wildcat", "Yarder", "ZEMA1", "ZOLE1",
]

NUM_CLASSES = 135


def build_keras_model(nclasses: int) -> Sequential:
    """Build the PNW-Cnet-5 CNN architecture."""
    dropout_prop = 0.30
    n_fc_nodes = 512

    model = Sequential()

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

    model.add(Conv2D(32, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    model.add(Conv2D(128, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    model.add(Conv2D(128, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout_prop))

    model.add(Flatten())
    model.add(Dense(n_fc_nodes, activation="relu"))
    model.add(Dropout(dropout_prop))
    model.add(Dense(nclasses, activation="sigmoid"))

    return model


def load_image(image_path: str) -> np.ndarray:
    """Load and preprocess a spectrogram image."""
    img = Image.open(image_path).convert("L")
    img = img.resize((1000, 257), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array[:, :, np.newaxis]
    return img_array


def load_images_batch(image_paths: list[str]) -> np.ndarray:
    """Load multiple images into a batch."""
    images = [load_image(p) for p in image_paths]
    return np.stack(images, axis=0)


def run_comparison(
    input_dir: str,
    h5_model_path: str,
    onnx_model_path: str,
    threshold: float = 0.5,
) -> dict:
    """
    Run inference with both models and compare results.

    Returns:
        Dictionary with comparison statistics
    """
    # Find images
    input_path = Path(input_dir)
    png_files = sorted(input_path.glob("*.png"))

    if not png_files:
        print(f"Error: No PNG files found in {input_dir}")
        return {}

    print(f"Found {len(png_files)} images")

    # Load Keras model
    print(f"\nLoading Keras model from: {h5_model_path}")
    keras_model = build_keras_model(NUM_CLASSES)
    keras_model.load_weights(h5_model_path)

    # Load ONNX model
    print(f"Loading ONNX model from: {onnx_model_path}")
    onnx_session = ort.InferenceSession(
        onnx_model_path,
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    onnx_input_name = onnx_session.get_inputs()[0].name

    # Load all images
    print("\nLoading images...")
    all_images = load_images_batch([str(f) for f in png_files])

    # Run Keras inference
    print("\nRunning Keras inference...")
    keras_start = time.perf_counter()
    keras_predictions = keras_model.predict(all_images, verbose=0)
    keras_time = time.perf_counter() - keras_start

    # Run ONNX inference
    print("Running ONNX inference...")
    onnx_start = time.perf_counter()
    onnx_predictions = onnx_session.run(None, {onnx_input_name: all_images})[0]
    onnx_time = time.perf_counter() - onnx_start

    # Compare predictions
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    # Numerical comparison
    abs_diff = np.abs(keras_predictions - onnx_predictions)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    std_diff = np.std(abs_diff)

    print(f"\nNumerical Comparison:")
    print(f"  Max absolute difference:  {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Std absolute difference:  {std_diff:.2e}")

    # Check if predictions match within tolerance
    tolerance = 1e-4
    matching = np.allclose(keras_predictions, onnx_predictions, atol=tolerance)
    print(f"\n  Predictions match (tolerance={tolerance}): {'YES' if matching else 'NO'}")

    # Classification comparison (using threshold)
    keras_detections = keras_predictions >= threshold
    onnx_detections = onnx_predictions >= threshold
    detection_match = np.array_equal(keras_detections, onnx_detections)

    total_predictions = keras_predictions.size
    matching_detections = np.sum(keras_detections == onnx_detections)
    detection_accuracy = matching_detections / total_predictions * 100

    print(f"\nClassification Comparison (threshold={threshold}):")
    print(f"  Detection decisions match: {'YES' if detection_match else 'NO'}")
    print(f"  Matching decisions: {matching_detections}/{total_predictions} ({detection_accuracy:.2f}%)")

    # Per-image comparison
    print(f"\nPer-Image Analysis:")
    images_with_diff = 0
    for i, filepath in enumerate(png_files):
        img_max_diff = np.max(np.abs(keras_predictions[i] - onnx_predictions[i]))
        if img_max_diff > tolerance:
            images_with_diff += 1

    print(f"  Images with differences > {tolerance}: {images_with_diff}/{len(png_files)}")

    # Timing comparison
    print(f"\nPerformance Comparison ({len(png_files)} images):")
    print(f"  Keras time:  {keras_time:.3f}s ({keras_time/len(png_files)*1000:.2f}ms per image)")
    print(f"  ONNX time:   {onnx_time:.3f}s ({onnx_time/len(png_files)*1000:.2f}ms per image)")
    speedup = keras_time / onnx_time if onnx_time > 0 else float('inf')
    print(f"  ONNX speedup: {speedup:.2f}x")

    # Top prediction comparison
    print(f"\nTop Prediction Comparison (first 5 images):")
    for i, filepath in enumerate(png_files[:5]):
        keras_top_idx = np.argmax(keras_predictions[i])
        onnx_top_idx = np.argmax(onnx_predictions[i])

        keras_top_class = CLASSES[keras_top_idx]
        onnx_top_class = CLASSES[onnx_top_idx]

        keras_top_conf = keras_predictions[i][keras_top_idx]
        onnx_top_conf = onnx_predictions[i][onnx_top_idx]

        match_str = "MATCH" if keras_top_idx == onnx_top_idx else "DIFFER"
        print(f"\n  {filepath.name}:")
        print(f"    Keras: {keras_top_class} ({keras_top_conf:.4f})")
        print(f"    ONNX:  {onnx_top_class} ({onnx_top_conf:.4f})")
        print(f"    Status: {match_str}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if matching and detection_match:
        print("\nModels produce IDENTICAL results within tolerance.")
        print("The ONNX conversion is successful.")
    elif detection_match:
        print("\nModels produce SAME classification decisions.")
        print("Minor numerical differences exist but don't affect predictions.")
    else:
        print("\nWARNING: Models produce DIFFERENT classification decisions.")
        print("Review the conversion process.")

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "matching": matching,
        "detection_match": detection_match,
        "detection_accuracy": detection_accuracy,
        "keras_time": keras_time,
        "onnx_time": onnx_time,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare Keras H5 and ONNX model inference results"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing spectrogram PNG images",
    )
    parser.add_argument(
        "--h5-model",
        type=str,
        default="model/Final_Model.h5",
        help="Path to Keras H5 model (default: model/Final_Model.h5)",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        default="model/Final_Model.onnx",
        help="Path to ONNX model (default: model/Final_Model.onnx)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for classification comparison (default: 0.5)",
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.input_dir).is_dir():
        print(f"Error: {args.input_dir} is not a valid directory")
        return 1

    if not Path(args.h5_model).exists():
        print(f"Error: H5 model not found: {args.h5_model}")
        return 1

    if not Path(args.onnx_model).exists():
        print(f"Error: ONNX model not found: {args.onnx_model}")
        return 1

    # Run comparison
    run_comparison(
        args.input_dir,
        args.h5_model,
        args.onnx_model,
        args.threshold,
    )

    return 0


if __name__ == "__main__":
    exit(main())
