"""
Compare inference results between Keras H5, ONNX, and ONNX-slim models.

This script runs inference on the same images using all models and compares
the outputs to verify the ONNX conversion is accurate and measure performance.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def get_onnx_providers() -> list:
    """Get the best available ONNX execution providers for the current platform."""
    available = ort.get_available_providers()

    if sys.platform == "darwin" and "CoreMLExecutionProvider" in available:
        return [
            ("CoreMLExecutionProvider", {
                "ModelFormat": "NeuralNetwork",
                "MLComputeUnits": "ALL",
            }),
            "CPUExecutionProvider",
        ]
    elif "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        return ["CPUExecutionProvider"]


def setup_tensorflow():
    """Configure TensorFlow to use CPU only (Metal GPU produces incorrect results)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_METAL_DEVICE_SELECTOR"] = ""
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')


def load_keras_model(h5_model_path: str, nclasses: int):
    """Load Keras model with architecture and weights."""
    from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
    from keras.models import Sequential

    dropout_prop = 0.30
    n_fc_nodes = 512

    model = Sequential()

    model.add(Input(shape=(257, 1000, 1)))
    model.add(
        Conv2D(
            32, (5, 5),
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

    model.load_weights(h5_model_path)
    return model


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


def create_onnx_session(model_path: str) -> ort.InferenceSession:
    """Create an ONNX Runtime session with hardware acceleration."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 0  # 0 = use all available cores
    sess_options.inter_op_num_threads = 0
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    providers = get_onnx_providers()
    return ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers,
    )


def run_onnx_inference(
    session: ort.InferenceSession,
    images: np.ndarray,
    name: str,
    batch_size: int = 100,
) -> tuple[np.ndarray, float]:
    """Run ONNX inference and return predictions with timing."""
    input_name = session.get_inputs()[0].name
    print(f"Running {name} inference on {len(images)} images...", flush=True)

    start = time.perf_counter()
    predictions_list = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_preds = session.run(None, {input_name: batch})[0]
        predictions_list.append(batch_preds)
        print(f"  Processed {min(i + batch_size, len(images))}/{len(images)}", flush=True)
    predictions = np.vstack(predictions_list)
    elapsed = time.perf_counter() - start
    print(f"  {name} completed in {elapsed:.2f}s", flush=True)

    return predictions, elapsed


def run_comparison(
    input_dir: str,
    h5_model_path: str,
    onnx_model_path: str,
    onnx_slim_model_path: str | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Run inference with all models and compare results.

    Returns:
        Dictionary with comparison statistics
    """
    # Configure TensorFlow to use CPU (Metal GPU produces incorrect results)
    setup_tensorflow()

    # Find images
    input_path = Path(input_dir)
    png_files = sorted(input_path.glob("*.png"))

    if not png_files:
        print(f"Error: No PNG files found in {input_dir}")
        return {}

    print(f"Found {len(png_files)} images")

    # Load Keras model
    print(f"\nLoading Keras model from: {h5_model_path}")
    keras_model = load_keras_model(h5_model_path, NUM_CLASSES)

    # Load ONNX model
    print(f"Loading ONNX model from: {onnx_model_path}")
    onnx_session = create_onnx_session(onnx_model_path)

    # Load ONNX-slim model if provided
    onnx_slim_session = None
    if onnx_slim_model_path:
        print(f"Loading ONNX-slim model from: {onnx_slim_model_path}")
        onnx_slim_session = create_onnx_session(onnx_slim_model_path)

    # Load all images
    print("\nLoading images...")
    all_images = load_images_batch([str(f) for f in png_files])

    # Run Keras inference
    print(f"\nRunning Keras inference on {len(all_images)} images...", flush=True)
    keras_start = time.perf_counter()
    keras_predictions = keras_model.predict(all_images, verbose=1)
    keras_time = time.perf_counter() - keras_start
    print(f"  Keras completed in {keras_time:.2f}s", flush=True)

    # Run ONNX inference
    onnx_predictions, onnx_time = run_onnx_inference(
        onnx_session, all_images, "ONNX"
    )

    # Run ONNX-slim inference if available
    onnx_slim_predictions = None
    onnx_slim_time = None
    if onnx_slim_session:
        onnx_slim_predictions, onnx_slim_time = run_onnx_inference(
            onnx_slim_session, all_images, "ONNX-slim"
        )

    # Compare predictions
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    tolerance = 1e-4

    # Helper function for pairwise comparison
    def compare_predictions(pred_a, pred_b, name_a, name_b):
        abs_diff = np.abs(pred_a - pred_b)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        matching = np.allclose(pred_a, pred_b, atol=tolerance)
        return max_diff, mean_diff, matching

    # Keras vs ONNX
    keras_onnx_max, keras_onnx_mean, keras_onnx_match = compare_predictions(
        keras_predictions, onnx_predictions, "Keras", "ONNX"
    )

    print(f"\nNumerical Comparison (Keras vs ONNX):")
    print(f"  Max absolute difference:  {keras_onnx_max:.2e}")
    print(f"  Mean absolute difference: {keras_onnx_mean:.2e}")
    print(f"  Predictions match (tolerance={tolerance}): {'YES' if keras_onnx_match else 'NO'}")

    # ONNX vs ONNX-slim (if available)
    onnx_slim_match = None
    if onnx_slim_predictions is not None:
        onnx_slim_max, onnx_slim_mean, onnx_slim_match = compare_predictions(
            onnx_predictions, onnx_slim_predictions, "ONNX", "ONNX-slim"
        )
        print(f"\nNumerical Comparison (ONNX vs ONNX-slim):")
        print(f"  Max absolute difference:  {onnx_slim_max:.2e}")
        print(f"  Mean absolute difference: {onnx_slim_mean:.2e}")
        print(f"  Predictions match (tolerance={tolerance}): {'YES' if onnx_slim_match else 'NO'}")

    # Classification comparison (using threshold)
    keras_detections = keras_predictions >= threshold
    onnx_detections = onnx_predictions >= threshold
    detection_match = np.array_equal(keras_detections, onnx_detections)

    total_predictions = keras_predictions.size
    matching_detections = np.sum(keras_detections == onnx_detections)
    detection_accuracy = matching_detections / total_predictions * 100

    print(f"\nClassification Comparison (threshold={threshold}):")
    print(f"  Keras vs ONNX match: {'YES' if detection_match else 'NO'} ({detection_accuracy:.2f}%)")

    if onnx_slim_predictions is not None:
        onnx_slim_detections = onnx_slim_predictions >= threshold
        slim_detection_match = np.array_equal(onnx_detections, onnx_slim_detections)
        slim_matching = np.sum(onnx_detections == onnx_slim_detections)
        slim_accuracy = slim_matching / total_predictions * 100
        print(f"  ONNX vs ONNX-slim match: {'YES' if slim_detection_match else 'NO'} ({slim_accuracy:.2f}%)")

    # Performance comparison
    print(f"\nPerformance Comparison ({len(png_files)} images):")
    print(f"  {'Model':<12} {'Time':>10} {'Per Image':>12} {'vs Keras':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

    keras_ms = keras_time / len(png_files) * 1000
    print(f"  {'Keras':<12} {keras_time:>9.2f}s {keras_ms:>10.2f}ms {'1.00x':>10}")

    onnx_ms = onnx_time / len(png_files) * 1000
    onnx_speedup = keras_time / onnx_time if onnx_time > 0 else float('inf')
    print(f"  {'ONNX':<12} {onnx_time:>9.2f}s {onnx_ms:>10.2f}ms {onnx_speedup:>9.2f}x")

    onnx_slim_speedup = None
    if onnx_slim_time is not None:
        slim_ms = onnx_slim_time / len(png_files) * 1000
        onnx_slim_speedup = keras_time / onnx_slim_time if onnx_slim_time > 0 else float('inf')
        slim_vs_onnx = onnx_time / onnx_slim_time if onnx_slim_time > 0 else float('inf')
        print(f"  {'ONNX-slim':<12} {onnx_slim_time:>9.2f}s {slim_ms:>10.2f}ms {onnx_slim_speedup:>9.2f}x")
        print(f"\n  ONNX-slim vs ONNX: {slim_vs_onnx:.2f}x {'faster' if slim_vs_onnx > 1 else 'slower'}")

    # Top prediction comparison
    print(f"\nTop Prediction Comparison (first 5 images):")
    for i, filepath in enumerate(png_files[:5]):
        keras_top_idx = np.argmax(keras_predictions[i])
        onnx_top_idx = np.argmax(onnx_predictions[i])

        keras_top_class = CLASSES[keras_top_idx]
        onnx_top_class = CLASSES[onnx_top_idx]

        keras_top_conf = keras_predictions[i][keras_top_idx]
        onnx_top_conf = onnx_predictions[i][onnx_top_idx]

        print(f"\n  {filepath.name}:")
        print(f"    Keras:     {keras_top_class} ({keras_top_conf:.4f})")
        print(f"    ONNX:      {onnx_top_class} ({onnx_top_conf:.4f})")

        if onnx_slim_predictions is not None:
            slim_top_idx = np.argmax(onnx_slim_predictions[i])
            slim_top_class = CLASSES[slim_top_idx]
            slim_top_conf = onnx_slim_predictions[i][slim_top_idx]
            print(f"    ONNX-slim: {slim_top_class} ({slim_top_conf:.4f})")

        all_match = keras_top_idx == onnx_top_idx
        if onnx_slim_predictions is not None:
            all_match = all_match and (onnx_top_idx == slim_top_idx)
        print(f"    Status: {'ALL MATCH' if all_match else 'DIFFER'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if keras_onnx_match and detection_match:
        print("\nKeras and ONNX produce IDENTICAL results within tolerance.")
    elif detection_match:
        print("\nKeras and ONNX produce SAME classification decisions.")
        print("Minor numerical differences exist but don't affect predictions.")
    else:
        print("\nWARNING: Keras and ONNX produce DIFFERENT classification decisions.")

    if onnx_slim_predictions is not None:
        if onnx_slim_match:
            print("ONNX and ONNX-slim produce IDENTICAL results.")
        else:
            print("ONNX and ONNX-slim have minor numerical differences.")

    return {
        "keras_onnx_max_diff": keras_onnx_max,
        "keras_onnx_match": keras_onnx_match,
        "detection_match": detection_match,
        "detection_accuracy": detection_accuracy,
        "keras_time": keras_time,
        "onnx_time": onnx_time,
        "onnx_speedup": onnx_speedup,
        "onnx_slim_time": onnx_slim_time,
        "onnx_slim_speedup": onnx_slim_speedup,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare Keras H5, ONNX, and ONNX-slim model inference results"
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
        "--onnx-slim-model",
        type=str,
        default=None,
        help="Path to ONNX-slim model (optional)",
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

    if args.onnx_slim_model and not Path(args.onnx_slim_model).exists():
        print(f"Error: ONNX-slim model not found: {args.onnx_slim_model}")
        return 1

    # Run comparison
    run_comparison(
        args.input_dir,
        args.h5_model,
        args.onnx_model,
        args.onnx_slim_model,
        args.threshold,
    )

    return 0


if __name__ == "__main__":
    exit(main())
