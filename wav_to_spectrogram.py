"""
Convert WAV audio files to spectrogram images using sox.

This script processes audio files by splitting them into 12-second clips
and converting each clip to a 257x1000 grayscale spectrogram image.
"""

import argparse
import os
import subprocess
from pathlib import Path

import librosa


def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in seconds."""
    return librosa.get_duration(path=file_path)


def generate_spectrogram(
    input_wav: str,
    output_png: str,
    offset: float = 0,
    duration: float = 12,
) -> None:
    """
    Generate a spectrogram image from a WAV file using sox.

    Args:
        input_wav: Path to input WAV file
        output_png: Path to output PNG file
        offset: Start time in seconds
        duration: Duration in seconds
    """
    cmd = [
        "sox",
        "-V1",  # Verbose level 1
        input_wav,
        "-n",  # No audio output (spectrogram only)
        "trim", str(offset), f"{duration:.3f}",
        "remix", "1",  # Convert to mono
        "rate", "8k",  # Resample to 8kHz
        "spectrogram",
        "-x", "1000",  # Width in pixels
        "-y", "257",   # Height in pixels
        "-z", "90",    # Dynamic range in dB
        "-m",          # Monochrome
        "-r",          # Raw (no axes/legend)
        "-o", output_png,
    ]
    subprocess.run(cmd, check=True)


def process_wav_file(
    input_wav: str,
    output_dir: str,
    clip_duration: float = 12,
) -> list[str]:
    """
    Process a WAV file into multiple spectrogram images.

    Splits the audio into clips of the specified duration and generates
    a spectrogram for each clip.

    Args:
        input_wav: Path to input WAV file
        output_dir: Directory to save spectrogram images
        clip_duration: Duration of each clip in seconds

    Returns:
        List of paths to generated spectrogram images
    """
    # Get audio duration
    total_duration = get_audio_duration(input_wav)
    num_clips = int(total_duration / clip_duration)

    if num_clips == 0:
        print(f"Warning: {input_wav} is shorter than {clip_duration}s, skipping")
        return []

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Generate base name from input file
    base_name = Path(input_wav).stem

    output_files = []
    for i in range(num_clips):
        offset = i * clip_duration

        # Handle last clip (may be shorter)
        if offset + clip_duration > total_duration:
            duration = total_duration - offset
        else:
            duration = clip_duration

        # Generate output filename
        output_png = os.path.join(output_dir, f"{base_name}_part_{i+1:03d}.png")

        print(f"  Generating: {output_png}")
        generate_spectrogram(input_wav, output_png, offset, duration)
        output_files.append(output_png)

    return output_files


def process_directory(
    input_dir: str,
    output_dir: str,
    clip_duration: float = 12,
) -> list[str]:
    """
    Process all WAV files in a directory.

    Args:
        input_dir: Directory containing WAV files
        output_dir: Directory to save spectrogram images
        clip_duration: Duration of each clip in seconds

    Returns:
        List of paths to all generated spectrogram images
    """
    wav_files = list(Path(input_dir).glob("**/*.wav"))
    print(f"Found {len(wav_files)} WAV files")

    all_outputs = []
    for wav_file in wav_files:
        print(f"Processing: {wav_file}")
        outputs = process_wav_file(str(wav_file), output_dir, clip_duration)
        all_outputs.extend(outputs)

    return all_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Convert WAV files to spectrogram images using sox"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input WAV file or directory containing WAV files",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for spectrogram images",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=12,
        help="Duration of each clip in seconds (default: 12)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        print(f"Processing single file: {input_path}")
        outputs = process_wav_file(str(input_path), args.output_dir, args.clip_duration)
    elif input_path.is_dir():
        print(f"Processing directory: {input_path}")
        outputs = process_directory(str(input_path), args.output_dir, args.clip_duration)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1

    print(f"\nGenerated {len(outputs)} spectrogram images")
    return 0


if __name__ == "__main__":
    exit(main())
