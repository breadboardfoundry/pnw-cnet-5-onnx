"""
Download audio files from the PNW-Cnet dataset on Zenodo.

Downloads a zip file and extracts .wav files to the recordings/ directory.

Dataset: https://zenodo.org/records/10895837
"""

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

ZENODO_RECORD_ID = "10895837"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"

# Available dataset files
DATASET_FILES = {
    "part1": "additional_recordings_part_1.zip",
    "part2": "additional_recordings_part_2.zip",
    "part3": "additional_recordings_part_3.zip",
    "part4": "additional_recordings_part_4.zip",
}

DEFAULT_OUTPUT_DIR = "recordings"


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def download_file(url: str, output_path: Path) -> None:
    """Download a file with progress display."""
    print(f"Downloading: {output_path.name}")
    print(f"  URL: {url}")

    downloaded = [0]
    def progress_hook(block_num, block_size, total_size):
        downloaded[0] = min(block_num * block_size, total_size)
        if total_size > 0:
            percent = downloaded[0] * 100 / total_size
            print(f"\r  Progress: {percent:.1f}% ({format_size(downloaded[0])})", end="", flush=True)
        else:
            print(f"\r  Downloaded: {format_size(downloaded[0])}", end="", flush=True)

    try:
        urlretrieve(url, output_path, reporthook=progress_hook)
        print()
    except URLError as e:
        print(f"\nError: Download failed: {e}")
        sys.exit(1)


def extract_zip(zip_path: Path, output_dir: Path) -> int:
    """Extract .wav files from a zip archive."""
    print(f"Extracting .wav files from {zip_path.name}...")

    wav_count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(".wav"):
                # Extract to flat directory structure
                basename = Path(name).name
                target = output_dir / basename
                if not target.exists():
                    with zf.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    wav_count += 1

    print(f"  Extracted {wav_count} .wav files")
    return wav_count


def main():
    parser = argparse.ArgumentParser(
        description="Download audio files from the PNW-Cnet dataset on Zenodo"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dataset files",
    )
    parser.add_argument(
        "--part",
        type=str,
        choices=list(DATASET_FILES.keys()),
        default="part1",
        help="Which dataset part to download (default: part1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for .wav files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file after extraction",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        print("\nAvailable dataset parts:")
        print("-" * 50)
        for key, filename in DATASET_FILES.items():
            url = f"{BASE_URL}/{filename}?download=1"
            print(f"  {key}: {filename}")
            print(f"       {url}")
        print()
        return 0

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    filename = DATASET_FILES[args.part]
    url = f"{BASE_URL}/{filename}?download=1"
    zip_path = output_dir / filename

    if zip_path.exists():
        print(f"Zip file already exists: {zip_path}")
    else:
        download_file(url, zip_path)

    # Extract
    wav_count = extract_zip(zip_path, output_dir)

    # Clean up
    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()
        print(f"  Removed {filename}")

    print(f"\nDone! {wav_count} .wav files saved to: {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
