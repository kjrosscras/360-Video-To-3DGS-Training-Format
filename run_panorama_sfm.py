# run_panorama_sfm.py
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Wrapper to run panorama_sfm.py for a given project folder")
    p.add_argument("base_dir", nargs="?", default=".", help="Folder that contains 'frames/' (and will get 'output/').")
    p.add_argument("--input_image_path", type=Path, default=None,
                   help="Override path to frames dir (defaults to <base_dir>/frames)")
    p.add_argument("--output_path", type=Path, default=None,
                   help="Override path to output dir (defaults to <base_dir>/output)")
    args = p.parse_args()

    base_dir = Path(args.base_dir).resolve()
    input_dir = (args.input_image_path or (base_dir / "frames")).resolve()
    output_dir = (args.output_path or (base_dir / "output")).resolve()

    cmd = [
        sys.executable,  # use same interpreter as GUI
        str(Path(__file__).parent / "panorama_sfm.py"),
        "--input_image_path", str(input_dir),
        "--output_path", str(output_dir),
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
