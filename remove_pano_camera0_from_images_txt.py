import argparse
from pathlib import Path
import shutil


def filter_images_txt(images_txt: Path, substring: str = "pano_camera0") -> None:
    """
    Remove all image entries from images.txt whose NAME contains `substring`.
    COLMAP images.txt format:
        - Comment / header lines start with '#'
        - Each image is represented by 2 lines:
            1) IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            2) POINTS2D[] as: X Y POINT3D_ID ...

    We must remove BOTH lines for each matching image.
    """

    images_txt = Path(images_txt)
    if not images_txt.exists():
        raise SystemExit(f"images.txt not found at: {images_txt}")

    backup_path = images_txt.with_suffix(".txt.bak")
    print(f"[INFO] Backing up original images.txt to: {backup_path}")
    shutil.copy2(images_txt, backup_path)

    with images_txt.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    out_lines = []
    i = 0
    removed_count = 0

    while i < len(lines):
        line = lines[i]

        # Preserve comment and blank lines as-is
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            i += 1
            continue

        # This should be an image header line. The next line should be its POINTS2D.
        header = line
        if i + 1 >= len(lines):
            # No second line, just keep header to avoid losing info
            out_lines.append(header)
            break

        points_line = lines[i + 1]

        parts = header.split()
        if len(parts) >= 10:
            image_id = parts[0]
            name = parts[9]
            if substring in name:
                # Skip this image entirely (header + points)
                removed_count += 1
                print(f"[REMOVE] IMAGE_ID {image_id}, NAME '{name}' (matched '{substring}')")
                i += 2
                continue

        # Otherwise, keep both lines
        out_lines.append(header)
        out_lines.append(points_line)
        i += 2

    # Write filtered file back
    with images_txt.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"[DONE] Filtered images.txt. Removed {removed_count} image(s) containing '{substring}'.")
    print(f"[INFO] Original file backed up at: {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove pano_camera0 images from COLMAP images.txt."
    )
    parser.add_argument(
        "--images_txt",
        type=Path,
        required=True,
        help="Path to COLMAP images.txt (e.g. .../sparse/0/images.txt)",
    )
    parser.add_argument(
        "--substring",
        type=str,
        default="pano_camera0",
        help="Substring to match in image file names (default: 'pano_camera0').",
    )
    args = parser.parse_args()

    filter_images_txt(args.images_txt, args.substring)


if __name__ == "__main__":
    main()
