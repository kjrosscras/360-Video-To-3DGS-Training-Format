import os
import shutil
from pathlib import Path
from collections import defaultdict

# === CONFIGURATION ===
SOURCE_DIR = Path("output/images")      # Where COLMAP's images are
DEST_DIR = Path("output/Segment_images")       # Where segmented groups go
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}  # Add/remove if needed

def get_prefix(filename):
    """Returns the prefix before the first underscore, e.g., 'Inside_0001.jpg' -> 'Inside'."""
    return filename.split("_")[0]

def collect_segment_images(source_dir):
    """Collects all valid images from pano_camera folders, organized by prefix."""
    segment_map = defaultdict(list)

    for cam_folder in sorted(source_dir.glob("pano_camera*")):
        if not cam_folder.is_dir():
            continue
        print(f"📂 Scanning {cam_folder.name}...")

        for img_path in sorted(cam_folder.glob("*")):
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            prefix = get_prefix(img_path.name)
            segment_map[prefix].append((cam_folder.name, img_path))

    return segment_map

def copy_segments(segment_map, dest_dir):
    count = 0
    for prefix, items in segment_map.items():
        for cam_folder_name, src_path in items:
            dest_path = dest_dir / prefix / cam_folder_name / src_path.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            count += 1
    print(f"\n✅ Copied {count} files into '{dest_dir}'.")

def delete_all_pano_camera0(source_dir):
    deleted_folders = 0
    for cam0_folder in source_dir.glob("**/pano_camera0"):
        if cam0_folder.is_dir():
            print(f"🗑️  Deleting folder: {cam0_folder}")
            shutil.rmtree(cam0_folder)
            deleted_folders += 1
    if deleted_folders == 0:
        print("ℹ️  No 'pano_camera0' folders found.")
    else:
        print(f"🗑️  Deleted {deleted_folders} 'pano_camera0' folders.")

if __name__ == "__main__":
    segments = collect_segment_images(SOURCE_DIR)
    copy_segments(segments, DEST_DIR)
    delete_all_pano_camera0(SOURCE_DIR)
