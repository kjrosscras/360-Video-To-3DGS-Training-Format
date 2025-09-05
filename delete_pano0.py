# delete_pano0.py
import shutil
import sys
from pathlib import Path

def delete_all_pano_camera0(source_dir: Path):
    deleted_folders = 0
    for cam0_folder in source_dir.glob("**/pano_camera0"):
        if cam0_folder.is_dir():
            print(f"🗑️  Deleting folder: {cam0_folder}")
            shutil.rmtree(cam0_folder)
            deleted_folders += 1
    if deleted_folders == 0:
        print("ℹ️  No 'pano_camera0' folders found.")
    else:
        print(f"✅ Deleted {deleted_folders} 'pano_camera0' folder(s).")

if __name__ == "__main__":
    # If an argument is provided, treat it as the project root (the folder you dropped).
    # We’ll delete <project_root>/output/images/**/pano_camera0
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).resolve()
        source = project_root / "output" / "images"
    else:
        # Fallback: current-dir/output/images (original behavior)
        source = Path("output/images").resolve()
    delete_all_pano_camera0(source)
