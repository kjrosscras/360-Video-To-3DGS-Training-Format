import argparse
from pathlib import Path
from PIL import Image

SUPPORTED_EXTS = {".png", ".webp", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}

def has_alpha(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA"):
        return True
    if img.mode == "P" and ("transparency" in img.info):
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, help=r"Folder to scan (e.g. ...\output\images or ...\output\Segment_images)")
    ap.add_argument("--output_dir", required=True, help=r"COLMAP output root (e.g. ...\output) where masks/ will be created")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)

    if not input_root.exists():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")
    if not output_dir.exists():
        raise FileNotFoundError(f"output_dir does not exist: {output_dir}")

    masks_root = output_dir / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)

    total = 0
    wrote = 0
    skipped_no_alpha = 0
    skipped_non_image = 0
    skipped_is_mask = 0
    errors = 0

    for src_path in input_root.rglob("*"):
        if not src_path.is_file():
            continue
        if src_path.suffix.lower() not in SUPPORTED_EXTS:
            skipped_non_image += 1
            continue

        total += 1

        # avoid reprocessing any existing mask-like outputs
        if src_path.name.endswith(".mask.jpg") or src_path.stem.endswith(".mask"):
            skipped_is_mask += 1
            continue

        rel = src_path.relative_to(input_root)

        try:
            with Image.open(src_path) as img:
                # Palette transparency => RGBA
                if img.mode == "P" and ("transparency" in img.info):
                    img = img.convert("RGBA")

                if not has_alpha(img):
                    skipped_no_alpha += 1
                    continue

                if img.mode not in ("RGBA", "LA"):
                    img = img.convert("RGBA")

                alpha = img.getchannel("A")
                mask = alpha.point(lambda a: 255 if a > 0 else 0, mode="L")

                out_path = (masks_root / rel).with_suffix("")  # drop original suffix
                out_path = out_path.with_name(f"{out_path.name}.mask.jpg")  # add .mask.jpg
                out_path.parent.mkdir(parents=True, exist_ok=True)

                mask.save(out_path, format="JPEG", quality=95, optimize=True)
                wrote += 1

        except Exception as e:
            errors += 1
            print(f"[ERROR] {rel}: {e}")

    print("\n--- Summary ---")
    print(f"Input root         : {input_root}")
    print(f"Masks root         : {masks_root}")
    print(f"Scanned image files: {total}")
    print(f"Wrote masks        : {wrote}")
    print(f"Skipped (no alpha) : {skipped_no_alpha}")
    print(f"Skipped (is mask)  : {skipped_is_mask}")
    print(f"Skipped (non-img)  : {skipped_non_image}")
    print(f"Errors             : {errors}")
    print("\nOutput pattern: output/masks/<relative_path_from_input_root>/<name>.mask.jpg")

if __name__ == "__main__":
    main()
