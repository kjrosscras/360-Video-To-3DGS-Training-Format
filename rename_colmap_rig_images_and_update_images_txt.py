import argparse
import re
from pathlib import Path

PANO_RE = re.compile(r"(^|/|\\)pano_camera(\d+)(/|\\|$)", re.IGNORECASE)

def infer_pano_idx(image_rel_path: str) -> int | None:
    m = PANO_RE.search(image_rel_path)
    if not m:
        return None
    return int(m.group(2))

def make_new_name(image_rel_path: str, pano_idx: int) -> str:
    p = Path(image_rel_path)
    # keep parent folders, only change filename
    new_filename = f"{p.stem}_pano_camera_{pano_idx}{p.suffix}"
    return str(p.with_name(new_filename)).replace("\\", "/")  # COLMAP usually uses forward slashes

def parse_images_txt_lines(lines: list[str]):
    """
    Yields tuples (i, line1, line2) for each image entry.
    COLMAP images.txt format:
      - comment lines start with '#'
      - image entries are 2 lines:
        line1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        line2: POINTS2D...
    """
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("#") or not line.strip():
            i += 1
            continue

        # Expect entry line + points line
        if i + 1 >= n:
            raise ValueError(f"Unexpected EOF: image entry at line {i+1} missing points2D line.")

        yield i, lines[i], lines[i + 1]
        i += 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_txt", required=True, help=r"Path to images.txt (e.g. E:\...\sparse\0\images.txt)")
    ap.add_argument("--images_root", required=True, help=r"Folder where the images live (e.g. E:\...\images)")
    ap.add_argument("--dry_run", action="store_true", help="Print planned changes without writing/renaming")
    args = ap.parse_args()

    images_txt = Path(args.images_txt)
    images_root = Path(args.images_root)

    if not images_txt.exists():
        raise FileNotFoundError(f"images.txt not found: {images_txt}")
    if not images_root.exists():
        raise FileNotFoundError(f"images_root not found: {images_root}")

    lines = images_txt.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    # Map old_rel -> new_rel (for rewriting images.txt)
    rename_pairs: list[tuple[str, str]] = []

    # We'll also rename files on disk (old_abs -> new_abs)
    fs_renames: list[tuple[Path, Path]] = []

    for idx, line1, line2 in parse_images_txt_lines(lines):
        parts = line1.strip().split()
        if len(parts) < 10:
            raise ValueError(f"Malformed image entry line at {idx+1}: {line1}")

        # NAME is the last token
        old_name = parts[-1]
        pano_idx = infer_pano_idx(old_name)
        if pano_idx is None:
            continue  # not a pano_camera path; leave untouched

        new_name = make_new_name(old_name, pano_idx)
        if new_name == old_name:
            continue

        old_abs = images_root / Path(old_name)
        new_abs = images_root / Path(new_name)

        rename_pairs.append((old_name, new_name))
        fs_renames.append((old_abs, new_abs))

    if not rename_pairs:
        print("No pano_camera images found to rename (nothing to do).")
        return

    # Check for collisions / missing files
    problems = 0
    seen_targets = set()
    for old_abs, new_abs in fs_renames:
        if not old_abs.exists():
            print(f"[WARN] Missing file on disk (will still update images.txt if you proceed): {old_abs}")
            # not necessarily fatal; some people keep images elsewhere
        if new_abs in seen_targets:
            print(f"[ERROR] Two sources map to the same target: {new_abs}")
            problems += 1
        seen_targets.add(new_abs)

        if new_abs.exists() and old_abs.exists() and old_abs.resolve() != new_abs.resolve():
            print(f"[ERROR] Target already exists (would overwrite): {new_abs}")
            problems += 1

    if problems:
        raise RuntimeError("Aborting due to rename collisions / existing targets. Fix the issues above and re-run.")

    # Show plan
    print("--- Planned renames ---")
    for old_name, new_name in rename_pairs[:25]:
        print(f"{old_name}  ->  {new_name}")
    if len(rename_pairs) > 25:
        print(f"... and {len(rename_pairs) - 25} more")

    if args.dry_run:
        print("\nDRY RUN: no files renamed, images.txt not written.")
        return

    # 1) Rename files on disk (best-effort; if file missing, we skip rename but still update images.txt)
    renamed_ok = 0
    renamed_skipped_missing = 0

    for old_abs, new_abs in fs_renames:
        if not old_abs.exists():
            renamed_skipped_missing += 1
            continue
        new_abs.parent.mkdir(parents=True, exist_ok=True)
        old_abs.rename(new_abs)
        renamed_ok += 1

    # 2) Rewrite images.txt NAME tokens
    #    We reconstruct line1 per entry to avoid accidental partial string replaces.
    name_map = dict(rename_pairs)

    out_lines = lines[:]  # copy
    for idx, line1, line2 in parse_images_txt_lines(lines):
        parts = line1.strip().split()
        old_name = parts[-1]
        if old_name not in name_map:
            continue
        parts[-1] = name_map[old_name]

        # Preserve original newline ending
        newline = "\n" if not line1.endswith("\r\n") else "\r\n"
        if line1.endswith("\r\n"):
            newline = "\r\n"
        elif line1.endswith("\n"):
            newline = "\n"
        else:
            newline = "\n"

        out_lines[idx] = " ".join(parts) + newline

    backup = images_txt.with_suffix(images_txt.suffix + ".bak")
    backup.write_text("".join(lines), encoding="utf-8")
    images_txt.write_text("".join(out_lines), encoding="utf-8")

    print("\n--- Done ---")
    print(f"Renamed files: {renamed_ok}")
    print(f"Skipped rename (missing on disk): {renamed_skipped_missing}")
    print(f"Updated images.txt: {images_txt}")
    print(f"Backup created: {backup}")
    print("\nNote: If you plan to use COLMAP's database workflow again, you'd also need to keep the DB consistent with filenames.")

if __name__ == "__main__":
    main()
