#!/usr/bin/env python3
# equi_seam_roll.py
#
# Tasks:
#   1) extract  — export an image sequence from a video (PNG/JPG), at a fixed time interval
#   2) roll     — lossless horizontal wrap (seam shift) by yaw degrees or pixel shift
#
# Notes:
#   - "roll" is a pure pixel wrap (numpy.roll), so there's NO resampling or geometry change.
#   - For equirectangular 360: shift_px = round(width * yaw_deg / 360.0).
#   - Use a negative yaw to "roll back" to the original seam position.
#
# Examples:
#   # Extract one frame every 1.0 s to PNG
#   python equi_seam_roll.py extract --input "E:\pano\native.mp4" --out_dir "E:\pano\frames" --ext png --every_seconds 1.0
#
#   # Shift seam by +90° (i.e., move seam 90° to the right); for 7680x3840 → 1920 px
#   python equi_seam_roll.py roll --in_dir "E:\pano\frames" --out_dir "E:\pano\frames_yaw90" --yaw_deg 90
#
#   # After Topaz enhancement, roll back by -90° to original:
#   python equi_seam_roll.py roll --in_dir "E:\pano\frames_yaw90_enh" --out_dir "E:\pano\frames_final" --yaw_deg -90
#
# Optional:
#   - If you prefer, specify --shift_px directly instead of --yaw_deg.
#   - Set --workers >1 for parallel rolling on multi-core CPUs.

import argparse
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def natural_sort_key(p: Path):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', p.name)]


def extract_frames_every_seconds(
    input_path: Path,
    out_dir: Path,
    ext: str = "png",
    start_sec: float = 0.0,
    end_sec: float | None = None,
    every_seconds: float = 1.0,
    jpg_quality: int = 95,
    name_by_time: bool = False,
) -> None:
    """
    Extract frames at a fixed time interval (every_seconds).
    - Uses timestamps (POS_MSEC) so it works with VFR and CFR.
    - For speed, seeks forward to the next sample time after saving a frame.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"[extract] Cannot open input: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / fps) if (total_frames > 0 and fps > 0) else None
    if end_sec is None and duration is not None:
        end_sec = duration

    # Clamp
    if every_seconds <= 0:
        raise SystemExit("[extract] --every_seconds must be > 0")

    # Seek to start
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    saved = 0
    idx = 0
    next_sample_time = start_sec

    print(f"[extract] Input: {input_path}")
    print(f"[extract] Interval: {every_seconds}s  Range: [{start_sec}s, {end_sec if end_sec is not None else 'end'}s]")
    print(f"[extract] Output: {out_dir} ({ext})")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Stop by time
        if end_sec is not None and t_sec > end_sec + 1e-3:
            break

        # When current playback time crosses the next sample time, save
        if t_sec + 1e-6 >= next_sample_time:
            if name_by_time:
                # filename encodes the target sample time
                t_name = int(round(next_sample_time * 1000.0))  # ms
                out_name = f"t{t_name:010d}.{ext}"
            else:
                out_name = f"frame_{idx:06d}.{ext}"

            out_path = out_dir / out_name
            if ext.lower() == "jpg":
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            else:
                cv2.imwrite(str(out_path), frame)

            saved += 1
            idx += 1

            # Jump forward to next requested time (helps a ton on long clips)
            next_sample_time += every_seconds
            cap.set(cv2.CAP_PROP_POS_MSEC, next_sample_time * 1000.0)

    cap.release()
    print(f"[extract] Done. Saved {saved} frames.")


def _roll_one(src: Path, dst: Path, shift_px: int) -> tuple[Path, bool, str]:
    try:
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            return (src, False, "cv2.imread failed")
        rolled = np.roll(img, shift_px, axis=1)  # horizontal wrap
        ok = cv2.imwrite(str(dst), rolled)
        if not ok:
            return (src, False, "cv2.imwrite failed")
        return (src, True, "")
    except Exception as e:
        return (src, False, str(e))


def roll_frames(
    in_dir: Path,
    out_dir: Path,
    yaw_deg: float | None = None,
    shift_px: int | None = None,
    glob: str = "*.png",
    workers: int = 0,
) -> None:
    """
    Seam-shift (roll) all frames in in_dir → out_dir by yaw degrees or pixel shift.
    For equirectangular frames: shift_px = round(width * yaw_deg / 360).
    Positive shift moves pixels to the right (seam moves left).
    """
    if not in_dir.is_dir():
        raise SystemExit(f"[roll] Input directory not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(glob), key=natural_sort_key)
    if not files:
        # Try more extensions
        candidates = []
        for g in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.webp"):
            candidates += list(in_dir.glob(g))
        files = sorted(candidates, key=natural_sort_key)
    if not files:
        raise SystemExit(f"[roll] No images found in {in_dir}")

    # Read size from first image
    first = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise SystemExit(f"[roll] Failed to read first image: {files[0]}")
    H, W = first.shape[:2]

    if shift_px is None:
        if yaw_deg is None:
            raise SystemExit("[roll] Need either --yaw_deg or --shift_px")
        shift_px = int(round(W * (yaw_deg / 360.0)))
    # Normalize shift to shortest wrap
    if W > 0:
        shift_px = ((shift_px % W) + W) % W
        if shift_px > W // 2:
            shift_px -= W

    print(f"[roll] Found {len(files)} images; resolution = {W}x{H}")
    if yaw_deg is not None:
        print(f"[roll] yaw_deg={yaw_deg}  =>  shift_px={shift_px}")
    else:
        print(f"[roll] shift_px={shift_px}")

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for src in files:
                dst = out_dir / src.name
                futs.append(ex.submit(_roll_one, src, dst, shift_px))
            ok_count = 0
            for fut in as_completed(futs):
                src, ok, msg = fut.result()
                if not ok:
                    print(f"[roll] FAIL {src.name}: {msg}")
                else:
                    ok_count += 1
        print(f"[roll] Done. Wrote {ok_count}/{len(files)} frames to {out_dir}")
    else:
        ok_count = 0
        for i, src in enumerate(files, 1):
            dst = out_dir / src.name
            _, ok, msg = _roll_one(src, dst, shift_px)
            if not ok:
                print(f"[roll] FAIL {src.name}: {msg}")
            if i % 200 == 0:
                print(f"[roll] {i}/{len(files)}...")
            ok_count += int(ok)
        print(f"[roll] Done. Wrote {ok_count}/{len(files)} frames to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Equirect seam rotation via lossless horizontal roll + time-based extraction.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # extract
    ap_ex = sub.add_parser("extract", help="Extract frames every N seconds (time-based).")
    ap_ex.add_argument("--input", required=True, type=Path, help="Input video path")
    ap_ex.add_argument("--out_dir", required=True, type=Path, help="Output directory for frames")
    ap_ex.add_argument("--ext", default="png", choices=["png", "jpg"], help="Image format")
    ap_ex.add_argument("--start_sec", type=float, default=0.0, help="Start time (seconds)")
    ap_ex.add_argument("--end_sec", type=float, default=None, help="End time (seconds)")
    ap_ex.add_argument("--every_seconds", type=float, default=1.0, help="Interval (seconds) between extracted frames")
    ap_ex.add_argument("--jpg_quality", type=int, default=95, help="JPEG quality (if ext=jpg)")
    ap_ex.add_argument("--name_by_time", action="store_true", help="Name files by sample time (ms) instead of frame index")

    # roll
    ap_roll = sub.add_parser("roll", help="Lossless seam shift on an image folder.")
    ap_roll.add_argument("--in_dir", required=True, type=Path, help="Input image folder")
    ap_roll.add_argument("--out_dir", required=True, type=Path, help="Output image folder")
    grp = ap_roll.add_mutually_exclusive_group(required=True)
    grp.add_argument("--yaw_deg", type=float, help="Yaw degrees (+right/-left). shift_px = round(W*yaw/360)")
    grp.add_argument("--shift_px", type=int, help="Direct pixel shift (+right/-left)")
    ap_roll.add_argument("--glob", default="*.png", help="Filename glob (default: *.png)")
    ap_roll.add_argument("--workers", type=int, default=0, help="Parallel threads (0=serial)")

    args = ap.parse_args()

    if args.cmd == "extract":
        extract_frames_every_seconds(
            input_path=args.input,
            out_dir=args.out_dir,
            ext=args.ext,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            every_seconds=args.every_seconds,
            jpg_quality=args.jpg_quality,
            name_by_time=args.name_by_time,
        )
    elif args.cmd == "roll":
        roll_frames(
            in_dir=args.in_dir,
            out_dir=args.out_dir,
            yaw_deg=getattr(args, "yaw_deg", None),
            shift_px=getattr(args, "shift_px", None),
            glob=args.glob,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
