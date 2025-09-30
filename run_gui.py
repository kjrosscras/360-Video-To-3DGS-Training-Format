import os
import sys
import json
import cv2
import time
import threading
import subprocess
from pathlib import Path

from tkinter import (
    Label, Entry, StringVar, Frame, Checkbutton, BooleanVar,
    filedialog, Button, Text, END, BOTH, DISABLED, NORMAL
)
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import ttk

import numpy as np
import shutil

# =========================
# Seam + Topaz config
# =========================
SEAM_YAW_DEG = 90.0                         # positive yaw moves seam left (view rotates right)
ENHANCED_SUBFOLDER = "frames_seam_yaw_enh"  # Topaz output folder (images)
SETTINGS_FILE = Path(__file__).with_name("settings_gui.json")

# Fixed target size (equirectangular 2:1)
TARGET_WIDTH = 8192
TARGET_HEIGHT = TARGET_WIDTH // 2

# =========================
# Angle profiles
# =========================
MASKING_PITCH_YAW_PAIRS = [
        (0, 90), #Reference Pose
        (33, 0),
        (-42, 0),
        (0, 42),
        (0, -27),
        (42, 180),
        (-33, 180),
        (0, 207),
        (0, 138),
        
    ]
MASKING_REF_IDX = 0

NO_MASKING_PITCH_YAW_PAIRS = [
        (0, 90), #Reference Pose
        (33, 0),
        (-42, 0),
        (0, 42),
        (0, -27),
        (42, 180),
        (-33, 180),
        (0, 207),
        (0, 138),
        
    ]
NO_MASKING_REF_IDX = 0

VALID_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# =========================
# Minimal settings (persist)
# =========================
DEFAULT_TOPAZ_CLI = {
    "use_topaz_cli": True,  # <— new: GUI toggle controls this
    "ffmpeg_path": r"C:\Program Files\Topaz Labs LLC\Topaz Video\ffmpeg.exe",
    # Paste the Topaz “Show Export Command” here; IO will be swapped to our sequence
    "topaz_cli_cmd": (
        r'ffmpeg "-hide_banner" '
        r'"-i" "E:/Test/Test/VID_20250924_125035_00_001.mp4" '
        r'"-sws_flags" "spline+accurate_rnd+full_chroma_int" '
        r'"-filter_complex" '
        r'"tvai_up=model=thm-2:scale=1:device=0:vram=1:instances=0,'
        r'tvai_up=model=amq-13:scale=0:w=8192:h=4096:device=0:vram=1:instances=1,'
        r'scale=w=8192:h=4096:flags=lanczos:threads=0" '
        r'"-c:v" "png" "-pix_fmt" "rgb48be" '
        r'"-start_number" "0" '
        r'"-metadata" "videoai=Motion blur removed using thm-2. Enhanced using amq-13. Changed resolution to 8192x4096" '
        r'"E:/Test/Test/VID_20250924_125035_00_001_thm2_amq13/%06d.png"'
    ),
    "model_dir":  r"C:\ProgramData\Topaz Labs LLC\Topaz Video\models",
    "model_data_dir": r"C:\ProgramData\Topaz Labs LLC\Topaz Video\models",
    "seconds_per_frame": "1",
    "use_masking": True,
}

def load_settings() -> dict:
    d = {}
    if SETTINGS_FILE.exists():
        try:
            d.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    for k, v in DEFAULT_TOPAZ_CLI.items():
        d.setdefault(k, v)
    return d

def save_settings(d: dict) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass

# =========================
# UI helpers
# =========================

_root = None
status_var = None
progress_main = None
progress_sub = None
log_text = None

use_masking = None
use_topaz_cli_var = None
frame_interval = None
drop_zone = None
run_btn = None
browse_btn = None

def ui_status(msg: str):
    status_var.set(msg); _root.update_idletasks()

def ui_log(msg: str):
    log_text.configure(state=NORMAL)
    log_text.insert(END, msg.rstrip() + "\n")
    log_text.see(END)
    log_text.configure(state=DISABLED)
    _root.update_idletasks()

def ui_main_progress(value: float | None = None, indeterminate: bool = False):
    try: progress_main.stop()
    except Exception: pass
    progress_main["mode"] = "indeterminate" if indeterminate else "determinate"
    if not indeterminate: progress_main["value"] = 0 if value is None else value
    if indeterminate: progress_main.start(12)
    _root.update_idletasks()

def ui_sub_progress(value: float | None = None, indeterminate: bool = False):
    try: progress_sub.stop()
    except Exception: pass
    progress_sub["mode"] = "indeterminate" if indeterminate else "determinate"
    if not indeterminate: progress_sub["value"] = 0 if value is None else value
    if indeterminate: progress_sub.start(12)
    _root.update_idletasks()

def ui_disable_inputs(disabled=True):
    state = DISABLED if disabled else NORMAL
    for w in (drop_zone, run_btn, browse_btn):
        w.configure(state=state)

# =========================
# Small utils
# =========================

def write_rotation_override(root_dir: Path, pairs, ref_idx: int) -> Path:
    payload = {"pitch_yaw_pairs": pairs, "ref_idx": ref_idx}
    out_path = root_dir / "rotation_override.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

def list_videos(video_dir: Path):
    return sorted([p for p in video_dir.glob("*") if p.suffix.lower() in VALID_VIDEO_EXT])

def list_images(img_dir: Path):
    return sorted([p for p in img_dir.glob("*") if p.suffix.lower() in VALID_IMAGE_EXT])

def natural_sort_key(p: Path):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', p.name)]

# Cleanup helpers
def _delete_dir_safe(path: Path, label: str = ""):
    try:
        if path.exists():
            shutil.rmtree(path)
            ui_log(f"[CLEAN] Removed {label or path.name} → {path}")
    except Exception as e:
        ui_log(f"[WARN] Could not remove {label or path.name} ({path}): {e}")

def _delete_if_empty(path: Path, label: str = ""):
    try:
        if path.exists() and not any(path.iterdir()):
            path.rmdir()
            ui_log(f"[CLEAN] Removed empty {label or path.name} → {path}")
    except Exception as e:
        ui_log(f"[WARN] Could not remove empty {label or path.name} ({path}): {e}")

# =========================
# Frame extraction
# =========================

def extract_frames_with_progress(video_dir: Path, interval_seconds: float) -> tuple[int, float]:
    """
    Extract frames from all videos directly inside video_dir into: video_dir/frames/
    Returns: (num_videos, fps_of_first_video_or_24.0)
    """
    output_base_dir = video_dir / "frames"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    vids = list_videos(video_dir)
    if not vids:
        ui_log(f"[WARN] No videos found in: {video_dir}")
        return 0, 24.0

    total_vids = len(vids)
    fps_first = 24.0

    for vid_idx, video_file in enumerate(vids, start=1):
        video_name = video_file.stem
        ui_status(f"Extracting frames: {video_file.name}")
        ui_log(f"[EXTRACT] {video_file.name}")

        cap = cv2.VideoCapture(str(video_file))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if vid_idx == 1 and fps > 0:
            fps_first = fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if not fps or fps <= 0:
            ui_log(f"[ERROR] Could not read FPS from {video_file.name}; defaulting to 24fps timing.")
            fps = 24.0

        step_frames = max(1, int(round(fps * float(interval_seconds))))
        frame_idx = 0
        saved_idx = 0

        ui_sub_progress(0, indeterminate=False)
        last_update = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step_frames == 0:
                frame_path = output_base_dir / f"{video_name}_frame_{saved_idx:05d}.png"
                cv2.imwrite(str(frame_path), frame)
                saved_idx += 1

            frame_idx += 1

            if frame_count > 0 and (time.time() - last_update) > 0.05:
                ui_sub_progress(min(100.0, 100.0 * frame_idx / max(1, frame_count)), indeterminate=False)
                last_update = time.time()

        cap.release()
        ui_sub_progress(100.0, indeterminate=False)
        ui_log(f"[OK] Saved {saved_idx} frames into {output_base_dir}")
        ui_main_progress(min(100.0, 100.0 * vid_idx / total_vids), indeterminate=False)

    return total_vids, fps_first

# =========================
# Seam roll (lossless)
# =========================

def _roll_image_horiz(img: np.ndarray, shift_px: int) -> np.ndarray:
    return np.roll(img, shift_px, axis=1)  # +shift -> pixels right (seam left)

def roll_folder_lossless(in_dir: Path, out_dir: Path, yaw_deg: float) -> tuple[int, int]:
    imgs = list_images(in_dir)
    if not imgs: return 0, 0
    out_dir.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(imgs[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        ui_log(f"[ERROR] Failed to read image: {imgs[0]}")
        return 0, 0
    H0, W0 = first.shape[:2]
    shift_px = int(round(W0 * (yaw_deg / 360.0)))

    ui_log(f"[SEAM] Rolling {len(imgs)} frame(s) by yaw={yaw_deg}° → shift_px={shift_px} (W={W0})")
    ui_status("Seam shifting frames (lossless wrap)…")
    ui_sub_progress(0, indeterminate=False)

    ok = 0
    last = time.time()
    for i, p in enumerate(imgs, 1):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            ui_log(f"[WARN] Read fail: {p.name}")
            continue
        H, W = img.shape[:2]
        if (H, W) != (H0, W0):
            ui_log(f"[ERROR] Resolution mismatch at {p.name}: got {W}x{H}, expected {W0}x{H0}")
            return ok, W0
        rolled = _roll_image_horiz(img, shift_px)
        out_path = out_dir / p.name
        if not cv2.imwrite(str(out_path), rolled):
            ui_log(f"[WARN] Write fail: {out_path.name}")
            continue
        ok += 1
        if (time.time() - last) > 0.05:
            ui_sub_progress(100.0 * i / len(imgs), indeterminate=False)
            last = time.time()

    ui_sub_progress(100.0, indeterminate=False)
    ui_log(f"[SEAM] Wrote {ok}/{len(imgs)} rolled frame(s) → {out_dir}")
    return ok, W0

def make_numeric_sequence(src_dir: Path, dst_dir: Path) -> int:
    """
    Copy images from src_dir into dst_dir as frame_%06d.png (contiguous).
    """
    files = list_images(src_dir)
    if not files:
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(sorted(files, key=natural_sort_key), 0):
        dst = dst_dir / f"frame_{i:06d}{p.suffix.lower()}"
        shutil.copy2(p, dst)
    ui_log(f"[SEQ] Reindexed {len(files)} images → {dst_dir}")
    return len(files)

# =========================
# YOLO masking (optional, writes PNG RGBA)
# =========================

def run_yolo_masking(frames_root: Path) -> int:
    try:
        from ultralytics import YOLO
        import numpy as np
        import cv2
        import torch, time, os
    except Exception:
        ui_log("[ERROR] YOLO/torch/OpenCV not available.")
        ui_log("       pip install ultralytics opencv-python torch torchvision torchaudio")
        return 0

    model_name  = "yolov8x-seg.pt"
    conf, iou, imgsz = 0.35, 0.45, 1024
    classes, grow_px = [0], 30  # person + dilation

    def morph_expand(mask_bool: np.ndarray, grow: int) -> np.ndarray:
        if grow == 0: return mask_bool
        mask = (mask_bool.astype(np.uint8) * 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        step = 3
        iters = max(1, int(abs(grow) / step))
        out = cv2.dilate(mask, k, iterations=iters) if grow > 0 else cv2.erode(mask, k, iterations=iters)
        return (out > 0)

    jpgs  = sorted(frames_root.glob("*.jpg")) + sorted(frames_root.glob("*.jpeg"))
    pngs  = sorted(frames_root.glob("*.png"))
    tifs  = sorted(frames_root.glob("*.tif")) + sorted(frames_root.glob("*.tiff"))

    stems_png = {p.stem for p in pngs}
    img_paths = [p for p in jpgs + tifs if p.stem not in stems_png] + pngs

    if not img_paths:
        ui_log(f"[WARN] No frames found in {frames_root} (.jpg/.png/.tif/.tiff)")
        return 0

    device = "cuda" if 'torch' in sys.modules and hasattr(sys.modules['torch'], 'cuda') and sys.modules['torch'].cuda.is_available() else "cpu"
    ui_log(f"[YOLO] Loading {model_name} on {device} …")
    model = YOLO(model_name)

    total, processed = len(img_paths), 0
    ui_status("Masking people into alpha (RGBA)…")
    ui_sub_progress(0, indeterminate=False)
    last_update = time.time()

    for i, img_path in enumerate(img_paths, 1):
        try:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Failed to read image: {img_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            results = model.predict(
                source=rgb, conf=conf, iou=iou, imgsz=imgsz,
                classes=classes, verbose=False, device=None
            )
            alpha = np.full((h, w), 255, dtype=np.uint8)
            if results and results[0].masks is not None and len(results[0].masks) > 0:
                masks_tensor = results[0].masks.data.cpu().numpy()
                person_small = (masks_tensor.max(axis=0) > 0.5).astype(np.uint8)
                person = cv2.resize(person_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                person = morph_expand(person, grow_px)
                alpha[person] = 0

            rgba = np.dstack([rgb, alpha])
            out_png = frames_root / f"{img_path.stem}.png"
            cv2.imwrite(str(out_png), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            if img_path.suffix.lower() != ".png" and out_png.exists():
                try: os.remove(str(img_path))
                except Exception: pass
            processed += 1

        except Exception as e:
            ui_log(f"[ERROR] RGBA write {img_path.name}: {e}")

        if (time.time() - last_update) > 0.05:
            ui_sub_progress(min(100.0, 100.0 * i / total), indeterminate=False)
            last_update = time.time()

    ui_sub_progress(100.0, indeterminate=False)
    ui_log(f"[OK] RGBA conversion: {processed}/{total} → PNG with alpha in {frames_root}")
    return processed

# =========================
# COLMAP wrapper (stream logs)
# =========================

def run_panorama_sfm(project_root: Path, images_dir: Path) -> bool:
    """
    Stream stdout/stderr live into the GUI log using images_dir.
    No matcher is passed; panorama_sfm.py will use its own default.
    """
    panorama = Path(__file__).parent / "panorama_sfm.py"
    wrapper  = Path(__file__).parent / "run_panorama_sfm.py"  # fallback

    env = os.environ.copy()

    if panorama.exists():
        cmd = [sys.executable, "-u", str(panorama),
               "--input_image_path", str(images_dir),
               "--output_path",      str(project_root / "output")]
    elif wrapper.exists():
        env["COLMAP_INPUT_IMAGE_PATH"] = str(images_dir)
        cmd = [sys.executable, "-u", str(wrapper), str(project_root)]
        ui_log("[WARN] panorama_sfm.py not found; calling run_panorama_sfm.py (uses COLMAP_INPUT_IMAGE_PATH).")
    else:
        ui_log("[ERROR] Neither panorama_sfm.py nor run_panorama_sfm.py found.")
        return False

    ui_log(f"[RUN] {' '.join(cmd)}")
    ui_status("Running COLMAP pipeline…")
    ui_sub_progress(indeterminate=True)

    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        ) as proc:
            if proc.stdout is not None:
                for line in proc.stdout:
                    if line:
                        ui_log(line.rstrip())
            ret = proc.wait()

        ui_sub_progress(100, indeterminate=False)
        if ret == 0:
            ui_log("[OK] panorama_sfm finished.")
            return True
        else:
            ui_log(f"[ERROR] panorama_sfm exited with code {ret}.")
            return False
    except Exception as e:
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[ERROR] panorama_sfm failed: {e}")
        return False


def delete_pano_camera0(project_root: Path) -> None:
    deleter = Path(__file__).parent / "delete_pano0.py"
    if not deleter.exists():
        ui_log(f"[WARN] Missing delete_pano0.py (skipping): {deleter}")
        return
    cmd = [sys.executable, str(deleter), str(project_root)]
    ui_log(f"[RUN] {' '.join(cmd)}")
    ui_status("Deleting pano_camera0 folders…")
    ui_sub_progress(indeterminate=True)
    try:
        subprocess.run(cmd, check=True)
        ui_sub_progress(100, indeterminate=False)
        ui_log("[OK] Deleted all pano_camera0 folders.")
    except subprocess.CalledProcessError as e:
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[WARN] delete_pano0.py reported an error: {e}")

def run_segment_images(project_root: Path) -> bool:
    seg = Path(__file__).parent / "segment_images.py"
    if not seg.exists():
        ui_log(f"[WARN] Missing segment_images.py (skipping): {seg}")
        return False
    cmd = [sys.executable, str(seg)]
    ui_log(f"[RUN] {' '.join(cmd)} (cwd={project_root})")
    ui_status("Segmenting COLMAP images by clip prefix…")
    ui_sub_progress(indeterminate=True)
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        ui_sub_progress(100, indeterminate=False)
        ui_log("[OK] Segmented images (see Segment_images/).")
        return True
    except subprocess.CalledProcessError as e:
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[WARN] segment_images.py reported an error: {e}")
        return False

# =========================
# Topaz headless CLI
# =========================

def run_topaz_via_cli_command(raw_cmd: str,
                              ffmpeg_path: str,
                              model_dir: str,
                              model_data_dir: str,
                              seq_in_dir: Path,
                              seq_out_dir: Path,
                              fps: float = 24.0) -> bool:
    """
    Use Topaz's copied ffmpeg command, but:
      - force our ffmpeg exe first
      - normalize tokens (strip quotes)
      - remove all pre-existing -i inputs and stray -hide_banner/-framerate/-start_number
      - inject: -hide_banner -framerate <fps> -start_number 0 -i <sequence>
      - replace the final output with our %06d.<ext> (ext parsed from raw command)
    """
    import shlex, re

    imgs = sorted(seq_in_dir.glob("frame_*.png"))
    if not imgs:
        ui_log(f"[ERROR] No frames in {seq_in_dir}")
        return False
    seq_out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if model_dir:
        env["TVAI_MODEL_DIR"] = model_dir
    if model_data_dir:
        env["TVAI_MODEL_DATA_DIR"] = model_data_dir

    in_pat = str(seq_in_dir / "frame_%06d.png")

    # parse tokens
    try:
        tokens = shlex.split(raw_cmd, posix=False)
    except Exception as e:
        ui_log(f"[ERROR] Could not parse Topaz CLI command: {e}")
        return False

    def norm(tok: str) -> str:
        return tok.strip().strip('"').strip("'")

    tokens = [norm(t) for t in tokens]

    def is_flag(tok: str, name: str) -> bool:
        return tok.lower() == name.lower()

    def is_ffmpeg(tok: str) -> bool:
        t = tok.replace("\\", "/").lower()
        return t.endswith("/ffmpeg.exe") or t.endswith("/ffmpeg") or t == "ffmpeg"

    # detect output extension
    raw = " ".join(tokens)
    m = re.search(r'%0?\d+d\.(png|tif|tiff|bmp|jpg|jpeg)', raw, flags=re.IGNORECASE)
    out_ext = (m.group(1).lower() if m else "tiff")
    if out_ext == "tif":
        out_ext = "tiff"
    out_pat = str(seq_out_dir / f"%06d.{out_ext}")

    # drop any ffmpeg token anywhere
    tokens = [t for t in tokens if not is_ffmpeg(t)]

    # remove ALL pre-existing inputs (-i ARG)
    cleaned = []
    skip = 0
    for i, t in enumerate(tokens):
        if skip:
            skip -= 1
            continue
        if is_flag(t, "-i"):
            skip = 1
            continue
        cleaned.append(t)
    tokens = cleaned

    # remove stray -hide_banner / -framerate <arg> / -start_number <arg>
    cleaned = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if is_flag(t, "-hide_banner"):
            i += 1; continue
        if is_flag(t, "-framerate") and i + 1 < len(tokens):
            i += 2; continue
        if is_flag(t, "-start_number") and i + 1 < len(tokens):
            i += 2; continue
        cleaned.append(t); i += 1
    tokens = cleaned

    # replace final output (last non-flag token) with our out_pat
    def replace_output(tt: list[str]) -> list[str]:
        if not tt:
            return [out_pat]
        j = len(tt) - 1
        while j >= 0 and tt[j].startswith("-"):
            j -= 1
        if j >= 0:
            tt[j] = out_pat
            return tt
        tt.append(out_pat); return tt

    tokens = replace_output(tokens)

    # build final command
    final_tokens = [
        ffmpeg_path, "-hide_banner",
        "-framerate", f"{fps:.6f}",
        "-start_number", "0",
        "-i", in_pat,
    ] + tokens

    ui_log("[TOPAZ/CLI] " + " ".join(final_tokens))
    ui_status("Running Topaz export (headless)…")
    ui_sub_progress(indeterminate=True)
    try:
        proc = subprocess.run(final_tokens, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        ui_sub_progress(100, indeterminate=False)
        if proc.returncode == 0:
            first = next(iter(sorted(seq_out_dir.glob(f"*.{out_ext}"))), None)
            if first:
                img = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    ui_log(f"[VERIFY] First enhanced frame: {w}x{h} ({first.name})")
            ui_log("[OK] Topaz CLI export finished.")
            return True
        else:
            ui_log(proc.stdout or "")
            ui_log(f"[ERROR] Topaz CLI returned {proc.returncode}.")
            return False
    except FileNotFoundError:
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[ERROR] ffmpeg not found: {ffmpeg_path}")
        return False
    except Exception as e:
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[ERROR] Failed to run Topaz CLI: {e}")
        return False

# =========================
# Pipeline (thread)
# =========================

def pipeline_thread(project_root: Path, seconds_per_frame: float, masking_enabled: bool, use_topaz: bool, cfg: dict):
    # Predefine working paths (for cleanup)
    frames_root      = project_root / "frames"
    rolled_root      = project_root / "frames_seam_yaw"
    seq_root         = project_root / "frames_seam_yaw_seq"
    enh_root         = project_root / ENHANCED_SUBFOLDER
    final_frames_dir = project_root / "frames_final"
    output_dir       = project_root / "output"

    try:
        ui_disable_inputs(True)
        ui_main_progress(0, indeterminate=False)

        # A) Extract frames
        ui_status("Preparing extraction…")
        video_count, fps_first = extract_frames_with_progress(project_root, seconds_per_frame)
        if not list_images(frames_root):
            ui_log("[ERROR] No frames extracted. Aborting."); ui_status("Idle."); return
        ui_main_progress(10, indeterminate=False)

        # B) If Topaz enabled → seam roll → sequence → Topaz → roll back
        if use_topaz:
            # Seam shift
            cnt, in_width = roll_folder_lossless(frames_root, rolled_root, SEAM_YAW_DEG)
            if cnt == 0:
                ui_log("[ERROR] Seam roll produced no images. Aborting."); ui_status("Idle."); return
            # Clean early: raw frames are no longer needed once rolled frames exist
            _delete_dir_safe(frames_root, "raw extracted frames")
            ui_main_progress(15, indeterminate=False)

            # Numeric sequence for Topaz
            nseq = make_numeric_sequence(rolled_root, seq_root)
            if nseq == 0:
                ui_log("[ERROR] Failed to create numeric sequence for Topaz."); ui_status("Idle."); return

            # Run Topaz
            ok = run_topaz_via_cli_command(
                raw_cmd=cfg.get("topaz_cli_cmd", DEFAULT_TOPAZ_CLI["topaz_cli_cmd"]),
                ffmpeg_path=cfg.get("ffmpeg_path", DEFAULT_TOPAZ_CLI["ffmpeg_path"]),
                model_dir=cfg.get("model_dir", DEFAULT_TOPAZ_CLI["model_dir"]),
                model_data_dir=cfg.get("model_data_dir", DEFAULT_TOPAZ_CLI["model_data_dir"]),
                seq_in_dir=seq_root,
                seq_out_dir=enh_root,
                fps=fps_first if fps_first > 0 else 24.0
            )
            if not ok:
                ui_status("Idle."); return
            # Clean: numeric sequence can go now
            _delete_dir_safe(seq_root, "Topaz numeric sequence")
            ui_main_progress(35, indeterminate=False)

            # Roll back seam → frames_final
            cnt_back, _ = roll_folder_lossless(enh_root, final_frames_dir, -SEAM_YAW_DEG)
            if cnt_back == 0:
                ui_log("[ERROR] Roll-back produced no images. Aborting."); ui_status("Idle."); return
            # Clean: rolled and enhanced intermediates can go now
            _delete_dir_safe(rolled_root, "seam-shifted frames")
            _delete_dir_safe(enh_root, "Topaz enhanced frames")
            ui_main_progress(50, indeterminate=False)

            images_dir_for_next = final_frames_dir

        else:
            ui_log("[TOPAZ] Skipped Topaz enhancement (checkbox off).")
            images_dir_for_next = frames_root  # use raw extracted frames directly
            # Ensure final_frames_dir mirrors current working dir (so downstream paths stay consistent)
            # We won't duplicate files; masking & COLMAP will read from images_dir_for_next.

        # C) Masking (writes PNG with alpha in-place at images_dir_for_next)
        if masking_enabled:
            ui_status("Writing rotation override (masking)…")
            write_rotation_override(project_root, MASKING_PITCH_YAW_PAIRS, MASKING_REF_IDX)
            ui_log(f"[OK] Wrote rotation_override.json (masking) in {project_root}")
            ui_status("Masking frames (RGBA alpha)…")
            processed = run_yolo_masking(images_dir_for_next)
            if processed == 0:
                ui_log("[WARN] No frames were masked (or masking failed). Continuing…")
        else:
            ui_status("Writing rotation override (no masking)…")
            write_rotation_override(project_root, NO_MASKING_PITCH_YAW_PAIRS, NO_MASKING_REF_IDX)
            ui_log(f"[OK] Wrote rotation_override.json (no masking) in {project_root}")

        ui_main_progress(65, indeterminate=False)

        # D) COLMAP on the chosen folder
        output_dir.mkdir(parents=True, exist_ok=True)

        if not run_panorama_sfm(project_root, images_dir_for_next):
         ui_status("COLMAP failed. See log."); return


        ui_main_progress(85, indeterminate=False)

        # E) Delete pano_camera0
        delete_pano_camera0(project_root)
        ui_main_progress(92, indeterminate=False)

        # F) Segment images if multiple input videos
        if video_count > 1:
            ui_log(f"[INFO] Multiple videos detected ({video_count}). Running segment_images…")
            run_segment_images(project_root)

        # G) Final cleanup: delete EVERYTHING except output/ and original video(s)
        ui_status("Final cleanup…")

        KEEP_DIRS  = {"output"}
        KEEP_EXTS  = {ext.lower() for ext in VALID_VIDEO_EXT}

        for child in project_root.iterdir():
            if child.is_dir():
                if child.name.lower() in KEEP_DIRS:
                    continue
                _delete_dir_safe(child, f"folder {child.name}")
            else:
                if child.suffix.lower() in KEEP_EXTS:
                    continue
                try:
                    child.unlink()
                    ui_log(f"[CLEAN] Removed file → {child}")
                except Exception as e:
                    ui_log(f"[WARN] Could not remove file {child}: {e}")

        ui_main_progress(100, indeterminate=False)
        ui_status("All done.")
        ui_log("[DONE] Pipeline complete. Kept original video(s) and the 'output' folder.")

    finally:
        try:
            progress_main.stop(); progress_sub.stop()
        except Exception: pass
        ui_disable_inputs(False)

# =========================
# GUI
# =========================

_last_folder = None

def start_pipeline_with_path(folder_path: Path):
    global _last_folder
    _last_folder = folder_path
    try:
        seconds = float(frame_interval.get())
        if seconds <= 0: raise ValueError
    except Exception:
        seconds = 1.0; frame_interval.set("1")

    cfg = load_settings()
    cfg["seconds_per_frame"] = seconds
    cfg["use_masking"] = bool(use_masking.get())
    cfg["use_topaz_cli"] = bool(use_topaz_cli_var.get())
    save_settings(cfg)

    threading.Thread(
        target=pipeline_thread,
        args=(folder_path, seconds, use_masking.get(), use_topaz_cli_var.get(), cfg),
        daemon=True
    ).start()

def on_drop(event):
    folder_path = Path(event.data.strip("{}"))
    if folder_path.is_dir():
        run_btn.configure(state=NORMAL)
        start_pipeline_with_path(folder_path)
    else:
        ui_log("[ERROR] Please drop a valid folder.")

def browse_folder():
    chosen = filedialog.askdirectory(title="Select folder containing 360 video(s)")
    if chosen:
        run_btn.configure(state=NORMAL)
        start_pipeline_with_path(Path(chosen))

def run_last():
    if _last_folder and Path(_last_folder).exists():
        start_pipeline_with_path(Path(_last_folder))
    else:
        ui_log("[ERROR] No valid last folder. Please Browse or Drop a folder.")

def main():
    global _root, status_var, progress_main, progress_sub, log_text
    global use_masking, use_topaz_cli_var, frame_interval, drop_zone, run_btn, browse_btn

    s = load_settings()

    _root = TkinterDnD.Tk()
    _root.title("360 Video → (Optional Topaz) → Training Pipeline")
    _root.geometry("920x820")
    _root.configure(bg="black")
    _root.resizable(False, False)

    # Header
    header = Frame(_root, bg="black"); header.pack(pady=(14, 8))
    Label(header, text="📁", font=("Arial", 44), bg="black", fg="white").pack(side="left", padx=(0, 12))
    Label(header,
          text="Insta360 Video(s) → Optional Topaz (CLI) → Mask → COLMAP",
          bg="black", fg="white", font=("Helvetica", 16, "bold")).pack(side="left")

    # Controls
    ctrl = Frame(_root, bg="black"); ctrl.pack(pady=(8, 2))
    Label(ctrl, text="Extract 1 frame per", bg="black", fg="white").pack(side="left")
    frame_interval = StringVar(value=str(s.get("seconds_per_frame", "1")))
    Entry(ctrl, textvariable=frame_interval, width=6).pack(side="left", padx=(6, 6))
    Label(ctrl, text="seconds", bg="black", fg="white").pack(side="left", padx=(0, 16))

    use_topaz_cli_var = BooleanVar(value=bool(s.get("use_topaz_cli", True)))
    Checkbutton(
        ctrl, text="Use Topaz enhancement", variable=use_topaz_cli_var,
        onvalue=True, offvalue=False, bg="black", fg="white",
        activebackground="black", selectcolor="black"
    ).pack(side="left", padx=(0, 16))

    use_masking = BooleanVar(value=bool(s.get("use_masking", True)))
    Checkbutton(
        ctrl, text="Enable Masking (remove people via alpha)",
        variable=use_masking, onvalue=True, offvalue=False,
        bg="black", fg="white", activebackground="black", selectcolor="black"
    ).pack(side="left", padx=(0,16))

    # Show Topaz CLI settings (read-only hints)
    Label(_root, text=f"Topaz ffmpeg: {s.get('ffmpeg_path')}", bg="black", fg="#666").pack()
    Label(_root, text=f"Model dir:    {s.get('model_dir')}", bg="black", fg="#666").pack()
    Label(_root, text=f"Target: {TARGET_WIDTH}×{TARGET_HEIGHT}  |  Seam yaw: {SEAM_YAW_DEG:g}°", bg="black", fg="#aaa").pack()

    # Drop zone
    global drop_zone, run_btn, browse_btn
    drop_zone = Label(
        _root, text="Drop Folder With Insta360 Video(s) Here",
        bg="#222", fg="white", width=92, height=6,
        relief="ridge", bd=2
    )
    drop_zone.pack(pady=10)
    drop_zone.drop_target_register(DND_FILES)
    drop_zone.dnd_bind('<<Drop>>', on_drop)

    # Buttons
    btns = Frame(_root, bg="black"); btns.pack()
    browse_btn = Button(btns, text="Browse…", command=browse_folder); browse_btn.pack(side="left", padx=6)
    run_btn = Button(btns, text="Run Last Chosen Folder", state=DISABLED, command=run_last); run_btn.pack(side="left", padx=6)

    # Progress
    prog = Frame(_root, bg="black"); prog.pack(fill=BOTH, padx=16, pady=(12, 4))
    Label(prog, text="Overall Progress", bg="black", fg="#ccc").pack(anchor="w")
    global progress_main, progress_sub
    progress_main = ttk.Progressbar(prog, orient="horizontal", mode="determinate", length=840); progress_main.pack(pady=(2, 8))
    Label(prog, text="Current Task", bg="black", fg="#ccc").pack(anchor="w")
    progress_sub = ttk.Progressbar(prog, orient="horizontal", mode="determinate", length=840); progress_sub.pack(pady=(2, 8))

    global status_var, log_text
    status_var = StringVar(value="Idle.")
    Label(_root, textvariable=status_var, bg="black", fg="white").pack(pady=(0, 6))

    log_frame = Frame(_root, bg="black"); log_frame.pack(fill=BOTH, expand=True, padx=16, pady=(0, 12))
    log_text = Text(log_frame, height=16, bg="#111", fg="#ddd", insertbackground="white")
    log_text.pack(fill=BOTH, expand=True)
    log_text.configure(state=DISABLED)

    _root.mainloop()

if __name__ == "__main__":
    main()
