import os

# =========================
# PyInstaller / OpenMP runtime guard
# =========================
# In packaged builds, libraries like NumPy/MKL, OpenCV, Torch/Ultralytics,
# and pycolmap can load different OpenMP runtimes in the same process.
# Without this, the app can abort at the COLMAP step with:
#   OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
#
# KMP_DUPLICATE_LIB_OK is an unsafe workaround according to OpenMP's warning,
# but it is commonly used to keep packaged desktop apps from aborting.
# The thread limits reduce the chance of runtime conflicts and oversubscription.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import json
import cv2
import time
import threading
import subprocess
import webbrowser
import runpy
import io
import contextlib
import traceback
import faulthandler
from pathlib import Path

from tkinter import (
    Label, Entry, StringVar, Frame, Checkbutton, BooleanVar,
    filedialog, Button, Text, END, BOTH, DISABLED, NORMAL, Radiobutton, Toplevel
)
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import ttk

import numpy as np
import shutil

# =========================
# Seam + Topaz config
# =========================
SEAM_YAW_DEG = 90.0
ENHANCED_SUBFOLDER = "frames_seam_yaw_enh"
SETTINGS_FILE = Path(__file__).with_name("settings_gui.json")

# Replace this with your actual tutorial/how-to webpage URL.
HOW_TO_URL = "https://www.vrestateviewings.com/user-guide"

APP_ICON_ICO = Path(__file__).with_name("colmap_pipeline_icon.ico")

def app_base_dir() -> Path:
    """
    Returns where bundled resource files live.

    Normal Python: folder beside run_gui.py
    PyInstaller: sys._MEIPASS, where --add-data files are unpacked
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).parent

def app_exe_dir() -> Path:
    """
    Returns the folder beside the EXE when frozen, otherwise beside this script.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent

def resource_path(filename: str) -> Path:
    return app_base_dir() / filename

def crash_log_path() -> Path:
    return app_exe_dir() / "vrev_pipeline_crash.log"


_crash_log_file = None


def setup_crash_logging():
    """
    Writes uncaught Python exceptions and many native crashes to:
      vrev_pipeline_crash.log

    This is useful in PyInstaller builds where the app can close before
    the error is readable.
    """
    global _crash_log_file

    try:
        log_path = crash_log_path()
        _crash_log_file = open(log_path, "a", encoding="utf-8", buffering=1)

        _crash_log_file.write("\n\n==============================\n")
        _crash_log_file.write(f"VREV Pipeline launch: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        _crash_log_file.write(f"Executable: {sys.executable}\n")
        _crash_log_file.write(f"Base dir: {app_base_dir()}\n")
        _crash_log_file.write("==============================\n")

        faulthandler.enable(file=_crash_log_file, all_threads=True)

        def _excepthook(exc_type, exc_value, exc_tb):
            _crash_log_file.write("\n[UNCAUGHT EXCEPTION]\n")
            _crash_log_file.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            _crash_log_file.flush()

        sys.excepthook = _excepthook

        if hasattr(threading, "excepthook"):
            def _threading_excepthook(args):
                _crash_log_file.write("\n[UNCAUGHT THREAD EXCEPTION]\n")
                _crash_log_file.write("".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))
                _crash_log_file.flush()

            threading.excepthook = _threading_excepthook

    except Exception:
        pass


def write_crash_log_message(message: str):
    try:
        with open(crash_log_path(), "a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")
    except Exception:
        pass



# =========================
# Modern black UI palette
# =========================
BG = "#050505"
PANEL = "#0E0E0E"
PANEL_2 = "#181818"
BORDER = "#3A3A3A"
TEXT = "#F5F5F5"
MUTED = "#A6A6A6"
DIM = "#6F6F6F"
BUTTON_BG = "#F2F2F2"
BUTTON_TEXT = "#050505"
BUTTON_HOVER = "#FFFFFF"
LOG_BG = "#111111"
ENTRY_BG = "#101010"
PROGRESS_TROUGH = "#1C1C1C"
PROGRESS_FILL = "#D9D9D9"

# Fixed target size (equirectangular 2:1)
TARGET_WIDTH = 8192
TARGET_HEIGHT = TARGET_WIDTH // 2

# =========================
# Angle profiles
# =========================

RIGHT_SIDE_PITCH_YAW_PAIRS = [
    (0, 90),
    (42, 0),
    (-42, 0),
    (0, 42),
    (0, -42),
    (42, 180),
    (-42, 180),
    (0, 222),
    (0, 138),
]
RIGHT_SIDE_REF_IDX = 0

LEFT_SIDE_PITCH_YAW_PAIRS = [
    (0, 270),
    (42, 0),
    (-42, 0),
    (0, -42),
    (0, 42),
    (42, 180),
    (-42, 180),
    (0, 222),
    (0, 138),
]
LEFT_SIDE_REF_IDX = 0

MASKING_PITCH_YAW_PAIRS = RIGHT_SIDE_PITCH_YAW_PAIRS
MASKING_REF_IDX = RIGHT_SIDE_REF_IDX
NO_MASKING_PITCH_YAW_PAIRS = RIGHT_SIDE_PITCH_YAW_PAIRS
NO_MASKING_REF_IDX = RIGHT_SIDE_REF_IDX

VALID_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# =========================
# Minimal settings (persist)
# =========================
DEFAULT_TOPAZ_CLI = {
    "use_topaz_cli": True,
    "ffmpeg_path": r"C:\Program Files\Topaz Labs LLC\Topaz Video\ffmpeg.exe",
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
    "auto_flip_rotations": True,
    "auto_flip_debug_image": True,
    # COLMAP matching method: auto | sequential | exhaustive
    "matcher": "auto",
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
matcher_var = None
stop_btn = None
_stop_requested = False
_current_proc = None

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
        if w is not None:
            w.configure(state=state)

    # Stop button should only be active while the pipeline is running.
    if stop_btn is not None:
        stop_btn.configure(state=(NORMAL if disabled else DISABLED))


def request_stop_pipeline():
    global _stop_requested, _current_proc

    _stop_requested = True
    ui_status("Stopping pipeline…")
    ui_log("[STOP] Stop requested. The pipeline will stop as soon as the current step can safely end.")

    proc = _current_proc
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            ui_log("[STOP] Sent terminate signal to the active subprocess.")
        except Exception as e:
            ui_log(f"[WARN] Could not terminate active subprocess: {e}")


def confirm_stop_pipeline():
    if stop_btn is None or str(stop_btn.cget("state")) == DISABLED:
        return

    popup = Toplevel(_root)
    popup.title("Stop Pipeline?")
    popup.configure(bg=BG)
    popup.resizable(False, False)
    popup.transient(_root)
    popup.grab_set()

    x = _root.winfo_rootx() + 320
    y = _root.winfo_rooty() + 260
    popup.geometry(f"300x145+{x}+{y}")

    Label(
        popup,
        text="Stop the current pipeline?",
        bg=BG,
        fg=TEXT,
        font=("Helvetica", 11, "bold")
    ).pack(pady=(18, 6))

    Label(
        popup,
        text="This will cancel the active run.\nTemporary files may remain.",
        bg=BG,
        fg=MUTED,
        justify="center",
        font=("Helvetica", 9)
    ).pack(pady=(0, 12))

    row = Frame(popup, bg=BG)
    row.pack()

    Button(
        row,
        text="Cancel",
        command=popup.destroy,
        bg=PANEL_2,
        fg=TEXT,
        activebackground="#242424",
        activeforeground=TEXT,
        relief="flat",
        bd=0,
        padx=14,
        pady=6,
        cursor="hand2"
    ).pack(side="left", padx=6)

    def stop_and_close():
        popup.destroy()
        request_stop_pipeline()

    Button(
        row,
        text="Stop Pipeline",
        command=stop_and_close,
        bg="#7A1E1E",
        fg="white",
        activebackground="#9A2929",
        activeforeground="white",
        relief="flat",
        bd=0,
        padx=14,
        pady=6,
        cursor="hand2"
    ).pack(side="left", padx=6)


def should_stop_pipeline() -> bool:
    return bool(_stop_requested)


def stop_if_requested() -> bool:
    if should_stop_pipeline():
        ui_log("[STOP] Pipeline stopped by user.")
        ui_status("Stopped.")
        return True
    return False


class UILogWriter:
    """Redirect helper-script print/stdout/stderr into the GUI console."""
    def __init__(self):
        self._buf = ""

    def write(self, text):
        if not text:
            return
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                ui_log(line.rstrip())

    def flush(self):
        if self._buf.strip():
            ui_log(self._buf.rstrip())
        self._buf = ""


def run_py_script_in_process(script_name: str, argv: list[str], cwd: Path | None = None) -> bool:
    """
    PyInstaller-safe replacement for:
        subprocess.run([sys.executable, script.py, ...])

    Inside a PyInstaller EXE, sys.executable points to this GUI EXE.
    Calling [sys.executable, script.py] relaunches the GUI, which is why new
    app windows appeared after masking. This runs helper scripts in-process.
    """
    if should_stop_pipeline():
        return False

    script_path = resource_path(script_name)
    if not script_path.exists():
        ui_log(f"[ERROR] Missing helper script: {script_path}")
        return False

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    writer = UILogWriter()

    try:
        sys.argv = [str(script_path)] + [str(a) for a in argv]
        if cwd is not None:
            os.chdir(str(cwd))

        ui_log(f"[RUN-IN-PROCESS] {script_name} {' '.join(str(a) for a in argv)}")

        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            runpy.run_path(str(script_path), run_name="__main__")

        writer.flush()
        return not should_stop_pipeline()

    except SystemExit as e:
        writer.flush()
        code = e.code
        if code in (0, None):
            return not should_stop_pipeline()
        ui_log(f"[ERROR] {script_name} exited with code {code}.")
        return False

    except Exception:
        writer.flush()
        ui_log(f"[ERROR] {script_name} failed:")
        ui_log(traceback.format_exc().rstrip())
        return False

    finally:
        sys.argv = old_argv
        os.chdir(str(old_cwd))


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

def find_preextracted_frames_source(project_root: Path) -> Path | None:
    """
    Supports dropping/browsing either:
      1) a parent folder containing a frames/ folder
      2) a folder that directly contains image frames

    Returns the folder that contains image frames, or None if no pre-extracted frames are found.
    """
    frames_dir = project_root / "frames"
    if frames_dir.is_dir() and list_images(frames_dir):
        return frames_dir

    if list_images(project_root):
        return project_root

    return None

def prepare_preextracted_frames(project_root: Path, source_dir: Path) -> tuple[Path, bool]:
    """
    Returns (images_dir_for_pipeline, created_temp_frames_dir).

    If frames already live in project_root/frames, use that folder directly.
    If frames live directly in the dropped parent folder, copy them into project_root/frames
    so the rest of the pipeline can use the same expected folder structure while preserving
    the user's original files.
    """
    frames_dir = project_root / "frames"

    if source_dir.resolve() == frames_dir.resolve():
        return frames_dir, False

    frames_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src_img in list_images(source_dir):
        dst_img = frames_dir / src_img.name
        if src_img.resolve() == dst_img.resolve():
            continue
        shutil.copy2(src_img, dst_img)
        copied += 1

    ui_log(f"[FRAMES] Copied {copied} pre-extracted frame(s) into: {frames_dir}")
    return frames_dir, True

def natural_sort_key(p: Path):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', p.name)]

def _delete_dir_safe(path: Path, label: str = ""):
    try:
        if path.exists():
            shutil.rmtree(path)
            ui_log(f"[CLEAN] Removed {label or path.name} → {path}")
    except Exception as e:
        ui_log(f"[WARN] Could not remove {label or path.name} ({path}): {e}")

# =========================
# Post-COLMAP: cleanup sparse .bin if .txt exists
# =========================

def delete_sparse_bins_if_txt_exists(output_dir: Path) -> None:
    """
    If a sparse model folder contains any *.txt, remove *.bin there.
    """
    sparse_root = output_dir / "sparse"
    if not sparse_root.exists():
        ui_log(f"[INFO] No sparse/ folder found at {sparse_root} (skip .bin cleanup).")
        return

    removed = 0
    for model_dir in sorted([p for p in sparse_root.glob("*") if p.is_dir()]):
        txts = list(model_dir.glob("*.txt"))
        bins = list(model_dir.glob("*.bin"))
        if not txts or not bins:
            continue

        ui_log(f"[CLEAN] Found txt in {model_dir.name}; removing {len(bins)} bin file(s)…")
        for b in bins:
            try:
                b.unlink()
                removed += 1
            except Exception as e:
                ui_log(f"[WARN] Could not delete {b}: {e}")

    ui_log(f"[CLEAN] Sparse .bin cleanup removed {removed} file(s).")


def clear_colmap_masks_folder(output_dir: Path) -> None:
    """
    panorama_sfm creates output/masks during the run.
    We want a clean slate before writing our own *.mask outputs there.
    """
    masks_dir = output_dir / "masks"
    if not masks_dir.exists():
        ui_log(f"[INFO] No existing masks/ folder at {masks_dir} (nothing to clear).")
        return

    removed = 0
    failed = 0

    for p in masks_dir.rglob("*"):
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
                removed += 1
        except Exception as e:
            failed += 1
            ui_log(f"[WARN] Could not delete {p}: {e}")

    for d in sorted([d for d in masks_dir.rglob("*") if d.is_dir()], reverse=True):
        try:
            d.rmdir()
        except Exception:
            pass

    ui_log(f"[CLEAN] Cleared existing output/masks contents: removed={removed}, failed={failed}")

# =========================
# Post-COLMAP: rename rig images + update images.txt
# =========================

def run_rename_colmap_rig(project_root: Path) -> bool:
    renamer = resource_path("rename_colmap_rig_images_and_update_images_txt.py")
    images_txt = project_root / "output" / "sparse" / "0" / "images.txt"
    images_root = project_root / "output" / "images"

    if not renamer.exists():
        ui_log(f"[WARN] Missing renamer script (skipping): {renamer}")
        return False
    if not images_txt.exists():
        ui_log(f"[WARN] images.txt not found (skipping rename): {images_txt}")
        return False
    if not images_root.exists():
        ui_log(f"[WARN] images_root not found (skipping rename): {images_root}")
        return False

    ui_status("Renaming pano_camera images + updating images.txt…")
    ui_sub_progress(indeterminate=True)

    ok = run_py_script_in_process(
        "rename_colmap_rig_images_and_update_images_txt.py",
        ["--images_txt", str(images_txt), "--images_root", str(images_root)]
    )

    ui_sub_progress(100 if ok else 0, indeterminate=False)
    if ok:
        ui_log("[OK] Renamed rig images and updated images.txt.")
        return True
    ui_log("[ERROR] rename_colmap_rig failed.")
    return False

# =========================
# Post-COLMAP: alpha->bw masks (write *.mask.jpg next to images)
# =========================

def run_alpha_to_bw(project_root: Path) -> bool:
    masker = resource_path("alpha_to_bw_mask.py")
    output_dir = project_root / "output"
    images_root = output_dir / "images"
    seg_root = output_dir / "Segment_images"  # optional

    if not masker.exists():
        ui_log(f"[WARN] Missing alpha_to_bw_mask.py (skipping): {masker}")
        return False

    ok_any = False

    def run_one(root: Path) -> bool:
        if not root.exists():
            return False
        ui_status(f"Creating masks into output/masks from {root.name}…")
        ui_sub_progress(indeterminate=True)

        ok = run_py_script_in_process(
            "alpha_to_bw_mask.py",
            ["--input_root", str(root), "--output_dir", str(output_dir)]
        )

        ui_sub_progress(100 if ok else 0, indeterminate=False)
        if ok:
            ui_log(f"[OK] Wrote masks under: {output_dir / 'masks'}")
            return True

        ui_log(f"[ERROR] alpha_to_bw failed for {root}")
        return False

    if run_one(images_root):
        ok_any = True

    if seg_root.exists():
        if run_one(seg_root):
            ok_any = True

    return ok_any


# =========================
# Frame extraction
# =========================

def extract_frames_with_progress(video_dir: Path, interval_seconds: float) -> tuple[int, float]:
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
            if should_stop_pipeline():
                ui_log("[STOP] Frame extraction cancelled.")
                break
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
    return np.roll(img, shift_px, axis=1)

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
        if should_stop_pipeline():
            ui_log("[STOP] Seam roll cancelled.")
            break
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
    files = list_images(src_dir)
    if not files:
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(sorted(files, key=natural_sort_key), 0):
        if should_stop_pipeline():
            ui_log("[STOP] Sequence creation cancelled.")
            break
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
    classes, grow_px = [0], 30

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
        if should_stop_pipeline():
            ui_log("[STOP] Masking cancelled.")
            break
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
# Auto-flip decision (ONE frame)
# =========================

def _pick_mid_frame(images_dir: Path) -> Path | None:
    imgs = list_images(images_dir)
    if not imgs:
        return None
    imgs = sorted(imgs, key=natural_sort_key)
    return imgs[len(imgs) // 2]

def decide_operator_side_one_frame(images_dir: Path, debug_out_path: Path | None = None) -> str:
    try:
        from ultralytics import YOLO
        import numpy as np
        import cv2
        import time
    except Exception:
        ui_log("[AUTOFLIP] ultralytics/torch/OpenCV not available; defaulting to RIGHT.")
        ui_log("          pip install ultralytics opencv-python torch torchvision torchaudio")
        return "unknown"

    mid = _pick_mid_frame(images_dir)
    if mid is None:
        ui_log(f"[AUTOFLIP] No images found in {images_dir}; defaulting to RIGHT.")
        return "unknown"

    model_name = "yolov8n-seg.pt"
    conf, iou, imgsz = 0.30, 0.45, 1024
    classes = [0]

    ui_log(f"[AUTOFLIP] Analyzing one frame: {mid.name}")
    t0 = time.time()
    model = YOLO(model_name)

    bgr = cv2.imread(str(mid), cv2.IMREAD_COLOR)
    if bgr is None:
        ui_log(f"[AUTOFLIP] Failed to read {mid.name}; defaulting to RIGHT.")
        return "unknown"

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    results = model.predict(
        source=rgb, conf=conf, iou=iou, imgsz=imgsz,
        classes=classes, verbose=False, device=None
    )

    if not results or results[0].masks is None or len(results[0].masks) == 0:
        ui_log(f"[AUTOFLIP] No person mask found (took {time.time()-t0:.2f}s). Defaulting to RIGHT.")
        return "unknown"

    masks_tensor = results[0].masks.data.cpu().numpy()
    merged_small = (masks_tensor.max(axis=0) > 0.5).astype(np.uint8)
    merged = cv2.resize(merged_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    left_area = int(merged[:, :w//2].sum())
    right_area = int(merged[:, w//2:].sum())

    side = "right" if right_area >= left_area else "left"
    dt = time.time() - t0
    ui_log(f"[AUTOFLIP] mask_area_left={left_area:,}  mask_area_right={right_area:,}  → {side.upper()} (t={dt:.2f}s)")

    if debug_out_path is not None:
        try:
            overlay = bgr.copy()
            overlay[merged] = (0, 0, 255)
            out = cv2.addWeighted(bgr, 0.65, overlay, 0.35, 0.0)
            cv2.imwrite(str(debug_out_path), out)
            ui_log(f"[AUTOFLIP] Wrote debug overlay → {debug_out_path}")
        except Exception as e:
            ui_log(f"[AUTOFLIP] Could not write debug overlay: {e}")

    return side

def choose_rotation_profile(images_dir_for_next: Path, cfg: dict, project_root: Path, base_pairs: list[tuple[float, float]], base_ref_idx: int) -> tuple[list[tuple[float, float]], int, str]:
    auto_flip = bool(cfg.get("auto_flip_rotations", True))
    write_debug = bool(cfg.get("auto_flip_debug_image", True))

    if not auto_flip:
        return base_pairs, base_ref_idx, "RIGHT(default; auto-flip off)"

    debug_path = (project_root / "autoflip_debug.jpg") if write_debug else None
    side = decide_operator_side_one_frame(images_dir_for_next, debug_out_path=debug_path)

    if side == "left":
        return LEFT_SIDE_PITCH_YAW_PAIRS, LEFT_SIDE_REF_IDX, "LEFT(flipped)"
    if side == "right":
        return base_pairs, base_ref_idx, "RIGHT(default)"
    return base_pairs, base_ref_idx, "RIGHT(default; unknown/no-yolo)"

# =========================
# COLMAP wrapper (stream logs)
# =========================

def run_panorama_sfm(
    project_root: Path,
    images_dir: Path,
    matcher: str = "exhaustive",
    walking_speed_mps: float = 0.7,
    fps_for_scale: float = 1.0,
    export_baseline_m: float = 0.015,
) -> bool:
    panorama = resource_path("panorama_sfm.py")
    wrapper  = resource_path("run_panorama_sfm.py")

    if panorama.exists():
        script_name = "panorama_sfm.py"
        argv = [
            "--input_image_path", str(images_dir),
            "--output_path",      str(project_root / "output"),
            "--matcher",          str(matcher),
            "--walking_speed_mps", str(walking_speed_mps),
            "--fps",               str(fps_for_scale),
            "--export_baseline_m", str(export_baseline_m),
        ]
    elif wrapper.exists():
        script_name = "run_panorama_sfm.py"
        argv = [str(project_root)]
        ui_log("[WARN] panorama_sfm.py not found; calling run_panorama_sfm.py.")
    else:
        ui_log("[ERROR] Neither panorama_sfm.py nor run_panorama_sfm.py found.")
        return False

    ui_status(f"Running COLMAP pipeline… (matcher={matcher})")
    ui_sub_progress(indeterminate=True)

    ok = run_py_script_in_process(script_name, argv, cwd=project_root)

    ui_sub_progress(100 if ok else 0, indeterminate=False)

    if should_stop_pipeline():
        ui_log("[STOP] panorama_sfm was cancelled.")
        return False

    if ok:
        ui_log("[OK] panorama_sfm finished.")
        return True

    ui_log("[ERROR] panorama_sfm failed.")
    return False

def delete_pano_camera0(project_root: Path) -> None:
    deleter = resource_path("delete_pano0.py")
    if not deleter.exists():
        ui_log(f"[WARN] Missing delete_pano0.py (skipping): {deleter}")
        return
    ui_status("Deleting pano_camera0 folders…")
    ui_sub_progress(indeterminate=True)

    ok = run_py_script_in_process("delete_pano0.py", [str(project_root)])

    ui_sub_progress(100 if ok else 0, indeterminate=False)
    if ok:
        ui_log("[OK] Deleted all pano_camera0 folders.")
    else:
        ui_log("[WARN] delete_pano0.py reported an error.")

def run_segment_images(project_root: Path) -> bool:
    seg = resource_path("segment_images.py")
    if not seg.exists():
        ui_log(f"[WARN] Missing segment_images.py (skipping): {seg}")
        return False
    ui_status("Segmenting COLMAP images by clip prefix…")
    ui_sub_progress(indeterminate=True)

    ok = run_py_script_in_process("segment_images.py", [], cwd=project_root)

    ui_sub_progress(100 if ok else 0, indeterminate=False)
    if ok:
        ui_log("[OK] Segmented images (see Segment_images/).")
        return True

    ui_log("[WARN] segment_images.py reported an error.")
    return False

# =========================
# Topaz headless CLI (unchanged)
# =========================

def run_topaz_via_cli_command(raw_cmd: str,
                              ffmpeg_path: str,
                              model_dir: str,
                              model_data_dir: str,
                              seq_in_dir: Path,
                              seq_out_dir: Path,
                              fps: float = 24.0) -> bool:
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

    raw = " ".join(tokens)
    m = re.search(r'%0?\d+d\.(png|tif|tiff|bmp|jpg|jpeg)', raw, flags=re.IGNORECASE)
    out_ext = (m.group(1).lower() if m else "tiff")
    if out_ext == "tif":
        out_ext = "tiff"
    out_pat = str(seq_out_dir / f"%06d.{out_ext}")

    tokens = [t for t in tokens if not is_ffmpeg(t)]

    cleaned = []
    skip = 0
    for t in tokens:
        if skip:
            skip -= 1
            continue
        if is_flag(t, "-i"):
            skip = 1
            continue
        cleaned.append(t)
    tokens = cleaned

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

    final_tokens = [
        ffmpeg_path, "-hide_banner",
        "-framerate", f"{fps:.6f}",
        "-start_number", "0",
        "-i", in_pat,
    ] + tokens

    ui_log("[TOPAZ/CLI] " + " ".join(final_tokens))
    ui_status("Running Topaz export (headless)…")
    ui_sub_progress(indeterminate=True)
    global _current_proc

    try:
        proc = subprocess.Popen(
            final_tokens,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
        _current_proc = proc

        if proc.stdout is not None:
            for line in proc.stdout:
                if line:
                    ui_log(line.rstrip())
                if should_stop_pipeline():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break

        ret = proc.wait()
        _current_proc = None
        ui_sub_progress(100, indeterminate=False)

        if should_stop_pipeline():
            ui_log("[STOP] Topaz CLI export was cancelled.")
            return False

        if ret == 0:
            first = next(iter(sorted(seq_out_dir.glob(f"*.{out_ext}"))), None)
            if first:
                img = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    ui_log(f"[VERIFY] First enhanced frame: {w}x{h} ({first.name})")
            ui_log("[OK] Topaz CLI export finished.")
            return True

        ui_log(f"[ERROR] Topaz CLI returned {ret}.")
        return False
    except FileNotFoundError:
        _current_proc = None
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[ERROR] ffmpeg not found: {ffmpeg_path}")
        return False
    except Exception as e:
        _current_proc = None
        ui_sub_progress(0, indeterminate=False)
        if should_stop_pipeline():
            ui_log("[STOP] Topaz CLI export was cancelled.")
        else:
            ui_log(f"[ERROR] Failed to run Topaz CLI: {e}")
        return False

# =========================
# Pipeline (thread)
# =========================

def pipeline_thread(
    project_root: Path,
    seconds_per_frame: float,
    masking_enabled: bool,
    use_topaz: bool,
    cfg: dict,
    matcher: str
):
    frames_root      = project_root / "frames"
    rolled_root      = project_root / "frames_seam_yaw"
    seq_root         = project_root / "frames_seam_yaw_seq"
    enh_root         = project_root / ENHANCED_SUBFOLDER
    final_frames_dir = project_root / "frames_final"
    output_dir       = project_root / "output"

    # Your current best settings
    walking_speed_mps  = 0.7
    export_baseline_m  = 0.015

    # New: allow a folder with pre-extracted frames instead of videos.
    preextracted_source = find_preextracted_frames_source(project_root)
    has_videos = len(list_videos(project_root)) > 0
    using_preextracted_frames = bool(preextracted_source and not has_videos)
    copied_preextracted_frames = False

    global _stop_requested

    try:
        _stop_requested = False
        ui_disable_inputs(True)
        ui_main_progress(0, indeterminate=False)

        # A) Input preparation
        if using_preextracted_frames:
            ui_status("Using pre-extracted frames…")
            ui_log(f"[FRAMES] Pre-extracted frames detected in: {preextracted_source}")
            frames_root, copied_preextracted_frames = prepare_preextracted_frames(project_root, preextracted_source)

            video_count = 0
            fps_first = 24.0

            # For pre-extracted frames, keep the existing scale assumption:
            # seconds_per_frame controls the effective sample interval used by panorama_sfm.
            ui_log("[FRAMES] Skipping video extraction.")
            ui_main_progress(10, indeterminate=False)
            if stop_if_requested():
                return

        else:
            # Original behavior: extract frames from video(s).
            ui_status("Preparing extraction…")
            video_count, fps_first = extract_frames_with_progress(project_root, seconds_per_frame)

            if not list_images(frames_root):
                ui_log("[ERROR] No frames extracted. Aborting.")
                ui_status("Idle.")
                return
            ui_main_progress(10, indeterminate=False)
            if stop_if_requested():
                return

        # Compute effective sampling FPS for panorama_sfm scale normalization
        try:
            fps_for_scale = 1.0 / float(seconds_per_frame)
        except Exception:
            fps_for_scale = 1.0

        if fps_for_scale <= 0:
            fps_for_scale = 1.0

        ui_log(
            f"[SCALE] seconds_per_frame={seconds_per_frame:.6f}  "
            f"-> fps_for_scale={fps_for_scale:.6f}  "
            f"(target_step={walking_speed_mps / fps_for_scale:.6f})  "
            f"baseline={export_baseline_m:.6f}"
        )

        # B) Topaz (optional)
        if use_topaz:
            if using_preextracted_frames:
                ui_log("[TOPAZ] Pre-extracted frames detected; running Topaz enhancement on those frames.")
            ui_status("Seam rolling frames…")
            cnt, in_width = roll_folder_lossless(frames_root, rolled_root, SEAM_YAW_DEG)
            if cnt == 0:
                ui_log("[ERROR] Seam roll produced no images. Aborting.")
                ui_status("Idle.")
                return
            _delete_dir_safe(frames_root, "raw extracted frames")
            ui_main_progress(15, indeterminate=False)
            if stop_if_requested():
                return

            ui_status("Preparing Topaz sequence…")
            nseq = make_numeric_sequence(rolled_root, seq_root)
            if nseq == 0:
                ui_log("[ERROR] Failed to create numeric sequence for Topaz.")
                ui_status("Idle.")
                return

            ui_status("Running Topaz…")
            ok = run_topaz_via_cli_command(
                raw_cmd=cfg.get("topaz_cli_cmd", DEFAULT_TOPAZ_CLI["topaz_cli_cmd"]),
                ffmpeg_path=cfg.get("ffmpeg_path", DEFAULT_TOPAZ_CLI["ffmpeg_path"]),
                model_dir=cfg.get("model_dir", DEFAULT_TOPAZ_CLI["model_dir"]),
                model_data_dir=cfg.get("model_data_dir", DEFAULT_TOPAZ_CLI["model_data_dir"]),
                seq_in_dir=seq_root,
                seq_out_dir=enh_root,
                fps=fps_first if fps_first > 0 else 24.0,
            )
            if not ok:
                ui_status("Idle.")
                return

            _delete_dir_safe(seq_root, "Topaz numeric sequence")
            ui_main_progress(35, indeterminate=False)
            if stop_if_requested():
                return

            ui_status("Rolling Topaz output back…")
            cnt_back, _ = roll_folder_lossless(enh_root, final_frames_dir, -SEAM_YAW_DEG)
            if cnt_back == 0:
                ui_log("[ERROR] Roll-back produced no images. Aborting.")
                ui_status("Idle.")
                return

            _delete_dir_safe(rolled_root, "seam-shifted frames")
            _delete_dir_safe(enh_root, "Topaz enhanced frames")
            ui_main_progress(50, indeterminate=False)
            if stop_if_requested():
                return

            images_dir_for_next = final_frames_dir
        else:
            ui_log("[TOPAZ] Skipped Topaz enhancement (checkbox off).")
            images_dir_for_next = frames_root

        # C) Rotation override + optional masking
        base_pairs = MASKING_PITCH_YAW_PAIRS if masking_enabled else NO_MASKING_PITCH_YAW_PAIRS
        base_ref   = MASKING_REF_IDX if masking_enabled else NO_MASKING_REF_IDX

        ui_status("Choosing virtual rotation profile…")
        pairs, ref_idx, label = choose_rotation_profile(
            images_dir_for_next,
            cfg=cfg,
            project_root=project_root,
            base_pairs=base_pairs,
            base_ref_idx=base_ref,
        )

        ui_status(f"Writing rotation override ({label})…")
        write_rotation_override(project_root, pairs, ref_idx)
        ui_log(f"[OK] rotation_override.json → {label}")
        ui_log(f"[OK] pitch_yaw_pairs: {pairs}")

        if masking_enabled:
            ui_status("Masking frames (RGBA alpha)…")
            processed = run_yolo_masking(images_dir_for_next)
            if processed == 0:
                ui_log("[WARN] No frames were masked (or masking failed). Continuing…")
        else:
            ui_log("[MASKING] Skipped masking (checkbox off).")

        ui_main_progress(65, indeterminate=False)
        if stop_if_requested():
            return

        # D) COLMAP / panorama_sfm
        output_dir.mkdir(parents=True, exist_ok=True)

        ui_status("Running panorama_sfm (COLMAP)…")
        ok_colmap = run_panorama_sfm(
            project_root,
            images_dir_for_next,
            matcher=matcher,
            walking_speed_mps=walking_speed_mps,
            fps_for_scale=fps_for_scale,
            export_baseline_m=export_baseline_m,
        )
        if not ok_colmap:
            ui_status("COLMAP failed. See log.")
            return

        ui_main_progress(85, indeterminate=False)
        if stop_if_requested():
            return

        # E) Delete pano_camera0 (your existing post-step)
        ui_status("Deleting pano_camera0…")
        delete_pano_camera0(project_root)
        ui_main_progress(90, indeterminate=False)
        if stop_if_requested():
            return

        # F) Segment images if multiple input videos
        if video_count > 1:
            ui_log(f"[INFO] Multiple videos detected ({video_count}). Running segment_images…")
            run_segment_images(project_root)
        ui_main_progress(92, indeterminate=False)
        if stop_if_requested():
            return

        # G) Ensure sparse .bin is removed if txt exists
        ui_status("Cleaning sparse binaries (if txt exists)…")
        delete_sparse_bins_if_txt_exists(output_dir)
        ui_main_progress(94, indeterminate=False)
        if stop_if_requested():
            return

        # H) Post-step: rename rig images then alpha->bw masks
        ui_status("Post-processing COLMAP output…")

        ok_rename = run_rename_colmap_rig(project_root)
        if not ok_rename:
            ui_log("[WARN] rename_colmap_rig step failed or was skipped.")

        ui_status("Clearing COLMAP masks folder…")
        clear_colmap_masks_folder(output_dir)

        ok_masks = run_alpha_to_bw(project_root)
        if not ok_masks:
            ui_log("[WARN] alpha_to_bw step failed or was skipped.")

        # I) Final cleanup: delete temporary pipeline files.
        ui_status("Final cleanup…")

        KEEP_DIRS = {"output"}
        KEEP_EXTS = {ext.lower() for ext in VALID_VIDEO_EXT}

        # If the user dropped a folder of image frames, do not delete their source image files.
        if using_preextracted_frames:
            KEEP_EXTS.update({ext.lower() for ext in VALID_IMAGE_EXT})

        for child in project_root.iterdir():
            if child.is_dir():
                if child.name.lower() in KEEP_DIRS:
                    continue

                # Preserve an existing frames/ source folder if the user supplied one.
                if using_preextracted_frames and child.resolve() == frames_root.resolve() and not copied_preextracted_frames:
                    ui_log(f"[CLEAN] Preserved source frames folder → {child}")
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
        if using_preextracted_frames:
            ui_log("[DONE] Pipeline complete. Kept original frame input(s) and the 'output' folder.")
        else:
            ui_log("[DONE] Pipeline complete. Kept original video(s) and the 'output' folder.")

    finally:
        try:
            progress_main.stop()
            progress_sub.stop()
        except Exception:
            pass
        ui_disable_inputs(False)


# =========================
# GUI helpers
# =========================

class HoverTooltip:
    """
    Small hover tooltip for helper text.
    Replace the tooltip_text values in main() with your final copy.
    """
    def __init__(self, widget, text: str, delay_ms: int = 250, wraplength: int = 280):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self.tip = None
        self.after_id = None

        widget.bind("<Enter>", self.schedule)
        widget.bind("<Leave>", self.hide)
        widget.bind("<ButtonPress>", self.hide)

    def schedule(self, _event=None):
        self.cancel()
        self.after_id = self.widget.after(self.delay_ms, self.show)

    def cancel(self):
        if self.after_id:
            try:
                self.widget.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def show(self):
        if self.tip or not self.text:
            return

        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + 22

        self.tip = tip = Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")

        box = Label(
            tip,
            text=self.text,
            justify="left",
            bg="#1A1A1A",
            fg=TEXT,
            relief="solid",
            bd=1,
            padx=10,
            pady=7,
            wraplength=self.wraplength,
            font=("Helvetica", 9)
        )
        box.pack()

    def hide(self, _event=None):
        self.cancel()
        if self.tip:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None


def open_how_to_page():
    try:
        webbrowser.open(HOW_TO_URL)
        ui_log(f"[INFO] Opened how-to page: {HOW_TO_URL}")
    except Exception as e:
        ui_log(f"[WARN] Could not open how-to page: {e}")


# =========================
# GUI actions
# =========================

_last_folder = None

def resolve_matcher_selection(folder_path: Path, selected_matcher: str) -> tuple[str, int]:
    selected_matcher = (selected_matcher or "auto").strip().lower()
    if selected_matcher not in {"auto", "sequential", "exhaustive"}:
        selected_matcher = "auto"

    videos = list_videos(folder_path)
    video_count = len(videos)
    has_preextracted_frames = find_preextracted_frames_source(folder_path) is not None

    if selected_matcher == "auto":
        # One video is a continuous path, so sequential matching is faster/cleaner.
        # Multiple videos or pre-extracted frame folders may not be temporally connected,
        # so exhaustive matching is safer.
        if video_count > 1 or (video_count == 0 and has_preextracted_frames):
            return "exhaustive", video_count
        return "sequential", video_count

    return selected_matcher, video_count

def start_pipeline_with_path(folder_path: Path):
    global _last_folder
    _last_folder = folder_path
    try:
        seconds = float(frame_interval.get())
        if seconds <= 0:
            raise ValueError
    except Exception:
        seconds = 1.0
        frame_interval.set("1")

    cfg = load_settings()
    matcher_selection = str(matcher_var.get() or "auto")
    resolved_matcher, video_count = resolve_matcher_selection(folder_path, matcher_selection)

    cfg["seconds_per_frame"] = seconds
    cfg["use_masking"] = bool(use_masking.get())
    cfg["auto_flip_rotations"] = True  # Always enabled; UI toggle removed.
    cfg["use_topaz_cli"] = bool(use_topaz_cli_var.get())
    cfg["matcher"] = matcher_selection
    save_settings(cfg)

    if matcher_selection == "auto":
        if video_count == 0 and find_preextracted_frames_source(folder_path) is not None:
            ui_log(f"[MATCHER] Auto selected: pre-extracted frames found → using {resolved_matcher} matcher.")
        else:
            ui_log(f"[MATCHER] Auto selected: {video_count} video(s) found → using {resolved_matcher} matcher.")
    else:
        ui_log(f"[MATCHER] Manual selection → using {resolved_matcher} matcher.")

    threading.Thread(
        target=pipeline_thread,
        args=(folder_path, seconds, use_masking.get(), use_topaz_cli_var.get(), cfg, resolved_matcher),
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
    chosen = filedialog.askdirectory(title="Select folder containing 360 video(s) or pre-extracted frames")
    if chosen:
        run_btn.configure(state=NORMAL)
        start_pipeline_with_path(Path(chosen))

def run_last():
    if _last_folder and Path(_last_folder).exists():
        start_pipeline_with_path(Path(_last_folder))
    else:
        ui_log("[ERROR] No valid last folder. Please Browse or Drop a folder.")

# =========================
# Main GUI
# =========================

def main():
    global _root, status_var, progress_main, progress_sub, log_text
    global use_masking, use_topaz_cli_var, frame_interval
    global drop_zone, run_btn, browse_btn, matcher_var, stop_btn

    s = load_settings()

    _root = TkinterDnD.Tk()
    _root.title("VREV 360 Video SFM Pipeline v0.1")
    _root.geometry("920x730")
    _root.configure(bg=BG)
    _root.resizable(False, False)

    # App/window icon only. The logo image is intentionally not displayed inside the app UI.
    icon_path = resource_path("colmap_pipeline_icon.ico")
    if icon_path.exists():
        try:
            _root.iconbitmap(default=str(icon_path))
        except Exception:
            pass

    style = ttk.Style(_root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(
        "VREV.Horizontal.TProgressbar",
        troughcolor=PROGRESS_TROUGH,
        background=PROGRESS_FILL,
        bordercolor=PROGRESS_TROUGH,
        lightcolor=PROGRESS_FILL,
        darkcolor=PROGRESS_FILL,
        thickness=9,
    )

    header = Frame(_root, bg=BG); header.pack(pady=(60, 12))
    Label(header,
          text="VREV Local 360 Panoramic Video SFM Pipeline",
          bg=BG, fg=TEXT, font=("Helvetica", 16, "bold")).pack()

    # ---- Row 1: interval + checkboxes ----
    ctrl = Frame(_root, bg=BG)
    ctrl.pack(pady=(8, 2))

    Label(ctrl, text="Extract 1 frame per", bg=BG, fg=TEXT).pack(side="left")
    frame_interval = StringVar(value=str(s.get("seconds_per_frame", "1")))
    Entry(
        ctrl,
        textvariable=frame_interval,
        width=6,
        bg=ENTRY_BG,
        fg=TEXT,
        insertbackground=TEXT,
        relief="flat",
        bd=0,
        highlightbackground=BORDER,
        highlightcolor="#777777",
        highlightthickness=1
    ).pack(side="left", padx=(6, 6), ipady=3)
    Label(ctrl, text="seconds", bg=BG, fg=TEXT).pack(side="left", padx=(0, 16))

    use_topaz_cli_var = BooleanVar(value=bool(s.get("use_topaz_cli", True)))
    Checkbutton(
        ctrl, text="Use Topaz enhancement", variable=use_topaz_cli_var,
        onvalue=True, offvalue=False, bg=BG, fg=TEXT,
        activebackground=BG, activeforeground=TEXT, selectcolor=PANEL_2
    ).pack(side="left", padx=(0, 16))

    use_masking = BooleanVar(value=bool(s.get("use_masking", True)))
    Checkbutton(
        ctrl, text="Enable Masking",
        variable=use_masking, onvalue=True, offvalue=False,
        bg=BG, fg=TEXT, activebackground=BG, activeforeground=TEXT, selectcolor=PANEL_2
    ).pack(side="left", padx=(0, 16))

    # ---- Row 2: matcher (moved to its own row so it doesn't go off-screen) ----
    matcher_row = Frame(_root, bg=BG)
    matcher_row.pack(pady=(2, 6))

    matcher_default = str(s.get("matcher", "auto")).strip().lower()
    if matcher_default not in {"auto", "sequential", "exhaustive"}:
        matcher_default = "auto"
    matcher_var = StringVar(value=matcher_default)
    Label(matcher_row, text="Matcher:", bg=BG, fg=TEXT).pack(side="left", padx=(0, 10))

    Radiobutton(
        matcher_row, text="Auto", variable=matcher_var, value="auto",
        bg=BG, fg=TEXT, activebackground=BG, activeforeground=TEXT, selectcolor=PANEL_2
    ).pack(side="left", padx=(0, 12))

    Radiobutton(
        matcher_row, text="Sequential", variable=matcher_var, value="sequential",
        bg=BG, fg=TEXT, activebackground=BG, activeforeground=TEXT, selectcolor=PANEL_2
    ).pack(side="left", padx=(0, 3))

    sequential_help = Label(
        matcher_row,
        text="?",
        bg=PANEL_2,
        fg=TEXT,
        width=2,
        cursor="question_arrow",
        font=("Helvetica", 8, "bold")
    )
    sequential_help.pack(side="left", padx=(0, 12))
    HoverTooltip(
        sequential_help,
        "Sequential matcher is best used for frames extracted in a sequential order (for ex: 1 frame is extracted per second in order). It is generally faster and is the default choice if one single video is in the parent folder."
    )

    Radiobutton(
        matcher_row, text="Exhaustive", variable=matcher_var, value="exhaustive",
        bg=BG, fg=TEXT, activebackground=BG, activeforeground=TEXT, selectcolor=PANEL_2
    ).pack(side="left", padx=(0, 3))

    exhaustive_help = Label(
        matcher_row,
        text="?",
        bg=PANEL_2,
        fg=TEXT,
        width=2,
        cursor="question_arrow",
        font=("Helvetica", 8, "bold")
    )
    exhaustive_help.pack(side="left")
    HoverTooltip(
        exhaustive_help,
        "Exhaustive matcher is best used for frames taken in a non sequential order (for ex: multiple videos taken in different areas). It is generally slower and is the default choice if more than one video or pre-extracted frames are in the parent folder."
    )

    # Info labels
    Label(_root, text=f"Topaz ffmpeg: {s.get('ffmpeg_path')}", bg=BG, fg=DIM).pack()
    Label(_root, text=f"Model dir:    {s.get('model_dir')}", bg=BG, fg=DIM).pack()

    # Drop area
    drop_zone = Label(
        _root, text="Drop Folder With 360 Panoramic Video(s) or Pre Extracted Frames Here",
        bg=PANEL_2, fg=TEXT, width=120, height=6,
        relief="flat", bd=0,
        highlightbackground=BORDER,
        highlightcolor="#777777",
        highlightthickness=1
    )
    drop_zone.pack(pady=10)
    drop_zone.drop_target_register(DND_FILES)
    drop_zone.dnd_bind('<<Drop>>', on_drop)

    # Buttons
    btns = Frame(_root, bg=BG); btns.pack()
    browse_btn = Button(
        btns,
        text="Browse…",
        command=browse_folder,
        bg=BUTTON_BG,
        fg=BUTTON_TEXT,
        activebackground=BUTTON_HOVER,
        activeforeground=BUTTON_TEXT,
        relief="flat",
        bd=0,
        padx=16,
        pady=7,
        font=("Helvetica", 9, "bold"),
        cursor="hand2"
    ); browse_btn.pack(side="left", padx=6)

    how_to_btn = Button(
        btns,
        text="View User Guide",
        command=open_how_to_page,
        bg=PANEL_2,
        fg=TEXT,
        activebackground="#242424",
        activeforeground=TEXT,
        relief="flat",
        bd=0,
        padx=16,
        pady=7,
        font=("Helvetica", 9, "bold"),
        cursor="hand2",
        highlightbackground=BORDER,
        highlightthickness=1
    )
    how_to_btn.pack(side="left", padx=6)

    stop_btn = Button(
        btns,
        text="Stop Pipeline",
        command=confirm_stop_pipeline,
        bg="#7A1E1E",
        fg="white",
        activebackground="#9A2929",
        activeforeground="white",
        relief="flat",
        bd=0,
        padx=16,
        pady=7,
        font=("Helvetica", 9, "bold"),
        cursor="hand2",
        state=DISABLED
    )
    stop_btn.pack(side="left", padx=6)

    # Hidden compatibility button.
    # on_drop(), browse_folder(), run_last(), and ui_disable_inputs() still expect run_btn to exist.
    run_btn = Button(btns, text="Run Last Chosen Folder", state=DISABLED, command=run_last)
    run_btn.pack_forget()

    # Progress bars
    prog = Frame(_root, bg=BG); prog.pack(fill=BOTH, padx=16, pady=(12, 4))
    Label(prog, text="Overall Progress", bg=BG, fg=MUTED).pack(anchor="w")
    progress_main = ttk.Progressbar(prog, orient="horizontal", mode="determinate", length=840, style="VREV.Horizontal.TProgressbar"); progress_main.pack(pady=(2, 8))
    Label(prog, text="Current Task", bg=BG, fg=MUTED).pack(anchor="w")
    progress_sub = ttk.Progressbar(prog, orient="horizontal", mode="determinate", length=840, style="VREV.Horizontal.TProgressbar"); progress_sub.pack(pady=(2, 8))

    # Status + log
    status_var = StringVar(value="Idle.")
    Label(_root, textvariable=status_var, bg=BG, fg=TEXT).pack(pady=(0, 6))

    log_frame = Frame(
        _root,
        bg=LOG_BG,
        highlightbackground="#555555",
        highlightcolor="#555555",
        highlightthickness=1,
        bd=0
    )
    log_frame.pack(fill="x", expand=False, padx=16, pady=(0, 12))

    Label(
        log_frame,
        text="Console Log",
        bg=LOG_BG,
        fg="#CFCFCF",
        font=("Helvetica", 9, "bold")
    ).pack(anchor="w", padx=10, pady=(7, 0))

    log_text = Text(
        log_frame,
        height=11,
        bg=LOG_BG,
        fg="#E2E2E2",
        insertbackground=TEXT,
        relief="flat",
        bd=0,
        padx=10,
        pady=8
    )
    log_text.pack(fill="x", expand=False)
    log_text.configure(state=DISABLED)

    _root.mainloop()

if __name__ == "__main__":
    setup_crash_logging()
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        write_crash_log_message("\n[FATAL MAIN EXCEPTION]\n" + err)
        raise
