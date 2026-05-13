"""
panorama_sfm.py

Incremental SfM on 360 panoramas with:
- Virtual perspective rendering into pano_camera{idx}/...
- Mapping with ZERO baseline (rig offsets disabled during mapping)
- Post-mapping scale normalization using consecutive-frame distances from ONE stream
  (default: pano_camera0), using CAMERA-CENTER distance
- Uniform scaling of BOTH camera centers and 3D points about ONE SHARED PIVOT point
  (default pivot = median camera center of the chosen stream)
- Then apply a final export-time baseline offset (e.g. 0.015) to non-ref rig cameras
- Finally write the adjusted COLMAP TEXT model to output/sparse/0 (etc.)

Designed to be called from your run_gui wrapper.

Notes:
- Scale target step = walking_speed_mps / fps
  Example: walking_speed_mps=0.7 and fps=2.0 => target_step=0.35
- Filename parsing supports:
    pano_camera0/ShortTest(1)_frame_00030.png
    pano_camera0/ShortTest(1)_frame_00030_pano_camera_0.png
    pano_camera0/000030.png
    frame_000030.jpg
    000030.jpg
"""

import argparse
import json
import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import PIL.ExifTags
import PIL.Image
import pycolmap
from pycolmap import logging
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Rotation override helper
# -----------------------------------------------------------------------------

def load_rotation_override_if_any(input_image_path: Path | None):
    """
    Look for rotation_override.json written by the GUI.
    Search order:
      1) input_image_path (e.g. .../frames)
      2) parent of input_image_path
      3) current working directory
    Returns (pairs, ref_idx) or (None, None)
    """
    candidates = []
    if input_image_path:
        candidates.append(Path(input_image_path) / "rotation_override.json")
        candidates.append(Path(input_image_path).parent / "rotation_override.json")
    candidates.append(Path.cwd() / "rotation_override.json")

    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                pairs = data.get("pitch_yaw_pairs", None)
                ref_idx = data.get("ref_idx", 0)
                if isinstance(pairs, list) and len(pairs) > 0:
                    pairs = [(float(a), float(b)) for (a, b) in pairs]
                    ref_idx = int(ref_idx)
                    logging.info(f"Using rotation_override.json at: {p}")
                    return pairs, ref_idx
            except Exception as e:
                logging.warning(f"Failed to read {p}: {e}")
    return None, None


def _circ_dist_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def _should_flip_baseline_from_pairs(
    pairs: list[tuple[float, float]],
    ref_idx: int,
    default_ref_yaw_deg: float = 90.0,
    flipped_ref_yaw_deg: float = 270.0,
) -> bool:
    """
    Decide if we're in the "flipped" convention, based on the reference yaw.
    """
    if not pairs:
        return False
    ref_idx = max(0, min(int(ref_idx), len(pairs) - 1))
    ref_yaw = float(pairs[ref_idx][1]) % 360.0
    return _circ_dist_deg(ref_yaw, flipped_ref_yaw_deg) < _circ_dist_deg(ref_yaw, default_ref_yaw_deg)


# -----------------------------------------------------------------------------
# Virtual camera + spherical mapping
# -----------------------------------------------------------------------------

def create_virtual_camera(pano_height: int, fov_deg: float = 90) -> pycolmap.Camera:
    image_size = int(pano_height * fov_deg / 180)
    focal = image_size / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    return pycolmap.Camera.create(0, "PINHOLE", focal, image_size, image_size)


def get_virtual_camera_rays(camera: pycolmap.Camera) -> np.ndarray:
    size = (camera.width, camera.height)
    y, x = np.indices(size).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()])
    xy += 0.5
    xy_norm = camera.cam_from_img(xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(image_size, rays_in_cam: np.ndarray) -> np.ndarray:
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360° panoramas are supported.")
    if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
        raise ValueError(f"{rays_in_cam.shape=} but expected (N,3).")
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size


def get_virtual_rotations() -> Sequence[np.ndarray]:
    """
    Default 9-view set (your existing scheme).
    """
    pitch_yaw_pairs = [
        (0, 90),   # ref
        (42, 0),
        (-42, 0),
        (0, 42),
        (0, -42),
        (42, 180),
        (-42, 180),
        (0, 222),
        (0, 138),
    ]
    cams_from_pano_r = []
    for pitch_deg, yaw_deg in pitch_yaw_pairs:
        cam_from_pano_r = Rotation.from_euler("YX", [yaw_deg, pitch_deg], degrees=True).as_matrix()
        cams_from_pano_r.append(cam_from_pano_r)
    return cams_from_pano_r


# -----------------------------------------------------------------------------
# Rig config (mapping baseline is forced to 0.0 in this script)
# -----------------------------------------------------------------------------

def create_pano_rig_config(
    cams_from_pano_rotation: Sequence[np.ndarray],
    ref_idx: int = 0,
    baseline: float = 0.0,
) -> pycolmap.RigConfig:
    """
    Create a RigConfig.
    IMPORTANT: mapping baseline will be 0.0 (we pass baseline=0.0 during mapping).
    """
    rig_cameras = []
    for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T

            # Keep your original grouping convention (used later for export baseline too)
            side = 1 if idx <= 4 else -1
            local_offset = np.array([-baseline * side, 0.0, 0.0], dtype=np.float64)
            translation = cam_from_ref_rotation @ local_offset

            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation),
                translation,
            )

        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=(idx == ref_idx),
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )

    return pycolmap.RigConfig(cameras=rig_cameras)


def render_perspective_images(
    pano_image_names: Sequence[str],
    pano_image_dir: Path,
    output_image_dir: Path,
    mask_dir: Path,
    override_pairs: list[tuple[float, float]] | None = None,
    override_ref_idx: int | None = None,
):
    """
    Render the virtual perspective images + masks.

    Returns:
      rig_config (with baseline=0.0 for mapping)
      ref_idx
      flipped_convention (bool)  -> for baseline sign auto-flip if desired
    """
    if override_pairs is not None:
        cams_from_pano_rotation = []
        for pitch_deg, yaw_deg in override_pairs:
            R = Rotation.from_euler("YX", [yaw_deg, pitch_deg], degrees=True).as_matrix()
            cams_from_pano_rotation.append(R)
        ref_idx = 0 if override_ref_idx is None else override_ref_idx
        logging.info(f"Loaded {len(cams_from_pano_rotation)} rotations from override (ref_idx={ref_idx}).")
    else:
        cams_from_pano_rotation = get_virtual_rotations()
        ref_idx = 0
        logging.info(f"Using built-in get_virtual_rotations() (ref_idx={ref_idx}).")

    flipped = False
    if override_pairs is not None:
        flipped = _should_flip_baseline_from_pairs(override_pairs, ref_idx)

    # Mapping baseline is forced to 0.0
    rig_config = create_pano_rig_config(cams_from_pano_rotation, ref_idx=ref_idx, baseline=0.0)

    cam_centers_in_pano = np.einsum("nij,i->nj", cams_from_pano_rotation, [0, 0, 1])

    camera = pano_size = rays_in_cam = None
    for pano_name in tqdm(pano_image_names):
        pano_path = pano_image_dir / pano_name
        try:
            pano_image = PIL.Image.open(pano_path)
        except PIL.Image.UnidentifiedImageError:
            logging.info(f"Skipping file {pano_path} as it cannot be read.")
            continue

        pano_exif = pano_image.getexif()
        pano_image = np.asarray(pano_image)
        gpsonly_exif = PIL.Image.Exif()
        gpsonly_exif[PIL.ExifTags.IFD.GPSInfo] = pano_exif.get_ifd(PIL.ExifTags.IFD.GPSInfo)

        pano_height, pano_width, *_ = pano_image.shape
        if pano_width != pano_height * 2:
            raise ValueError("Only 360° panoramas are supported.")

        if camera is None:
            camera = create_virtual_camera(pano_height)
            for rig_camera in rig_config.cameras:
                rig_camera.camera = camera
            pano_size = (pano_width, pano_height)
            rays_in_cam = get_virtual_camera_rays(camera)
        else:
            if (pano_width, pano_height) != pano_size:
                raise ValueError("Panoramas of different sizes are not supported.")

        for cam_idx, cam_from_pano_r in enumerate(cams_from_pano_rotation):
            rays_in_pano = rays_in_cam @ cam_from_pano_r
            xy_in_pano = spherical_img_from_cam(pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(camera.width, camera.height, 2).astype(np.float32)
            xy_in_pano -= 0.5
            image = cv2.remap(
                pano_image,
                *np.moveaxis(xy_in_pano, -1, 0),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )

            closest_camera = np.argmax(rays_in_pano @ cam_centers_in_pano.T, -1)
            mask = (((closest_camera == cam_idx) * 255).astype(np.uint8)).reshape(camera.width, camera.height)

            image_name = rig_config.cameras[cam_idx].image_prefix + pano_name
            mask_name = f"{image_name}.png"

            image_path = output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            PIL.Image.fromarray(image).save(image_path, exif=gpsonly_exif)

            mask_path = mask_dir / mask_name
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            if not pycolmap.Bitmap.from_array(mask).write(mask_path):
                raise RuntimeError(f"Cannot write {mask_path}")

    return rig_config, ref_idx, flipped


# -----------------------------------------------------------------------------
# Text model parsing / writing
# -----------------------------------------------------------------------------

def strip_ext(name: str) -> str:
    return str(Path(name).with_suffix("")).replace("\\", "/")


def parse_images_txt(images_txt_path: Path):
    lines = images_txt_path.read_text(encoding="utf-8").splitlines()
    records = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if i + 1 >= len(lines):
            raise RuntimeError(f"Malformed images.txt near line {i + 1}: missing points2D line.")
        pose_line = lines[i]
        points2d_line = lines[i + 1]
        parts = pose_line.split()
        if len(parts) < 10:
            raise RuntimeError(f"Malformed pose line in images.txt:\n{pose_line}")

        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        camera_id = int(parts[8])
        name = " ".join(parts[9:])

        records.append(
            dict(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                points2d_line=points2d_line,
            )
        )
        i += 2
    return records


def write_images_txt(records, out_path: Path):
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(records)}, mean observations per image: 0\n")
        for rec in records:
            q = rec["qvec"]
            t = rec["tvec"]
            f.write(
                f'{rec["image_id"]} '
                f'{q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} '
                f'{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} '
                f'{rec["camera_id"]} {rec["name"]}\n'
            )
            f.write(rec["points2d_line"].rstrip("\n") + "\n")


def parse_points3D_txt(points3D_txt_path: Path):
    lines = points3D_txt_path.read_text(encoding="utf-8").splitlines()
    points = []
    for line in lines:
        striped = line.strip()
        if not striped or striped.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        point3d_id = int(parts[0])
        xyz = np.array(list(map(float, parts[1:4])), dtype=np.float64)
        rgb = list(map(int, parts[4:7]))
        error = float(parts[7])
        track_elems = parts[8:]
        points.append(
            dict(
                point3d_id=point3d_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                track_elems=track_elems,
            )
        )
    return points


def write_points3D_txt(points, out_path: Path):
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 0\n")
        for p in points:
            xyz = p["xyz"]
            rgb = p["rgb"]
            track_str = " ".join(p["track_elems"]) if p["track_elems"] else ""
            line = (
                f'{p["point3d_id"]} '
                f'{xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g} '
                f'{rgb[0]} {rgb[1]} {rgb[2]} '
                f'{p["error"]:.17g}'
            )
            if track_str:
                line += " " + track_str
            f.write(line + "\n")


# -----------------------------------------------------------------------------
# Pose math (COLMAP convention)
# -----------------------------------------------------------------------------

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    COLMAP qvec: [qw, qx, qy, qz]
    scipy quat:  [qx, qy, qz, qw]
    """
    qvec = np.asarray(qvec, dtype=np.float64)
    qxyzw = np.array([qvec[1], qvec[2], qvec[3], qvec[0]], dtype=np.float64)
    return Rotation.from_quat(qxyzw).as_matrix()


def camera_center_from_qt(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    COLMAP stores world->cam: X_cam = R * X_world + t
    Camera center in world: C = -R^T t
    """
    Rm = qvec_to_rotmat(qvec)
    return -Rm.T @ tvec


def tvec_from_q_center(qvec: np.ndarray, center: np.ndarray) -> np.ndarray:
    Rm = qvec_to_rotmat(qvec)
    return -Rm @ center


# -----------------------------------------------------------------------------
# Name parsing: stream + frame index
# -----------------------------------------------------------------------------

def parse_camidx_and_stem(name: str):
    """
    Supports:
      pano_camera0/000030.png
      pano_camera_0/000030.png
      pano_camera0/ShortTest(1)_frame_00030.png
      ShortTest(1)_frame_00030_pano_camera_0
    Returns (cam_idx, stem_without_ext) or (None, None)
    """
    raw = name.replace("\\", "/").strip()
    raw_noext = strip_ext(raw)

    m = re.match(r"^pano_camera_?(\d+)/(.*)$", raw_noext, flags=re.IGNORECASE)
    if m:
        cam_idx = int(m.group(1))
        stem = strip_ext(Path(m.group(2)).name)
        return cam_idx, stem

    m = re.match(r"^(.*?)[_/]?pano_camera_?(\d+)$", raw_noext, flags=re.IGNORECASE)
    if m:
        stem = strip_ext(Path(m.group(1)).name)
        cam_idx = int(m.group(2))
        return cam_idx, stem

    return None, None


def build_group_key_from_name(name: str):
    """
    Group key for "same panorama frame across all pano_cameraX".
    For pano_camera0/000030.png -> key "000030"
    For pano_camera0/ShortTest(1)_frame_00030.png -> key "shorttest(1)_frame_00030"
    """
    cam_idx, stem = parse_camidx_and_stem(name)
    if cam_idx is None or stem is None:
        return None
    return stem.lower().strip()


def parse_stream_and_frame_index(name: str):
    """
    Supports:
      pano_camera0/ShortTest(1)_frame_00020.png
      pano_camera0/ShortTest(1)_frame_00020_pano_camera_0.png
      pano_camera0/000020.png
      frame_000020.jpg
      000020.jpg

    Returns (stream_name, frame_idx)
    stream_name is normalized with underscores removed: pano_camera0 -> panocamera0
    """
    raw = name.replace("\\", "/").strip()
    raw_noext = strip_ext(raw)

    # Case 1: stream folder
    m = re.match(r"^(pano_camera_?\d+)/(.*)$", raw_noext, flags=re.IGNORECASE)
    if m:
        stream_name = m.group(1).lower().replace("_", "")
        basename = Path(m.group(2)).name

        fm = re.search(r"frame_(\d+)", basename, flags=re.IGNORECASE)
        if fm:
            return stream_name, int(fm.group(1))

        fm = re.fullmatch(r"(\d+)", basename)
        if fm:
            return stream_name, int(fm.group(1))

        fm = re.search(r"(\d+)", basename)
        if fm:
            return stream_name, int(fm.group(1))

        return stream_name, None

    # Case 2: no stream folder
    basename = Path(raw_noext).name

    fm = re.search(r"frame_(\d+)", basename, flags=re.IGNORECASE)
    if fm:
        return "defaultstream", int(fm.group(1))

    fm = re.fullmatch(r"(\d+)", basename)
    if fm:
        return "defaultstream", int(fm.group(1))

    fm = re.search(r"(\d+)", basename)
    if fm:
        return "defaultstream", int(fm.group(1))

    return None, None


# -----------------------------------------------------------------------------
# Scale normalization: consecutive distances on one stream
# -----------------------------------------------------------------------------

def collect_consecutive_stream_distances(records, stream_name: str):
    """
    Collect consecutive-frame camera-center distances for ONE stream.
    Only uses pairs where frame indices are exactly consecutive (k -> k+1).

    Returns:
      ordered_pairs: list[dict]
      distances: np.ndarray
    """
    requested_key = stream_name.lower().replace("_", "")

    items = []
    for rec in records:
        rec_stream, frame_idx = parse_stream_and_frame_index(rec["name"])
        if rec_stream == requested_key and frame_idx is not None:
            items.append((frame_idx, rec))

    items.sort(key=lambda x: x[0])

    if len(items) < 2:
        # helpful debug preview
        preview = "\n".join(f"  - {r['name']}" for r in records[:30])
        raise RuntimeError(
            f"Could not find enough images for stream '{stream_name}' to compute consecutive distances.\n"
            f"First names seen in images.txt:\n{preview}"
        )

    ordered_pairs = []
    distances = []

    for i in range(len(items) - 1):
        idx_a, rec_a = items[i]
        idx_b, rec_b = items[i + 1]

        if idx_b != idx_a + 1:
            continue

        Ca = camera_center_from_qt(rec_a["qvec"], rec_a["tvec"])
        Cb = camera_center_from_qt(rec_b["qvec"], rec_b["tvec"])
        d = float(np.linalg.norm(Cb - Ca))

        if d > 0:
            ordered_pairs.append(
                dict(
                    frame_a=idx_a,
                    frame_b=idx_b,
                    name_a=rec_a["name"],
                    name_b=rec_b["name"],
                    distance=d,
                )
            )
            distances.append(d)

    if not distances:
        raise RuntimeError(f"No valid consecutive distances found for stream '{stream_name}'.")

    return ordered_pairs, np.array(distances, dtype=np.float64)


def compute_pivot_from_stream(records, stream_name: str) -> np.ndarray:
    """
    Shared scaling pivot P = median camera center of chosen stream (robust).
    """
    requested_key = stream_name.lower().replace("_", "")
    centers = []

    for rec in records:
        rec_stream, frame_idx = parse_stream_and_frame_index(rec["name"])
        if rec_stream == requested_key and frame_idx is not None:
            centers.append(camera_center_from_qt(rec["qvec"], rec["tvec"]))

    if len(centers) < 2:
        # fallback: all cameras
        centers = [camera_center_from_qt(r["qvec"], r["tvec"]) for r in records]

    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[0] < 1 or centers.shape[1] != 3:
        raise RuntimeError("Could not compute pivot (no camera centers found).")

    return np.median(centers, axis=0)


def apply_uniform_scale(records, points, scale_factor: float, pivot: np.ndarray):
    """
    Scale BOTH camera centers and 3D points about ONE shared pivot P.

      C' = P + s (C - P)
      X' = P + s (X - P)
    """
    pivot = np.asarray(pivot, dtype=np.float64).reshape(3)

    # cameras
    for rec in records:
        C = camera_center_from_qt(rec["qvec"], rec["tvec"])
        C_scaled = pivot + scale_factor * (C - pivot)
        rec["tvec"] = tvec_from_q_center(rec["qvec"], C_scaled)

    # points
    for p in points:
        p["xyz"] = pivot + scale_factor * (p["xyz"] - pivot)


def apply_export_baseline_to_records(records, ref_idx: int, export_baseline: float):
    """
    After scaling, apply a final baseline offset to non-ref rig cameras.
    This modifies ONLY camera translations, not points.

    Conventions:
      cam_idx 1–4 => side = +1
      cam_idx 5+  => side = -1
    """
    if abs(export_baseline) < 1e-12:
        logging.info("Export baseline is ~0, skipping final baseline application.")
        return

    groups: dict[str, dict[int, dict]] = {}
    for rec in records:
        key = build_group_key_from_name(rec["name"])
        cam_idx, _ = parse_camidx_and_stem(rec["name"])
        if key is None or cam_idx is None:
            continue
        groups.setdefault(key, {})[cam_idx] = rec

    total_adjusted = 0
    total_groups = 0

    for _, group in groups.items():
        if ref_idx not in group:
            continue
        total_groups += 1

        ref_rec = group[ref_idx]
        R_ref = qvec_to_rotmat(ref_rec["qvec"])
        C_ref = camera_center_from_qt(ref_rec["qvec"], ref_rec["tvec"])

        for cam_idx, rec in group.items():
            if cam_idx == ref_idx:
                continue

            side = 1 if cam_idx <= 4 else -1
            center_offset_ref = np.array([export_baseline * side, 0.0, 0.0], dtype=np.float64)
            center_offset_world = R_ref.T @ center_offset_ref
            C_new = C_ref + center_offset_world

            rec["tvec"] = tvec_from_q_center(rec["qvec"], C_new)
            total_adjusted += 1

    logging.info(
        f"Applied final export baseline {export_baseline:+.6f} to {total_adjusted} images "
        f"across {total_groups} pano groups."
    )


def adjust_text_model_stream_based(
    raw_model_dir: Path,
    final_model_dir: Path,
    scale_camera_stream: str,
    walking_speed_mps: float,
    fps: float,
    ref_idx: int,
    export_baseline: float,
    keep_raw_text_model: bool = False,
):
    images_txt = raw_model_dir / "images.txt"
    cameras_txt = raw_model_dir / "cameras.txt"
    points3D_txt = raw_model_dir / "points3D.txt"

    if not images_txt.exists():
        raise FileNotFoundError(f"Missing {images_txt}")
    if not cameras_txt.exists():
        raise FileNotFoundError(f"Missing {cameras_txt}")
    if not points3D_txt.exists():
        raise FileNotFoundError(f"Missing {points3D_txt}")

    if fps <= 0:
        raise ValueError("--fps must be > 0")
    if walking_speed_mps <= 0:
        raise ValueError("--walking_speed_mps must be > 0")

    target_step = float(walking_speed_mps) / float(fps)

    records = parse_images_txt(images_txt)
    points = parse_points3D_txt(points3D_txt)

    ordered_pairs, distances = collect_consecutive_stream_distances(records, scale_camera_stream)
    measured_median = float(np.median(distances))
    if measured_median <= 0:
        raise RuntimeError("Measured median distance is <= 0, cannot compute scale factor.")

    scale_factor = target_step / measured_median

    pivot = compute_pivot_from_stream(records, scale_camera_stream)
    logging.info(
        f"Scale stream: {scale_camera_stream}\n"
        f"Consecutive pair count: {len(distances)}\n"
        f"Target step (walking_speed_mps / fps): {walking_speed_mps:.6f} / {fps:.6f} = {target_step:.6f}\n"
        f"Measured median consecutive camera-center distance: {measured_median:.8f}\n"
        f"Scale factor: {scale_factor:.8f}\n"
        f"Scaling pivot (median camera center): [{pivot[0]:.6f}, {pivot[1]:.6f}, {pivot[2]:.6f}]"
    )

    # Preview first few pairs
    preview_n = min(5, len(ordered_pairs))
    for i in range(preview_n):
        pair = ordered_pairs[i]
        logging.info(
            f"Pair {i+1}: {pair['name_a']} -> {pair['name_b']} "
            f"(frames {pair['frame_a']} -> {pair['frame_b']})  d = {pair['distance']:.8f}"
        )

    apply_uniform_scale(records, points, scale_factor, pivot)

    # verify
    _, distances_after = collect_consecutive_stream_distances(records, scale_camera_stream)
    logging.info(f"Median consecutive distance after uniform scaling: {float(np.median(distances_after)):.8f}")

    apply_export_baseline_to_records(records, ref_idx=ref_idx, export_baseline=export_baseline)

    # baseline should not affect pano_camera0 itself, but we re-check anyway
    _, distances_final = collect_consecutive_stream_distances(records, scale_camera_stream)
    logging.info(f"Final median consecutive distance after export baseline: {float(np.median(distances_final)):.8f}")

    final_model_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy2(cameras_txt, final_model_dir / "cameras.txt")
    write_images_txt(records, final_model_dir / "images.txt")
    write_points3D_txt(points, final_model_dir / "points3D.txt")

    if not keep_raw_text_model:
        shutil.rmtree(raw_model_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run(args):
    image_dir = args.output_path / "images"
    mask_dir = args.output_path / "masks"
    image_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)

    database_path = args.output_path / "database.db"
    if database_path.exists():
        database_path.unlink()

    rec_path = args.output_path / "sparse"
    rec_path.mkdir(exist_ok=True, parents=True)

    pano_image_dir = args.input_image_path
    pano_image_names = sorted(
        p.relative_to(pano_image_dir).as_posix()
        for p in pano_image_dir.rglob("*")
        if not p.is_dir()
    )
    logging.info(f"Found {len(pano_image_names)} images in {pano_image_dir}.")

    override_pairs, override_ref_idx = load_rotation_override_if_any(pano_image_dir)

    rig_config, ref_idx, flipped = render_perspective_images(
        pano_image_names,
        pano_image_dir,
        image_dir,
        mask_dir,
        override_pairs=override_pairs,
        override_ref_idx=override_ref_idx,
    )

    # export baseline (user value) optionally auto-flipped by convention
    export_baseline = float(args.export_baseline_m)
    if args.baseline_auto_flip and flipped:
        export_baseline = -export_baseline

    logging.info(
        f"Mapping baseline: 0.000000 (forced)\n"
        f"Export baseline: {export_baseline:+.6f} (auto_flip={args.baseline_auto_flip}, flipped={flipped})"
    )

    pycolmap.set_random_seed(0)

    threads = int(os.environ.get("COLMAP_THREADS", "12"))
    threads = max(1, int(threads))

    # Feature extraction
    extraction_options = pycolmap.SiftExtractionOptions()
    extraction_options.use_gpu = True
    extraction_options.gpu_index = "0"
    if hasattr(extraction_options, "num_threads"):
        extraction_options.num_threads = threads

    pycolmap.extract_features(
        database_path,
        image_dir,
        reader_options={"mask_path": mask_dir},
        sift_options=extraction_options,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
    )

    with pycolmap.Database(database_path) as db:
        pycolmap.apply_rig_config([rig_config], db)

    # Matching
    matching_options = pycolmap.SiftMatchingOptions()
    matching_options.use_gpu = True
    matching_options.gpu_index = "0"
    if hasattr(matching_options, "num_threads"):
        matching_options.num_threads = threads

    matcher = args.matcher
    logging.info(f"Feature matching method: {matcher}")

    if matcher == "sequential":
        seq_opts = pycolmap.SequentialMatchingOptions(loop_detection=True)
        if hasattr(seq_opts, "num_threads"):
            seq_opts.num_threads = threads
        pycolmap.match_sequential(
            database_path,
            sift_options=matching_options,
            matching_options=seq_opts,
        )
    else:
        # exhaustive by default
        if hasattr(pycolmap, "ExhaustiveMatchingOptions"):
            ex_opts = pycolmap.ExhaustiveMatchingOptions()
            if hasattr(ex_opts, "num_threads"):
                ex_opts.num_threads = threads
            pycolmap.match_exhaustive(
                database_path,
                sift_options=matching_options,
                matching_options=ex_opts,
            )
        else:
            pycolmap.match_exhaustive(
                database_path,
                sift_options=matching_options,
            )

    # Mapping
    opts = pycolmap.IncrementalPipelineOptions(
        ba_refine_sensor_from_rig=False,
        ba_refine_focal_length=False,
        ba_refine_principal_point=False,
        ba_refine_extra_params=False,
    )

    recs = pycolmap.incremental_mapping(database_path, image_dir, rec_path, opts)
    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")

        final_out_dir = rec_path / str(idx)
        final_out_dir.mkdir(exist_ok=True, parents=True)

        raw_text_dir = rec_path / f"{idx}_raw_text"
        if raw_text_dir.exists():
            shutil.rmtree(raw_text_dir, ignore_errors=True)
        raw_text_dir.mkdir(exist_ok=True, parents=True)

        logging.info(f"Writing temporary raw text COLMAP model to {raw_text_dir}")
        rec.write_text(str(raw_text_dir))

        logging.info(
            "Adjusting raw text model using consecutive-frame stream scaling:\n"
            f"  scale stream     : {args.scale_camera_stream}\n"
            f"  walking speed    : {args.walking_speed_mps}\n"
            f"  fps              : {args.fps}\n"
            f"  target step      : {args.walking_speed_mps / args.fps}\n"
            f"  export baseline  : {export_baseline:+.6f}\n"
            f"  ref_idx          : {ref_idx}"
        )

        adjust_text_model_stream_based(
            raw_model_dir=raw_text_dir,
            final_model_dir=final_out_dir,
            scale_camera_stream=args.scale_camera_stream,
            walking_speed_mps=args.walking_speed_mps,
            fps=args.fps,
            ref_idx=ref_idx,
            export_baseline=export_baseline,
            keep_raw_text_model=args.keep_raw_text_model,
        )

        logging.info(f"Final adjusted text COLMAP model written to {final_out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)

    parser.add_argument(
        "--matcher",
        default="exhaustive",
        choices=["sequential", "exhaustive"],
        help="Matching method",
    )

    # Scale normalization controls
    parser.add_argument(
        "--scale_camera_stream",
        type=str,
        default="pano_camera0",
        help="Which rendered camera stream to use for consecutive-frame scale normalization.",
    )
    parser.add_argument(
        "--walking_speed_mps",
        type=float,
        default=0.7,
        help="Assumed walking speed in meters per second.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame sampling rate (frames per second) used for the input panorama sequence.",
    )

    # Export baseline controls
    parser.add_argument(
        "--export_baseline_m",
        type=float,
        default=0.015,
        help="Final export-time baseline offset (meters). Mapping uses baseline=0.",
    )
    parser.add_argument(
        "--baseline_auto_flip",
        action="store_true",
        help="If set, auto-flip baseline sign when rotation_override indicates flipped convention.",
    )

    parser.add_argument(
        "--keep_raw_text_model",
        action="store_true",
        help="Keep the temporary pre-adjustment raw text model folder for debugging.",
    )

    run(parser.parse_args())