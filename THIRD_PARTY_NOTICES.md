# Third-Party Notices

VREV 360 SFM Pipeline uses third-party open-source software. Each component remains licensed under its own license.

This notice is provided for convenience and is not a substitute for reviewing each dependency's full license terms.

## Major Dependencies

### Ultralytics / YOLO

This project uses Ultralytics / YOLO for optional masking functionality.

Ultralytics YOLO models and training code are licensed under AGPL-3.0 by default unless using a separate Ultralytics Enterprise License.

Because this project includes YOLO-based functionality, this project is released under AGPL-3.0.

Project:
- Ultralytics YOLO

License:
- AGPL-3.0 by default

### COLMAP

This project uses COLMAP / pycolmap functionality for structure-from-motion and COLMAP dataset generation.

COLMAP itself is licensed under the new BSD license. COLMAP's own documentation notes that third-party dependencies are separately licensed and may affect resulting binary distributions.

Project:
- COLMAP
- pycolmap

License:
- COLMAP: new BSD license
- pycolmap: see pycolmap/COLMAP license terms

### OpenCV / opencv-python

This project uses OpenCV through the `opencv-python` package for image and video processing.

The `opencv-python` package notes that its repository scripts are MIT licensed, OpenCV itself is Apache 2.0 licensed, and wheels may include third-party components such as FFmpeg under LGPL terms.

Project:
- OpenCV
- opencv-python

License:
- opencv-python package scripts: MIT
- OpenCV: Apache 2.0
- bundled FFmpeg in wheels: LGPL 2.1

### PyTorch / Torchvision

This project uses PyTorch and Torchvision for machine learning / inference dependencies used by the masking pipeline.

Project:
- PyTorch
- Torchvision

License:
- BSD-style license

### NumPy

This project uses NumPy for numerical processing.

Project:
- NumPy

License:
- BSD-style license

### SciPy

This project uses SciPy for scientific computing utilities.

Project:
- SciPy

License:
- BSD-style license

### Pillow

This project may use Pillow for image handling.

Project:
- Pillow

License:
- HPND-style / PIL Software License

### tqdm

This project uses tqdm for progress reporting in supporting scripts.

Project:
- tqdm

License:
- MPL-2.0 / MIT-style licensing depending on package version

### Requests

This project uses Requests for HTTP functionality where applicable.

Project:
- Requests

License:
- Apache 2.0

### PyInstaller

This project can be packaged with PyInstaller.

Project:
- PyInstaller

License:
- GPL with PyInstaller bootloader exception

### Python / Tkinter

This project uses Python and Tkinter for the desktop GUI.

Project:
- Python
- Tkinter

License:
- Python Software Foundation License
- Tcl/Tk license terms apply to Tkinter/Tk components

## Topaz Video AI

Topaz Video AI is not bundled with this project.

The app can optionally call a user-installed copy of Topaz Video AI / Topaz CLI tools if the user enables enhancement and has Topaz installed separately.

Topaz Video AI is owned by Topaz Labs. This project is not affiliated with, endorsed by, or sponsored by Topaz Labs.

## NVIDIA / CUDA

Depending on the user's installation and packaged build, GPU acceleration may rely on NVIDIA/CUDA-related runtime components through dependencies such as PyTorch.

NVIDIA software and CUDA components are governed by their own license terms.

## Notes for Binary Distribution

If distributing a prebuilt Windows binary, include:

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- Source code access information
- Build instructions
- Dependency list, such as `environment.yml`

The corresponding source code for this project should be made available to users who receive the binary, consistent with the AGPL-3.0 license.
