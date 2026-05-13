# VREV 360 SFM Pipeline

VREV 360 SFM Pipeline is a Windows desktop application for converting 360 panoramic video or pre-extracted frames into COLMAP-ready datasets for Gaussian Splat training workflows.

The app is designed to simplify a 360 video-to-COLMAP workflow by providing a GUI for frame extraction, optional enhancement, optional masking, and SFM/COLMAP dataset generation.

## Features

- Drag-and-drop GUI workflow
- 360 panoramic video input
- Pre-extracted frame folder support
- Automatic frame extraction
- Optional Topaz Video AI enhancement workflow
- Optional YOLO-based masking
- Sequential / exhaustive matching options
- COLMAP-ready output format
- Windows desktop build support with PyInstaller

## Requirements

Recommended system:

- Windows 10/11
- Dual-lens 360 camera footage exported as 2:1 panoramic video
- 8K or higher 360 video recommended
- NVIDIA 30-series GPU or better recommended for faster masking and processing
- Intel Core i9-9900K equivalent or better recommended
- Topaz Video AI installed separately if using the enhancement option

## Installation from Source

Clone the repository:

```bash
git clone https://github.com/kjrosscras/360-Video-To-3DGS-Training-Format.git
cd 360-Video-To-3DGS-Training-Format
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate 360pipeline
```

Run the GUI:

```bash
python run_gui.py
```

## Building the Windows App

This project can be packaged with PyInstaller.

Example:

```powershell
conda activate 360pipeline
pyinstaller "VREV 360 Pipeline.spec"
```

The generated app folder will be created in:

```text
dist/VREV 360 Pipeline/
```

## Topaz Video AI

Topaz Video AI is not included with this project.

The enhancement option only works if the user has Topaz Video installed separately. This project is not affiliated with, endorsed by, or sponsored by Topaz Labs.

## License

This project is licensed under the GNU Affero General Public License v3.0 or later.

See the `LICENSE` file for the full license text.

Prebuilt binaries may be distributed or sold as a convenience package. Users who receive a binary should also have access to the corresponding source code under the terms of the AGPL-3.0 license.

## Third-Party Software

This project uses third-party open-source software. See `THIRD_PARTY_NOTICES.md` for a summary of major dependencies and license notes.

## Disclaimer

This project is provided as-is, without warranty. Processing results may vary depending on capture quality, camera settings, lighting, hardware, and selected processing options.
