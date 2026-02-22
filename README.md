# DocumentQC

Lightweight pipeline to run YOLO-based document quality control (QC) on TIFF scans, save annotated output images and a CSV report.

## Features
- Run a pretrained YOLO model on TIFF batches
- Save annotated images for files with detected defects
- Generate a CSV summary of detected classes and confidences

## Prerequisites
- Windows 10/11
- Python 3.9+ (recommended)
- GPU drivers/CUDA if using GPU inference with ultralytics (optional)

## Setup (recommended)
1. Open an integrated terminal in VS Code.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip

3. Install required packages:

 - Optional: create a requirements.txt:

4. In VS Code, select the .venv interpreter: Command Palette → Python: Select Interpreter → choose .venv\Scripts\python.exe.

## Project layout
- input_tifs/ — put input .tif files here
- output/ — annotated images and CSV report are written here
- runs/train/doc_qc_model/weights/best.pt — expected model path (change MODEL_PATH in scrript if different)
- detect_qc.py — main script
- .gitignore — ignores data, runs, weights, outputs

### Usage
1. Place TIFF files in input_tifs.
2. Ensure model exists at runs/train/doc_qc_model/weights/best.pt or update MODEL_PATH in detect_qc.py.
3. Run:
    ```powershell
    python detect_qc.py
4. Results:
 - Annotated JPGs for images with detections are saved to output
 - CSV report saved to output/results.csv

### Configuration
- Edit top-level constants in detect_qc.py:

- MODEL_PATH
- INPUT_FOLDER
- OUTPUT_FOLDER

### Troubleshooting
- "Import ... could not be resolved": ensure VS Code uses the same interpreter where packages are installed; restart the language server (Command Palette → Python: Restart Language Server).
- Permission errors when installing packages: use a project virtualenv or
    ```powershell
    python -m pip install --user ....
- If ultralytics fails to use GPU, verify CUDA/cuDNN and drivers are installed and compatible.

### Notes
- Keep large datasets out of Git. See .gitignore for recommended ignores: data, runs, *.pt, output.
- For production or large-scale usage, consider batching, multiprocessing, or integrating with a job queue.

## License
- Add your preferred license file (e.g., LICENSE) to this repository.