import os
import pandas as pd
from ultralytics import YOLO
from PIL import Image


MODEL_PATH = 'runs/train/doc_qc_model/weights/best.pt'
INPUT_FOLDER = 'input_tifs'
OUTPUT_FOLDER = 'output'
REPORT_PATH = os.path.join(OUTPUT_FOLDER, 'results.csv')

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def run_qc():
    model = YOLO(MODEL_PATH)
    results_list = []

    for fname in os.listdir(INPUT_FOLDER):
        if fname.lower().endswith(".tif"):
            fpath = os.path.join(INPUT_FOLDER, fname)
            results = model(fpath)

            defects = []
            for box in results[0].boxes:
                cls_name = results[0].names[int(box.cls)]
                conf = float(box.conf)
                defects.append((cls_name, round(conf, 2)))

            if defects:
                results[0].save(filename=os.path.join(OUTPUT_FOLDER, f"{fname}.jpg"))
                print(f"Defects found in {fname}: {defects}")

            else:
                print(f" {fname}: Clean scan.")
            
            results_list.append({
                "file": fname,
                "defects": [d[0] for d in defects],
                "confidences": [d[1] for d in defects]
            })

    df = pd.DataFrame(results_list)
    df.to_csv(REPORT_PATH, index=False)
    print(f"QC report saved to {REPORT_PATH}")
    print("Process Complete")

if __name__ == "__main__":
    run_qc()