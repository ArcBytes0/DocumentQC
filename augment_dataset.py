import cv2
import numpy as np
import os
import random

# --- Config ---
INPUT_DIR = "clean_images"
OUTPUT_IMAGE_DIR = "data/train/images"
OUTPUT_LABEL_DIR = "data/train/labels"
IMG_SIZE = (1440, 1024)  # (height, width)
AUGMENT_COUNT_PER_IMAGE = 25  # how many variations per base image

# YOLO class mapping
CLASSES = {
    "dogear": 0,
    "scan_line": 1,
    "fold": 2,
    "sticky_note": 3,
}

# --- Create output dirs ---
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# --- Utility functions ---
def normalize_bbox(x, y, w, h, img_w, img_h):
    """Convert to YOLO normalized coordinates"""
    return (
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h,
    )

def add_scan_lines(image, boxes, img_w, img_h):
    img = image.copy()
    for _ in range(random.randint(1, 3)):
        y = random.randint(0, img.shape[0] - 1)
        thickness = random.randint(1, 3)
        color = random.randint(0, 50)
        cv2.line(img, (0, y), (img.shape[1], y), (color, color, color), thickness)
        x, w = 0, img_w
        h = thickness
        boxes.append((CLASSES["scan_line"], *normalize_bbox(x, y, w, h, img_w, img_h)))
    return img, boxes

def add_sticky_note(image, boxes, img_w, img_h):
    img = image.copy()
    h, w, _ = img.shape
    note_h, note_w = random.randint(100, 200), random.randint(100, 200)
    x, y = random.randint(0, w - note_w), random.randint(0, h - note_h)
    color = (random.randint(180, 255), random.randint(180, 255), random.randint(0, 80))
    cv2.rectangle(img, (x, y), (x + note_w, y + note_h), color, -1)
    boxes.append((CLASSES["sticky_note"], *normalize_bbox(x, y, note_w, note_h, img_w, img_h)))
    return img, boxes

def add_dogear(image, boxes, img_w, img_h):
    img = image.copy()
    h, w, _ = img.shape
    corner = random.choice(["tl", "tr", "bl", "br"])
    size = random.randint(50, 120)
    overlay = img.copy()
    pts = {
        "tl": np.array([[0, 0], [size, 0], [0, size]], np.int32),
        "tr": np.array([[w, 0], [w - size, 0], [w, size]], np.int32),
        "bl": np.array([[0, h], [0, h - size], [size, h]], np.int32),
        "br": np.array([[w, h], [w - size, h], [w, h - size]], np.int32),
    }
    cv2.fillConvexPoly(overlay, pts[corner], (200, 200, 200))
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    # bounding box for YOLO
    if corner == "tl":
        x, y, bw, bh = 0, 0, size, size
    elif corner == "tr":
        x, y, bw, bh = w - size, 0, size, size
    elif corner == "bl":
        x, y, bw, bh = 0, h - size, size, size
    else:
        x, y, bw, bh = w - size, h - size, size, size

    boxes.append((CLASSES["dogear"], *normalize_bbox(x, y, bw, bh, img_w, img_h)))
    return img, boxes

def add_fold(image, boxes, img_w, img_h):
    img = image.copy()
    h, w, _ = img.shape
    orientation = random.choice(["horizontal", "vertical"])
    overlay = img.copy()
    if orientation == "horizontal":
        y = random.randint(h // 4, 3 * h // 4)
        cv2.line(overlay, (0, y), (w, y), (220, 220, 220), 2)
        boxes.append((CLASSES["fold"], *normalize_bbox(0, y, w, 2, img_w, img_h)))
    else:
        x = random.randint(w // 4, 3 * w // 4)
        cv2.line(overlay, (x, 0), (x, h), (220, 220, 220), 2)
        boxes.append((CLASSES["fold"], *normalize_bbox(x, 0, 2, h, img_w, img_h)))
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    return img, boxes

def augment_dataset():
    image_paths = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ]

    print(f"Found {len(image_paths)} base images in {INPUT_DIR}")
    if not image_paths:
        print("⚠️ No images found! Make sure you have .tif files in data/original/")
        return

    for path in image_paths:
        base_name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))  # type: ignore # (width, height)
        img_h, img_w = IMG_SIZE

        for i in range(AUGMENT_COUNT_PER_IMAGE):
            aug_img = img.copy()
            boxes = []

            # Randomly apply defects
            if random.random() < 0.5:
                aug_img, boxes = add_scan_lines(aug_img, boxes, img_w, img_h)
            if random.random() < 0.5:
                aug_img, boxes = add_sticky_note(aug_img, boxes, img_w, img_h)
            if random.random() < 0.3:
                aug_img, boxes = add_dogear(aug_img, boxes, img_w, img_h)
            if random.random() < 0.3:
                aug_img, boxes = add_fold(aug_img, boxes, img_w, img_h)

            # save image + label
            image_out = os.path.join(OUTPUT_IMAGE_DIR, f"{base_name}_aug{i}.tif")
            label_out = os.path.join(OUTPUT_LABEL_DIR, f"{base_name}_aug{i}.txt")

            cv2.imwrite(image_out, aug_img)

            with open(label_out, "w") as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

            print(f"✅ Saved: {os.path.basename(image_out)} with {len(boxes)} labels")

    print("\n✨ Augmentation complete!")
    print(f"Check: {OUTPUT_IMAGE_DIR} and {OUTPUT_LABEL_DIR}")

# --- Run ---
if __name__ == "__main__":
    augment_dataset()
