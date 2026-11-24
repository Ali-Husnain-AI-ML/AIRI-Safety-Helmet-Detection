from pathlib import Path
import random
import cv2

root = Path("cv_project/dataset/helmet_detection")
output = Path("cv_project/dataset/audit_preview")
output.mkdir(parents=True, exist_ok=True)

classes = {
    0: ("Person", (255, 0, 0)),
    1: ("Helmet", (0, 255, 255)),
    2: ("No Helmet", (0, 0, 255))
}

records = []

for img in (root / "images" / "train").iterdir():
    label = root / "labels" / "train" / (img.stem + ".txt")
    if not label.exists():
        continue

    ids = set()
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            try:
                ids.add(int(parts[0]))
            except ValueError:
                pass

    records.append((img, label, ids))

random.seed(42)
selected = []

# Include examples of all three classes
for class_id in [2, 0, 1]:
    pool = [r for r in records if class_id in r[2] and r not in selected]
    random.shuffle(pool)
    selected.extend(pool[:4])

# Fill remaining spaces if necessary
remaining = [r for r in records if r not in selected]
random.shuffle(remaining)
selected.extend(remaining[:max(0, 12 - len(selected))])
selected = selected[:12]

tile_w, tile_h = 480, 300
canvas = 255 * __import__("numpy").ones((tile_h * 3, tile_w * 4, 3), dtype="uint8")

for index, (img_path, label_path, _) in enumerate(selected):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        cid = int(parts[0])
        if cid not in classes:
            continue

        xc, yc, bw, bh = map(float, parts[1:])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        name, color = classes[cid]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (max(0, x1), max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    scale = min(tile_w / w, (tile_h - 25) / h)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    x = (index % 4) * tile_w
    y = (index // 4) * tile_h

    canvas[y:y+25, x:x+tile_w] = 235
    cv2.putText(canvas, img_path.name[:48], (x+8, y+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    rh, rw = resized.shape[:2]
    canvas[y+25:y+25+rh, x:x+rw] = resized

file = output / "dataset_preview.jpg"
cv2.imwrite(str(file), canvas)

print("Preview created:", file.resolve())
print("Images displayed:", len(selected))
