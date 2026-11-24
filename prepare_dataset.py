from pathlib import Path
import random
import shutil
import sys

SOURCE = Path("cv_project/dataset/raw/extracted/construction-ppe")
OUTPUT = Path("cv_project/dataset/helmet_detection")

CLASS_MAP = {
    5: 0,  # Person
    0: 1,  # Helmet
    6: 2   # No Helmet
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not SOURCE.exists():
    sys.exit(f"Source dataset not found: {SOURCE}")

if OUTPUT.exists():
    sys.exit(f"Output already exists. Not overwriting: {OUTPUT}")

records = []

for source_split in ["train", "val"]:
    image_dir = SOURCE / "images" / source_split
    label_dir = SOURCE / "labels" / source_split

    if not image_dir.exists() or not label_dir.exists():
        sys.exit(f"Missing image or label folder for: {source_split}")

    for image in sorted(image_dir.iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            sys.exit(f"Missing label file: {label}")

        filtered_lines = []

        for line_number, line in enumerate(label.read_text().splitlines(), 1):
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) != 5:
                sys.exit(f"Invalid YOLO label: {label}, line {line_number}")

            try:
                old_class = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
            except ValueError:
                sys.exit(f"Invalid label values: {label}, line {line_number}")

            if old_class in CLASS_MAP:
                if not all(0 <= value <= 1 for value in coordinates):
                    sys.exit(f"Invalid coordinates: {label}, line {line_number}")

                new_class = CLASS_MAP[old_class]
                filtered_lines.append(
                    str(new_class) + " " + " ".join(parts[1:])
                )

        records.append((source_split, image, filtered_lines))

if not records:
    sys.exit("No images found.")

random.Random(42).shuffle(records)

total = len(records)
train_end = int(total * 0.70)
val_end = train_end + int(total * 0.20)

splits = {
    "train": records[:train_end],
    "val": records[train_end:val_end],
    "test": records[val_end:]
}

try:
    for split, items in splits.items():
        image_out = OUTPUT / "images" / split
        label_out = OUTPUT / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for source_split, image, lines in items:
            filename = f"{source_split}_{image.name}"
            shutil.copy2(image, image_out / filename)
            (label_out / f"{Path(filename).stem}.txt").write_text(
                "\n".join(lines),
                encoding="utf-8"
            )

    yaml_text = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: Person
  1: Helmet
  2: No Helmet
"""
    (OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

except Exception:
    shutil.rmtree(OUTPUT, ignore_errors=True)
    raise

print("\nDATASET CREATED SUCCESSFULLY")
print("Total images:", total)

for split, items in splits.items():
    class_images = [0, 0, 0]
    class_objects = [0, 0, 0]

    for _, _, lines in items:
        found = set()
        for line in lines:
            class_id = int(line.split()[0])
            class_objects[class_id] += 1
            found.add(class_id)

        for class_id in found:
            class_images[class_id] += 1

    print(f"\n{split.upper()}: {len(items)} images")
    print("Images containing Person, Helmet, No Helmet:", class_images)
    print("Object counts Person, Helmet, No Helmet:", class_objects)

print("\nDataset location:", OUTPUT.resolve())
print("YAML:", (OUTPUT / "data.yaml").resolve())
