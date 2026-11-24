from pathlib import Path
import csv

ROOT = Path("cv_project/dataset/raw/extracted/construction-ppe")
OUTPUT = Path("cv_project/dataset/candidate_images.csv")

CLASSES = {
    "0": "helmet",
    "5": "person",
    "6": "no_helmet"
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

rows = []

for split in ["train", "val"]:

    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    for image in sorted(image_dir.iterdir()):

        if image.suffix.lower() not in EXTENSIONS:
            continue

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            continue

        counts = {name: 0 for name in CLASSES.values()}

        for line in label.read_text(
            encoding="utf-8-sig"
        ).splitlines():

            parts = line.split()

            if len(parts) == 5 and parts[0] in CLASSES:
                counts[CLASSES[parts[0]]] += 1

        if sum(counts.values()) == 0:
            continue

        rows.append({
            "image": str(image.resolve()),
            "label": str(label.resolve()),
            "original_split": split,
            **counts,
            "total_objects": sum(counts.values())
        })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n===== DATASET CANDIDATE REPORT =====")
print("Total candidate images:", len(rows))

for name in CLASSES.values():
    print(
        name, "images:",
        sum(1 for row in rows if row[name] > 0)
    )

print("Total selected objects:",
      sum(row["total_objects"] for row in rows))

print("\nCSV saved:", OUTPUT)
