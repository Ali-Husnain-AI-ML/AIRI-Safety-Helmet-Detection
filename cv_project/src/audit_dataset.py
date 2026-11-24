from pathlib import Path

ROOT = Path("cv_project/dataset/raw/extracted/construction-ppe")
CLASSES = {
    "0": "helmet",
    "5": "person",
    "6": "no_helmet"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "val"]:

    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))
    label_map = {p.stem: p for p in labels}

    image_counts = {cid: 0 for cid in CLASSES}
    object_counts = {cid: 0 for cid in CLASSES}

    missing_labels = 0
    empty_labels = 0
    invalid_rows = 0

    for image_path in images:

        label_path = label_map.get(image_path.stem)

        if label_path is None:
            missing_labels += 1
            continue

        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels += 1
            continue

        found = set()

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                invalid_rows += 1
                continue

            try:
                values = [float(x) for x in parts]
            except ValueError:
                invalid_rows += 1
                continue

            if any(not 0 <= x <= 1 for x in values[1:]):
                invalid_rows += 1
                continue

            class_id = parts[0]

            if class_id in CLASSES:
                object_counts[class_id] += 1
                found.add(class_id)

        for class_id in found:
            image_counts[class_id] += 1

    image_stems = {p.stem for p in images}
    orphan_labels = sum(
        1 for p in labels if p.stem not in image_stems
    )

    print(f"\n========== {split.upper()} ==========")
    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Missing labels:", missing_labels)
    print("Orphan labels:", orphan_labels)
    print("Empty labels:", empty_labels)
    print("Invalid annotation rows:", invalid_rows)

    for cid, name in CLASSES.items():
        print(
            f"{name}: {image_counts[cid]} images, "
            f"{object_counts[cid]} objects"
        )
