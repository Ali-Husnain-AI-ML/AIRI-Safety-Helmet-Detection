from pathlib import Path
from collections import Counter

root = Path("cv_project/dataset/helmet_detection")
classes = {0: "Person", 1: "Helmet", 2: "No Helmet"}
errors = []
totals = Counter()

for split in ["train", "val", "test"]:
    images = list((root / "images" / split).glob("*"))
    labels = list((root / "labels" / split).glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing = image_stems - label_stems
    orphan = label_stems - image_stems

    for label in labels:
        for n, line in enumerate(label.read_text().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            try:
                if len(parts) != 5:
                    raise ValueError("Expected 5 values")
                cid = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                if cid not in classes:
                    raise ValueError(f"Unexpected class {cid}")
                if not all(0 <= x <= 1 for x in coords):
                    raise ValueError("Coordinates outside 0-1")
                totals[cid] += 1
            except Exception as e:
                errors.append(f"{label.name}, line {n}: {e}")

    print(f"\n{split.upper()}: images={len(images)}, labels={len(labels)}")
    print("Missing labels:", len(missing), "| Orphan labels:", len(orphan))

print("\nCLASS OBJECT TOTALS:")
for cid, name in classes.items():
    print(f"{name}: {totals[cid]}")

print("\nInvalid annotations:", len(errors))
for error in errors[:10]:
    print(error)

print("\nAUDIT:", "PASSED" if not errors else "NEEDS FIXING")
