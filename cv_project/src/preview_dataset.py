from pathlib import Path
import random
import cv2

ROOT = Path("cv_project/dataset/raw/extracted/construction-ppe")
OUTPUT = Path("cv_project/outputs/annotation_preview")

CLASSES = {
    0: "helmet",
    5: "person",
    6: "no_helmet"
}

# OpenCV uses BGR colour format
COLORS = {
    0: (0, 255, 255),   # Yellow - Helmet
    5: (255, 0, 0),     # Blue - Person
    6: (0, 0, 255)      # Red - No Helmet
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT.mkdir(parents=True, exist_ok=True)
random.seed(42)

selected = {}

for split in ["train", "val"]:

    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    for class_id, class_name in CLASSES.items():

        candidates = []

        for label_file in label_dir.glob("*.txt"):

            if not any(
                p.stem == label_file.stem
                and p.suffix.lower() in EXTENSIONS
                for p in image_dir.iterdir()
            ):
                continue

            lines = label_file.read_text(
                encoding="utf-8-sig"
            ).splitlines()

            if any(
                line.strip()
                and line.split()[0] == str(class_id)
                for line in lines
            ):
                candidates.append(label_file.stem)

        random.shuffle(candidates)

        for stem in candidates[:2]:
            selected[(split, stem)] = True

print("Preview images:", len(selected))

for index, (split, stem) in enumerate(selected.keys(), 1):

    image_dir = ROOT / "images" / split
    label_file = ROOT / "labels" / split / f"{stem}.txt"

    image_file = next(
        p for p in image_dir.iterdir()
        if p.stem == stem and p.suffix.lower() in EXTENSIONS
    )

    image = cv2.imread(str(image_file))

    if image is None:
        continue

    height, width = image.shape[:2]

    for line in label_file.read_text(
        encoding="utf-8-sig"
    ).splitlines():

        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))

        if class_id not in CLASSES:
            continue

        xc, yc, bw, bh = map(float, parts[1:])

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        color = COLORS[class_id]
        name = CLASSES[class_id]

        cv2.rectangle(
            image, (x1, y1), (x2, y2), color, 3
        )

        cv2.putText(
            image,
            name,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    output_file = OUTPUT / f"preview_{index:02d}.jpg"
    cv2.imwrite(str(output_file), image)

    print("Saved:", output_file)

print("\nCOLOUR-CODED PREVIEW COMPLETED!")
