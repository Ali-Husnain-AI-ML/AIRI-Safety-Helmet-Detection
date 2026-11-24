from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "cv_project" / "dataset" / "helmet_detection"
PERSON_CLASS_ID = "0"

TILE_W = 480
TILE_H = 330
COLS = 3
ROWS = 4
PER_SHEET = COLS * ROWS


def find_image(images_dir: Path, stem: str):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = list(images_dir.glob(stem + ".*"))
    return matches[0] if matches else None


def read_person_boxes(label_path: Path):
    boxes = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return boxes

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5 or parts[0] != PERSON_CLASS_ID:
            continue
        try:
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        if not all(0 <= value <= 1 for value in (x, y, w, h)):
            continue
        boxes.append((x, y, w, h))
    return boxes


def get_font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def count_close_pairs(boxes):
    """Heuristic: boxes with vertical overlap and a small horizontal gap."""
    xyxy = []
    for cx, cy, bw, bh in boxes:
        xyxy.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))

    close = 0
    for i in range(len(xyxy)):
        ax1, ay1, ax2, ay2 = xyxy[i]
        ah = max(0.0, ay2 - ay1)
        aw = max(0.0, ax2 - ax1)
        for j in range(i + 1, len(xyxy)):
            bx1, by1, bx2, by2 = xyxy[j]
            bh = max(0.0, by2 - by1)
            bw = max(0.0, bx2 - bx1)
            vertical_overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
            vertical_ratio = vertical_overlap / max(0.001, min(ah, bh))
            horizontal_gap = max(0.0, max(ax1, bx1) - min(ax2, bx2))
            max_gap = max(0.025, 0.20 * min(aw, bw))
            if vertical_ratio >= 0.25 and horizontal_gap <= max_gap:
                close += 1
    return close


def make_sheet(split: str):
    images_dir = DATASET / "images" / split
    labels_dir = DATASET / "labels" / split

    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"[{split}] Missing images or labels folder:")
        print(f"  images: {images_dir}")
        print(f"  labels: {labels_dir}")
        return

    records = []
    label_files = sorted(labels_dir.glob("*.txt"))
    total_person_boxes = 0
    images_with_person = 0
    multi_person_images = 0
    three_plus_images = 0
    five_plus_images = 0

    for label_path in label_files:
        boxes = read_person_boxes(label_path)
        count = len(boxes)
        total_person_boxes += count
        if count > 0:
            images_with_person += 1
        if count >= 2:
            multi_person_images += 1
            image_path = find_image(images_dir, label_path.stem)
            if image_path is not None:
                close_pairs = count_close_pairs(boxes)
                records.append((image_path, boxes, close_pairs))
        if count >= 3:
            three_plus_images += 1
        if count >= 5:
            five_plus_images += 1

    print(f"\n========== {split.upper()} ========== ")
    print(f"Label files: {len(label_files)}")
    print(f"Person boxes: {total_person_boxes}")
    print(f"Images with Person: {images_with_person}")
    print(f"Images with 2+ Persons: {multi_person_images}")
    print(f"Images with 3+ Persons: {three_plus_images}")
    print(f"Images with 5+ Persons: {five_plus_images}")
    print(f"Multi-person images with matching image files: {len(records)}")

    if not records:
        print("No multi-person images found to preview.")
        return

    # Prioritize images whose annotated person boxes are close together.
    records.sort(key=lambda item: (item[2], len(item[1])), reverse=True)
    chosen = records[:PER_SHEET]
    sheet = Image.new("RGB", (COLS * TILE_W, ROWS * TILE_H), (18, 27, 42))
    draw_sheet = ImageDraw.Draw(sheet)
    title_font = get_font(18)
    label_font = get_font(15)

    for idx, (image_path, boxes, close_pairs) in enumerate(chosen):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"Could not open {image_path.name}: {exc}")
            continue

        draw = ImageDraw.Draw(image)
        iw, ih = image.size
        line_width = max(3, int(min(iw, ih) / 250))

        for person_idx, (cx, cy, bw, bh) in enumerate(boxes, start=1):
            x1 = max(0, int((cx - bw / 2) * iw))
            y1 = max(0, int((cy - bh / 2) * ih))
            x2 = min(iw - 1, int((cx + bw / 2) * iw))
            y2 = min(ih - 1, int((cy + bh / 2) * ih))
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle((x1, y1, x2, y2), outline=(50, 235, 130), width=line_width)
            tag = f"P{person_idx}"
            tag_box = draw.textbbox((x1, max(0, y1 - 22)), tag, font=label_font)
            draw.rectangle(tag_box, fill=(10, 100, 55))
            draw.text((x1, max(0, y1 - 22)), tag, fill=(255, 255, 255), font=label_font)

        thumb = ImageOps.contain(image, (TILE_W - 16, TILE_H - 58))
        tile = Image.new("RGB", (TILE_W, TILE_H), (24, 37, 56))
        x = (TILE_W - thumb.width) // 2
        y = 34 + (TILE_H - 48 - thumb.height) // 2
        tile.paste(thumb, (x, y))
        tile_draw = ImageDraw.Draw(tile)
        label = f"{image_path.name} | Persons: {len(boxes)} | close pairs: {close_pairs}"
        tile_draw.text((9, 8), label[:62], fill=(235, 243, 252), font=title_font)

        col = idx % COLS
        row = idx // COLS
        sheet.paste(tile, (col * TILE_W, row * TILE_H))

    out_path = BASE_DIR / f"person_{split}_audit.jpg"
    sheet.save(out_path, quality=92)
    print(f"Top annotated multi-person images (close-pair heuristic):")
    for image_path, boxes, close_pairs in chosen[:8]:
        print(f"  {image_path.name}: {len(boxes)} Person boxes, {close_pairs} close pair(s)")
    print(f"Contact sheet saved: {out_path}")


def main():
    if not DATASET.exists():
        print(f"Dataset folder not found: {DATASET}")
        print("Place this script in the AIRI_PITB_CV_Task1 project root.")
        sys.exit(1)

    print(f"Dataset: {DATASET}")
    make_sheet("train")
    make_sheet("val")
    make_sheet("test")
    print("\nDone. Open the generated person_train_audit.jpg first.")


if __name__ == "__main__":
    main()
