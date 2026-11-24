from pathlib import Path
import csv
import random

SOURCE = Path("cv_project/dataset/candidate_images.csv")
OUTPUT = Path("cv_project/dataset/base_images_160.csv")

TARGET = 160
MINIMUM = {
    "person": 125,
    "helmet": 95,
    "no_helmet": 60
}

random.seed(42)

if not SOURCE.exists():
    raise FileNotFoundError(f"Missing file: {SOURCE}")

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) < TARGET:
    raise ValueError(f"Only {len(rows)} candidates available")

for row in rows:
    for cls in MINIMUM:
        row[cls] = int(row[cls])

remaining = rows.copy()
random.shuffle(remaining)

selected = []
coverage = {cls: 0 for cls in MINIMUM}

while len(selected) < TARGET:
    best = None
    best_score = -1

    for row in remaining:
        score = sum(
            (MINIMUM[cls] - coverage[cls]) / MINIMUM[cls]
            for cls in MINIMUM
            if row[cls] > 0 and coverage[cls] < MINIMUM[cls]
        )

        if score > best_score:
            best = row
            best_score = score

    if best is None or best_score <= 0:
        break

    selected.append(best)
    remaining.remove(best)

    for cls in MINIMUM:
        if best[cls] > 0:
            coverage[cls] += 1

# Fill remaining slots randomly.
random.shuffle(remaining)
selected.extend(remaining[:TARGET - len(selected)])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=selected[0].keys())
    writer.writeheader()
    writer.writerows(selected)

print("\n===== IMAGE SELECTION REPORT =====")
print("Selected images:", len(selected))

for cls in MINIMUM:
    count = sum(1 for row in selected if row[cls] > 0)
    print(f"{cls}: {count} images")

print("\nSaved:", OUTPUT)
print("Original dataset remains unchanged.")
