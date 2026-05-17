"""
Download Marqo/fashion200k from HuggingFace and build a training pairs CSV.

Positive pairs : cross-category outfit combinations
  (e.g. "Tops & Blouses" + "Pants & Jeans")
Negative pairs : same-category items (two tops, two dresses …)

Usage
-----
# Preview discovered categories without downloading images:
    venv/bin/python scripts/fetch_fashion200k.py --dry-run

# Full download (~3.5 GB) and pair generation:
    venv/bin/python scripts/fetch_fashion200k.py \
        --output-dir data/fashion200k \
        --pairs-per-category 500

# Then train:
    venv/bin/python scripts/train_polyvore.py \
        --pairs-csv data/fashion200k/pairs.csv \
        --epochs 5
"""

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------------------------
# Heuristic outfit-compatibility rules (keyword substring match on category1)
# Left keyword + right keyword = compatible outfit pair
# ---------------------------------------------------------------------------
OUTFIT_RULES = [
    ("top", "pant"),
    ("top", "skirt"),
    ("top", "short"),
    ("top", "jean"),
    ("blouse", "pant"),
    ("blouse", "skirt"),
    ("blouse", "short"),
    ("shirt", "pant"),
    ("shirt", "skirt"),
    ("shirt", "short"),
    ("shirt", "jean"),
    ("sweater", "pant"),
    ("sweater", "skirt"),
    ("sweater", "short"),
    ("cardigan", "pant"),
    ("cardigan", "skirt"),
    ("jacket", "pant"),
    ("jacket", "dress"),
    ("jacket", "skirt"),
    ("coat", "pant"),
    ("coat", "dress"),
    ("coat", "skirt"),
    ("dress", "shoe"),
    ("dress", "bag"),
    ("dress", "accessory"),
    ("top", "shoe"),
    ("pant", "shoe"),
    ("skirt", "shoe"),
    ("jean", "shoe"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Download Marqo/fashion200k and build outfit-compatibility pairs."
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("data/fashion200k"),
        help="Root output directory (images go in <output-dir>/images/).",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Max items to download (0 = all ~200 K). Use with --stride to sample evenly.",
    )
    p.add_argument(
        "--stride", type=int, default=1,
        help=(
            "Only process every Nth row. Use to sample across all categories without "
            "downloading the full dataset. E.g. --limit 5000 --stride 40 samples "
            "5000 rows spread evenly across all 200K rows."
        ),
    )
    p.add_argument(
        "--pairs-per-category", type=int, default=500,
        help="Max positive pairs per compatible category combo.",
    )
    p.add_argument(
        "--neg-ratio", type=float, default=1.0,
        help="Negative pairs per positive pair (default 1.0 = balanced).",
    )
    p.add_argument(
        "--outfit-combos", type=Path, default=None,
        help=(
            "Optional JSON file listing explicit compatible category pairs, e.g. "
            '[[\"Tops & Blouses\", \"Pants & Jeans\"], ...]'
            ". Overrides the built-in keyword heuristic."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Only discover and print categories; do not download images or write CSV.",
    )
    return p.parse_args()


def category_slug(name: str) -> str:
    """Filesystem-safe directory name from a category string."""
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in (name or "unknown")).strip()


def keywords_compatible(cat_a: str, cat_b: str) -> bool:
    a = cat_a.lower()
    b = cat_b.lower()
    for kw_a, kw_b in OUTFIT_RULES:
        if (kw_a in a and kw_b in b) or (kw_b in a and kw_a in b):
            return True
    return False


def find_compatible_pairs(categories: list[str], explicit_combos=None) -> list[tuple]:
    if explicit_combos is not None:
        # Keep only combos where both sides are present in discovered categories
        cat_set = set(categories)
        return [(a, b) for a, b in explicit_combos if a in cat_set and b in cat_set]

    pairs = []
    for cat_a, cat_b in itertools.combinations(categories, 2):
        if keywords_compatible(cat_a, cat_b):
            pairs.append((cat_a, cat_b))
    return pairs


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "The `datasets` package is required.\n"
            "  venv/bin/pip install datasets Pillow",
            file=sys.stderr,
        )
        sys.exit(1)

    explicit_combos = None
    if args.outfit_combos:
        with args.outfit_combos.open() as f:
            explicit_combos = [tuple(pair) for pair in json.load(f)]
        print(f"Loaded {len(explicit_combos)} explicit outfit combos from {args.outfit_combos}")

    images_dir = args.output_dir / "images"
    pairs_csv = args.output_dir / "pairs.csv"

    if not args.dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)

    print("Streaming Marqo/fashion200k from HuggingFace …")
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)

    items_by_cat: dict[str, list[Path]] = {}
    count = 0
    row_index = 0

    for row in ds:
        row_index += 1
        if args.stride > 1 and (row_index % args.stride) != 0:
            continue

        if args.limit and count >= args.limit:
            break

        cat = (row.get("category1") or "unknown").strip()
        item_id = (row.get("item_ID") or str(count)).strip()
        image = row.get("image")

        if image is None:
            continue

        if args.dry_run:
            items_by_cat.setdefault(cat, []).append(Path("(dry-run)"))
            count += 1
            if row_index % 10000 == 0:
                print(f"  scanned {row_index} rows, kept {count} …")
            continue

        slug = category_slug(cat)
        cat_dir = images_dir / slug
        cat_dir.mkdir(exist_ok=True)
        img_path = cat_dir / f"{item_id}.jpg"

        if not img_path.exists():
            try:
                image.save(img_path)
            except Exception as exc:
                print(f"  warning: could not save {item_id}: {exc}")
                continue

        items_by_cat.setdefault(cat, []).append(img_path)
        count += 1
        if count % 1000 == 0:
            print(f"  downloaded {count} items …")

    print(f"\nTotal items: {count}  |  Categories: {len(items_by_cat)}")
    print("\nDiscovered categories:")
    for cat in sorted(items_by_cat):
        print(f"  [{len(items_by_cat[cat]):>5}]  {cat}")

    compatible = find_compatible_pairs(list(items_by_cat), explicit_combos)

    if not compatible:
        print(
            "\nNo compatible category pairs matched the built-in rules.\n"
            "Check category names above and either:\n"
            "  • Pass --outfit-combos combos.json with explicit pairs, or\n"
            "  • Add keywords to OUTFIT_RULES in this script."
        )
    else:
        print(f"\nCompatible category pairs ({len(compatible)}):")
        for a, b in compatible:
            print(f"  {a!r}  +  {b!r}")

    if args.dry_run:
        return

    rng = random.Random(args.seed)

    # ---- Positive pairs ------------------------------------------------
    positive_pairs: list[tuple[Path, Path]] = []
    for cat_a, cat_b in compatible:
        pool_a = list(items_by_cat[cat_a])
        pool_b = list(items_by_cat[cat_b])
        rng.shuffle(pool_a)
        rng.shuffle(pool_b)
        n = min(len(pool_a), len(pool_b), args.pairs_per_category)
        for i in range(n):
            positive_pairs.append((pool_a[i], pool_b[i % len(pool_b)]))

    rng.shuffle(positive_pairs)

    # ---- Negative pairs (same-category) --------------------------------
    neg_target = max(1, int(len(positive_pairs) * args.neg_ratio))
    negative_pairs: list[tuple[Path, Path]] = []

    for cat, paths in items_by_cat.items():
        if len(paths) < 2:
            continue
        indices = list(range(len(paths)))
        rng.shuffle(indices)
        for i in range(0, len(indices) - 1, 2):
            negative_pairs.append((paths[indices[i]], paths[indices[i + 1]]))
            if len(negative_pairs) >= neg_target * 2:
                break
        if len(negative_pairs) >= neg_target * 2:
            break

    rng.shuffle(negative_pairs)
    negative_pairs = negative_pairs[:neg_target]

    # ---- Write CSV -----------------------------------------------------
    with pairs_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left_path", "right_path", "label"])
        for left, right in positive_pairs:
            writer.writerow([left, right, 1])
        for left, right in negative_pairs:
            writer.writerow([left, right, 0])

    print(
        f"\nWrote {len(positive_pairs)} positive + {len(negative_pairs)} negative pairs"
        f"  →  {pairs_csv}"
    )
    print(
        f"\nNext step:\n"
        f"  venv/bin/python scripts/train_polyvore.py \\\n"
        f"    --pairs-csv {pairs_csv} \\\n"
        f"    --epochs 5"
    )


if __name__ == "__main__":
    main()
