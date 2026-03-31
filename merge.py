"""
merge_classes.py
────────────────
รวม folder ที่เป็นผลไม้/ผักชนิดเดียวกัน (ต่างสายพันธุ์) ให้เป็น class เดียว
แล้ว copy ไปยัง dataset_merged/Fruit/<merged_label>/

โครงสร้างเดิม:  dataset/Fruit/<original_label>/
โครงสร้างใหม่:  dataset_merged/Fruit/<merged_label>/

Usage:
    python merge_classes.py                 # dry-run (แสดงผลแต่ไม่ copy)
    python merge_classes.py --execute       # copy จริง
    python merge_classes.py --execute --src dataset/Fruit --dst dataset_merged/Fruit
"""

import argparse
import shutil
import re
from pathlib import Path
from collections import defaultdict

# ────────────────────────────────────────────────────────────────────────────
# MERGE MAP
# key   = ชื่อ class ใหม่ (merged)
# value = list of keywords (case-insensitive) ที่ตรงกับชื่อ folder เดิม
# ────────────────────────────────────────────────────────────────────────────
MERGE_MAP = {
    "Apple":       ["apple"],
    "Avocado":     ["avocado"],
    "Banana":      ["banana"],
    "Blackberry":  ["blackberry"],
    "Cabbage":     ["cabbage"],
    "Cactus Fruit":["cactus fruit"],
    "Cantaloupe":  ["cantaloupe"],
    "Carrot":      ["carrot"],
    "Cherry":      ["cherry"],
    "Cucumber":    ["cucumber"],
    "Dates":       ["dates"],
    "Eggplant":    ["eggplant"],
    "Ginger":      ["ginger"],
    "Gooseberry":  ["gooseberry"],
    "Grape":       ["grape"],
    "Nectarine":   ["nectarine"],
    "Nut":         ["nut", "almond", "pistachio", "peanut"],
    "Onion":       ["onion"],
    "Orange":      ["orange"],
    "Papaya":      ["papaya"],
    "Peach":       ["peach"],
    "Pear":        ["pear"],
    "Pepper":      ["pepper"],
    "Plum":        ["plum"],
    "Quince":      ["quince"],
    "Raspberry":   ["raspberry"],
    "Strawberry":  ["strawberry"],
    "Tomato":      ["tomato"],
    "Zucchini":    ["zucchini"],
    # เพิ่ม class อื่นๆ ที่ต้องการ merge ได้ที่นี่
}


def get_merged_label(folder_name: str) -> str:
    """Return merged label for folder_name, or folder_name itself if no match."""
    name_lower = folder_name.lower()
    for merged_label, keywords in MERGE_MAP.items():
        for kw in keywords:
            if kw in name_lower:
                return merged_label
    return folder_name   # ไม่มีใน map → ใช้ชื่อเดิม


def build_plan(src: Path) -> dict[str, list[Path]]:
    """Return {merged_label: [src_folder, ...]} mapping."""
    plan = defaultdict(list)
    for folder in sorted(src.iterdir()):
        if folder.is_dir():
            label = get_merged_label(folder.name)
            plan[label].append(folder)
    return plan


def print_plan(plan: dict):
    print("\n" + "="*60)
    print(f"  Merge Plan  ({len(plan)} classes after merge)")
    print("="*60)
    for merged, folders in sorted(plan.items()):
        src_names = [f.name for f in folders]
        if len(src_names) == 1 and src_names[0] == merged:
            print(f"  {merged:20s}  (no change)")
        else:
            print(f"  {merged:20s}  ← {src_names}")
    print("="*60 + "\n")


def count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in exts)


def execute_merge(plan: dict, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    total_copied = 0

    for merged_label, src_folders in sorted(plan.items()):
        dst_label = dst / merged_label
        dst_label.mkdir(parents=True, exist_ok=True)

        img_idx = 0
        for src_folder in src_folders:
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            images = [f for f in src_folder.rglob("*") if f.suffix.lower() in exts]
            for img_path in sorted(images):
                # ตั้งชื่อไฟล์ใหม่เพื่อหลีกเลี่ยงชนกัน
                new_name = f"{img_idx:05d}{img_path.suffix.lower()}"
                shutil.copy2(img_path, dst_label / new_name)
                img_idx += 1
            print(f"  [{merged_label}] copied {len(images):4d} imgs from '{src_folder.name}'")
            total_copied += len(images)

    print(f"\n[DONE] Total images copied: {total_copied:,}")
    print(f"[DONE] Output → {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="dataset/Fruit",        help="Source dataset path")
    parser.add_argument("--dst", default="dataset_merged/Fruit", help="Destination path")
    parser.add_argument("--execute", action="store_true",        help="Actually copy files")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    plan = build_plan(src)
    print_plan(plan)

    # แสดงสถิติ
    total_classes_before = sum(len(v) for v in plan.values())
    total_classes_after  = len(plan)
    print(f"Classes before merge : {total_classes_before}")
    print(f"Classes after  merge : {total_classes_after}")
    print(f"Reduction            : -{total_classes_before - total_classes_after} classes\n")

    if args.execute:
        print(f"Copying to: {dst}\n")
        execute_merge(plan, dst)
    else:
        print("[ DRY RUN ] — เพิ่ม --execute เพื่อ copy จริง\n")
        print("  python merge_classes.py --execute\n")


if __name__ == "__main__":
    main()