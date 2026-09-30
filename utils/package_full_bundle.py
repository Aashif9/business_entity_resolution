#!/usr/bin/env python3
"""
Full Project Bundle Packager for Business Entity Resolution
Creates a complete, self-contained standalone zip archive containing:
  - dataset/ (train and test TSV files)
  - code/ (source files, trained LightGBM models, learned JSON artifacts, requirements.txt)
  - Business_Entity_Resolution.ipynb (full runnable notebook)
  - output/ (final candidate_pairs.tsv, matching_results.tsv)
  - utils/ (submission packaging & validation tools)
  - Full documentation & research briefs (README.md, Documentation_template.md, etc.)

Excludes intermediate scratch files (work/ directory) and previous submission zips.
"""

import os
import sys
import time
import zipfile
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Package complete Business Entity Resolution project bundle")
    parser.add_argument(
        "--output-zip",
        type=str,
        default="business_entity_resolution_full_bundle.zip",
        help="Filename for the output zip archive"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    zip_path = root_dir / args.output_zip
    
    print("=" * 70)
    print(f"[*] Packaging Full Project Bundle: {zip_path.name}")
    print(f"[*] Root directory: {root_dir}")
    print("=" * 70)

    # 1. Essential required files verification
    required_dataset_files = [
        "dataset/train/train_source1.tsv",
        "dataset/train/train_source2.tsv",
        "dataset/train/train_source3.tsv",
        "dataset/train/train_ground_truth.tsv",
        "dataset/test/test_source1.tsv",
        "dataset/test/test_source2.tsv",
        "dataset/test/test_source3.tsv",
    ]
    
    required_artifacts = [
        "code/business_entity_resolution/artifacts/lgb_matcher.txt",
        "code/business_entity_resolution/artifacts/lgb_matcher_unseen.txt",
        "code/business_entity_resolution/artifacts/lgb_stage2.txt",
        "code/business_entity_resolution/artifacts/native_states.json",
        "code/business_entity_resolution/artifacts/noise_tokens.json",
        "code/business_entity_resolution/artifacts/policy.json",
        "code/business_entity_resolution/artifacts/rule.json",
        "code/business_entity_resolution/artifacts/translit_map.json",
    ]
    
    required_outputs = [
        "output/matching_results.tsv",
        "output/candidate_pairs.tsv",
    ]
    
    required_notebooks = [
        "code/business_entity_resolution/Business_Entity_Resolution.ipynb",
    ]
    
    for rel in required_dataset_files + required_artifacts + required_outputs + required_notebooks:
        p = root_dir / rel
        if not p.is_file():
            print(f"[!] ERROR: Missing required file: {rel}")
            sys.exit(1)
            
    print("[+] All required datasets, models/artifacts, outputs, and notebooks verified.")

    # 2. Collect all files to include
    # Folders to traverse: dataset, code, output, utils
    # Files at root to include: markdown docs, notebooks, matching_results_UPLOAD_rescue.tsv, .gitignore
    files_to_pack = []
    
    # Dataset files
    for p in (root_dir / "dataset").rglob("*"):
        if p.is_file() and not any(part.startswith((".", "__pycache__")) for part in p.parts):
            files_to_pack.append(p)
            
    # Output files
    for p in (root_dir / "output").rglob("*"):
        if p.is_file():
            files_to_pack.append(p)
            
    # Code directory
    for p in (root_dir / "code").rglob("*"):
        if p.is_file() and not any(part.startswith((".", "__pycache__")) for part in p.parts):
            files_to_pack.append(p)
            
    # Utils directory
    for p in (root_dir / "utils").rglob("*"):
        if p.is_file() and not any(part.startswith((".", "__pycache__")) for part in p.parts):
            files_to_pack.append(p)
            
    # Root level documents & notebooks
    root_whitelist = [
        "README.md",
        "Documentation_template.md",
        "CODEX_RESEARCH_BRIEF.md",
        "CurrentStatus.md",
        "Business_Entity_Resolution.ipynb",
        "AWS_EC2_run_pipeline.ipynb",
        "matching_results_UPLOAD_rescue.tsv",
        ".gitignore",
    ]
    for name in root_whitelist:
        p = root_dir / name
        if p.is_file():
            files_to_pack.append(p)

    # Sort files by relative path for deterministic packaging
    files_to_pack = sorted(set(files_to_pack), key=lambda p: str(p.relative_to(root_dir)))
    
    total_uncompressed_bytes = sum(p.stat().st_size for p in files_to_pack)
    print(f"[*] Found {len(files_to_pack)} files to archive ({total_uncompressed_bytes / (1024**3):.2f} GB uncompressed)")

    # 3. Create zip archive
    start_time = time.time()
    print(f"[*] Writing archive: {zip_path.name} ...")
    
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for idx, file_path in enumerate(files_to_pack, 1):
            arcname = file_path.relative_to(root_dir).as_posix()
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  [{idx:2d}/{len(files_to_pack)}] Archiving {arcname} ({size_mb:.1f} MB)...")
            zf.write(file_path, arcname=arcname)

    elapsed = time.time() - start_time
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    zip_size_gb = zip_size_mb / 1024
    
    print("-" * 70)
    print(f"[+] Zip creation completed in {elapsed:.1f}s")
    print(f"[+] Archive Size: {zip_size_mb:.2f} MB ({zip_size_gb:.2f} GB)")
    print(f"[+] Compression ratio: {total_uncompressed_bytes / zip_path.stat().st_size:.2f}x")

    # 4. Integrity check
    print("[*] Testing archive integrity (testzip)...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            print(f"[!] Corrupted file found in zip: {bad_file}")
            sys.exit(1)
            
    print("[+] Zip integrity verified successfully: 0 errors detected.")
    print("=" * 70)

if __name__ == "__main__":
    main()
