#!/usr/bin/env python3
"""
Submission Packager for Business Entity Resolution Challenge
Builds the final submission zip with the exact structure required by the organizers:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import zipfile
import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Package submission into final zip archive")
    parser.add_argument("--team-name", type=str, required=True, help="Your team name for the zip filename")
    parser.add_argument("--skip-validation", action="store_true", help="Skip running validate_submission.py")
    return parser.parse_args()

def main():
    args = parse_args()
    team_name = args.team_name.strip().replace(" ", "_")
    
    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "output"
    code_dir = root_dir / "code" / "business_entity_resolution"
    doc_file = root_dir / "Documentation_template.md"
    zip_path = root_dir / f"{team_name}_submission.zip"
    
    matching_tsv = output_dir / "matching_results.tsv"
    candidate_tsv = output_dir / "candidate_pairs.tsv"
    
    print(f"[*] Packaging submission for team: {team_name}...")
    
    # 1. Check required output files
    if not matching_tsv.is_file():
        print(f"[!] Error: {matching_tsv} not found! Please run the pipeline first.")
        sys.exit(1)
        
    if not candidate_tsv.is_file():
        print(f"[!] Error: {candidate_tsv} not found! Please run candidate generation first.")
        sys.exit(1)
        
    # 2. Run validator unless skipped
    if not args.skip_validation:
        print("[*] Running submission validator...")
        validator_script = root_dir / "utils" / "validate_submission.py"
        test_dir = root_dir / "dataset" / "test"
        cmd = [
            sys.executable,
            str(validator_script),
            "--matching", str(matching_tsv),
            "--candidate", str(candidate_tsv),
            "--test-dir", str(test_dir)
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print("[!] Validation failed! Please fix the errors before packaging.")
            sys.exit(1)
            
    # 3. Create zip package
    print(f"[*] Creating archive: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add output files
        zf.write(matching_tsv, arcname=f"output/{matching_tsv.name}")
        zf.write(candidate_tsv, arcname=f"output/{candidate_tsv.name}")
        
        # Add code/business_entity_resolution
        for file_path in code_dir.rglob("*"):
            if file_path.is_file() and not any(part.startswith((".", "__pycache__")) for part in file_path.parts):
                rel_path = file_path.relative_to(root_dir / "code")
                zf.write(file_path, arcname=f"code/{rel_path.as_posix()}")
                
        # Add Documentation_template.md
        if doc_file.is_file():
            zf.write(doc_file, arcname="Documentation_template.md")
        else:
            print("[!] Warning: Documentation_template.md not found, skipping in zip.")

    file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[+] Successfully generated: {zip_path.name} ({file_size_mb:.2f} MB)")
    print("[+] Done.")

if __name__ == "__main__":
    main()
