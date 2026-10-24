import os, shutil, tarfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPORT_DIR = ROOT / "report"
OUTPUT_TAR = ROOT / "arxiv_submission.tar.gz"

def create_arxiv_package():
    # Files to include
    include_files = [
        REPORT_DIR / "main.tex",
        REPORT_DIR / "main.bbl",
        REPORT_DIR / "refs.bib",
    ]
    
    # Include all files in sections/
    sections_dir = REPORT_DIR / "sections"
    for p in sections_dir.glob("*.tex"):
        include_files.append(p)
        
    # Include all figures
    figures_dir = REPORT_DIR / "figures"
    for p in figures_dir.glob("*.png"):
        include_files.append(p)
        
    # Check that required files exist
    for f in include_files:
        if not f.exists():
            print(f"[pack_arxiv] Warning: {f} does not exist!")

    print(f"[pack_arxiv] Packaging {len(include_files)} files into {OUTPUT_TAR.name}...")
    
    with tarfile.open(OUTPUT_TAR, "w:gz") as tar:
        for f in include_files:
            if f.exists():
                arcname = f.relative_to(REPORT_DIR).as_posix()
                tar.add(f, arcname=arcname)
                print(f"  + {arcname}")
                
    size_kb = OUTPUT_TAR.stat().st_size / 1024
    print(f"[pack_arxiv] SUCCESS! Archive created: {OUTPUT_TAR} ({size_kb:.1f} KB)")
    print("  Ready for direct upload to https://arxiv.org/submit")

if __name__ == "__main__":
    create_arxiv_package()
