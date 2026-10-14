"""Hugging Face Space entry point."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
weight_root = Path(__file__).resolve().parent / 'weights'
if os.environ.get('NOTESEP_PROFILE', 'paper-table1') == 'paper-table1':
    weight_root = weight_root / 'paper'
os.environ.setdefault("NOTESEP_WEIGHTS", str(weight_root))
from notesep.studio import launch

if __name__ == "__main__":
    launch(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
