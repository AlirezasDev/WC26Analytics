"""
Root backward compatibility entrypoint for WC26 Analytics.
Delegates directly to the high-performance engine in src/wc26_engine.py.
"""
import sys
from pathlib import Path

# Add src/ to module search path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from wc26_engine import main

if __name__ == "__main__":
    main()
