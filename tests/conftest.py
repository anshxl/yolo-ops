import sys
import os

# add the parent directory (your repo root) to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Added {repo_root} to sys.path")