
import json
import os
import time

def save_result(result: dict, out_path: str):
    """Save a dict as a JSON file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

class Timer:
    """Simple context manager timer."""
    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self._t0
