import os
import sys
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def download_midas() -> None:
    _banner("MiDaS_small — downloading model")

    if os.path.exists(config.MIDAS_MODEL_PATH):
        print(f"[SKIP] Already cached: {config.MIDAS_MODEL_PATH}")
        return

    try:
        import torch
    except ImportError:
        print("[ERROR] torch is not installed.")
        sys.exit(1)

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    print("Downloading MiDaS_small from torch hub ...")

    midas = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small",
        force_reload=False
    )

    torch.save(midas.state_dict(), config.MIDAS_MODEL_PATH)

    print(f"[OK] Saved → {config.MIDAS_MODEL_PATH}")


if __name__ == "__main__":
    _banner("Model Download Script")

    download_midas()

    _banner("All models cached.")
    print(f"  {config.MIDAS_MODEL_PATH}\n")