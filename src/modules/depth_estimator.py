"""
depth_estimator.py — MiDaS depth estimation with local model caching.

- Mimari + Transform: torch.hub (kendi cache'ini kullanır, internet gerekmez)
- Weights sadece: models/midas_small.pt  (state_dict — saf tensörler)
"""

import os
import warnings

import cv2
import torch
import numpy as np

from src.utils import config

class DepthEstimator:
    def __init__(self) -> None:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DepthEstimator] Using device: {self.device}")

        os.makedirs(config.MODEL_DIR, exist_ok=True)

        # --- Mimari + Transform: her zaman torch.hub'dan ---
        # (hub kendi ~/.cache/torch/hub/ dizinini kullanır — internet gerekmez)
        print("[DepthEstimator] Loading MiDaS_small via torch.hub …")
        self.midas = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", force_reload=False,
        )
        self.transform = torch.hub.load(
            "intel-isl/MiDaS", "transforms", force_reload=False,
        ).small_transform

        # --- Weights: diskten yükle veya ilk çalışmada kaydet ---
        if os.path.exists(config.MIDAS_MODEL_PATH):
            print(f"[DepthEstimator] Loading weights from cache: {config.MIDAS_MODEL_PATH}")
            state_dict = torch.load(
                config.MIDAS_MODEL_PATH,
                map_location=self.device,
                weights_only=True,   # sadece tensörler — güvenli ✓
            )
            self.midas.load_state_dict(state_dict)
        else:
            print(f"[DepthEstimator] Saving weights → {config.MIDAS_MODEL_PATH}")
            torch.save(self.midas.state_dict(), config.MIDAS_MODEL_PATH)

        self.midas.to(self.device)
        self.midas.eval()
        print("[DepthEstimator] Ready.")

    # ------------------------------------------------------------------

    def estimate(self, frame) -> "np.ndarray":
        """Return a relative depth map (H W float32) for *frame* (BGR)."""
        import numpy as np

        img         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy()