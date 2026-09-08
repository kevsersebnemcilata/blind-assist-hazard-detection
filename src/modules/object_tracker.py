"""
object_tracker.py — YOLOv8 detection + ByteTrack persistent tracking + Re-ID.

Part 1: detect(frame)  → detections without track IDs  (track_id = None)
Part 2: track(frame)   → detections WITH persistent IDs from ByteTrack + ReIDBuffer

Bu kod, ekrandaki engelleri YOLOv8 ile kutu içine alır (tespit eder),
ByteTrack algoritması ile bu nesneleri kareler boyunca takip eder ve
**ReIDBuffer** sayesinde kameradan çıkıp giren nesnelere AYNI ID'yi yeniden atar.

Re-ID Algoritması:
  - Kaybolan her track'ın renk histogramı (96-boyutlu, EMA destekli) saklanır.
  - Yeni bir ByteTrack ID'si gelince kayıp track'larla kosinüs benzerliği hesaplanır.
  - Eşik aşılırsa → eski stable_id yeniden atanır (nesne "geri döndü").
  - Eşik aşılmazsa → yeni stable_id verilir (gerçekten yeni nesne).
"""

from __future__ import annotations

import math
import os
import numpy as np
import sys
from typing import TYPE_CHECKING

from src.utils import config  # PRIORITY_CLASSES + model paths

# ---------------------------------------------------------------------------
# Dependency guard — bağımlılık kontrolü
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO  # type: ignore[import]
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

if TYPE_CHECKING:
    import numpy as np


def _require_ultralytics() -> None:
    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "\n[ObjectTracker] 'ultralytics' is not installed.\n"
            "Fix:  pip install ultralytics\n"
            "      — or — pip install -r requirements.txt\n"
        )


# ---------------------------------------------------------------------------
# IDSwitchCounter
# ---------------------------------------------------------------------------

class IDSwitchCounter:
    """Counts how many times a detection center switches track_id between frames."""

    _DIST_THRESHOLD: float = 50.0  # px — centres closer than this are "same object"

    def __init__(self) -> None:
        """Initialise counter and previous-frame cache."""
        self._count: int = 0
        # Previous frame: list of (cx, cy, track_id)
        self._prev: list[tuple[float, float, int]] = []

    def update(self, detections: list[dict]) -> None:
        """Compare current detections against the previous frame and count ID switches."""
        current: list[tuple[float, float, int]] = []
        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                # ByteTrack disabled — nothing to count
                self._prev = []
                return
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            current.append((cx, cy, tid))

        # For each previous detection, look for the nearest current centre
        for px, py, p_tid in self._prev:
            best_dist = float("inf")
            best_tid: int | None = None
            for cx, cy, c_tid in current:
                d = math.hypot(cx - px, cy - py)
                if d < best_dist:
                    best_dist = d
                    best_tid = c_tid
            if best_tid is not None and best_dist < self._DIST_THRESHOLD and best_tid != p_tid:
                self._count += 1

        self._prev = current

    def get(self) -> int:
        """Return the cumulative ID-switch count since last reset."""
        return self._count

    def reset(self) -> None:
        """Reset the cumulative counter and clear the previous-frame cache."""
        self._count = 0
        self._prev = []


# ---------------------------------------------------------------------------
# ReIDBuffer — yeniden tanıma tamponu
# ---------------------------------------------------------------------------

class ReIDBuffer:
    """
    Kameradan çıkıp giren nesnelere aynı stable_id'yi yeniden atar.

    Nasıl çalışır:
      1. ByteTrack ID'si kaybolan her nesnenin renk histogramı + class_id
         saklanır (``_lost`` dict'i).
      2. Yeni bir ByteTrack ID'si geldiğinde kayıp track'larla
         kosinüs benzerliği karşılaştırılır (sadece aynı sınıf).
      3. Benzerlik ``MATCH_THRESHOLD`` üzerindeyse → eski ``stable_id``
         yeniden atanır; aksi hâlde yeni ``stable_id`` üretilir.
      4. Görünür nesnelerin özellikleri her karede EMA ile güncellenir;
         bu sayede aydınlatma değişimine karşı dayanıklılık artar.

    Özellik vektörü:
      - 3 renk kanalı (B, G, R) × 32 bin = 96 boyutlu normalize histogram.
      - Ek bağımlılık gerektirmez (sadece NumPy).

    Parametreler:
        max_lost_frames: Bir track'ın kaç kare bellekte tutulacağı.
                         Varsayılan 90 kare ≈ 30 fps'de 3 saniye.
        match_threshold: Yeniden atama için gereken minimum kosinüs benzerliği.
                         Daha yüksek → daha katı eşleme (yanlış pozitif az).
        ema_alpha:       Özellik güncellemesinde yeni kareye verilen ağırlık.
                         0.0 → özellik hiç güncellenmez; 1.0 → sadece son kare.
    """

    def __init__(
        self,
        max_lost_frames: int = 90,
        match_threshold: float = 0.75,
        ema_alpha: float = 0.25,
    ) -> None:
        self._max_lost_frames = max_lost_frames
        self._match_threshold = match_threshold
        self._ema_alpha = ema_alpha

        # stable_id → {feature, class_id, frames_lost}
        self._lost: dict[int, dict] = {}

        # ByteTrack tracker_id → stable_id (aktif eşleme)
        self._tracker_to_stable: dict[int, int] = {}

        # stable_id → özellik vektörü (son görünen karedeki)
        self._stable_features: dict[int, np.ndarray] = {}

        # stable_id → class_id (eşleme sonrası class kontrolü için)
        self._stable_class: dict[int, int] = {}

        # Monotonik artan stable ID üreteci
        self._next_id: int = 1

        # Bir önceki karede ByteTrack'ten gelen ID'ler
        self._prev_tracker_ids: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: "np.ndarray", detections: list[dict]) -> list[dict]:
        """
        Her ``track()`` çağrısından sonra çağrılır.

        1. Kaybolan ByteTrack ID'lerini ``_lost`` tamponuna taşır.
        2. Kayıp track'ları eskitir ve süre dolunca siler.
        3. Yeni ByteTrack ID'lerini kayıp track'larla eşleştirmeye çalışır.
        4. Deteksiyonlardaki ``track_id`` alanını ``stable_id`` ile değiştirir.

        Args:
            frame:       BGR kare (cv2 formatında).
            detections:  ``ObjectTracker.track()``'ten gelen ham deteksiyon listesi.

        Returns:
            ``track_id`` alanı ``stable_id`` ile güncellenmiş deteksiyon listesi.
            Geri kalan alanlar değişmez.
        """
        current_tracker_ids: set[int] = {
            d["track_id"] for d in detections if d["track_id"] is not None
        }

        # 1. Kaybolan tracker ID'lerini tampon'a taşı
        disappeared = self._prev_tracker_ids - current_tracker_ids
        for tid in disappeared:
            stable_id = self._tracker_to_stable.pop(tid, None)
            if stable_id is not None and stable_id not in self._lost:
                feature = self._stable_features.get(stable_id, np.zeros(96, dtype=np.float32))
                class_id = self._stable_class.get(stable_id, -1)
                self._lost[stable_id] = {
                    "feature":     feature,
                    "class_id":    class_id,
                    "frames_lost": 0,
                }

        # 2. Kayıp track'ları eskit; süresi dolanları sil
        expired = [
            sid for sid, info in self._lost.items()
            if info["frames_lost"] >= self._max_lost_frames
        ]
        for sid in expired:
            del self._lost[sid]
            self._stable_features.pop(sid, None)
            self._stable_class.pop(sid, None)

        for info in self._lost.values():
            info["frames_lost"] += 1

        # 3. Yeni tracker ID'lerini eşleştir (Re-ID)
        new_tracker_ids = current_tracker_ids - set(self._tracker_to_stable.keys())
        if new_tracker_ids and self._lost:
            # Yeni ID'ler için tek seferde özellik çıkar
            new_feats: dict[int, np.ndarray] = {}
            new_class: dict[int, int] = {}
            for det in detections:
                tid = det["track_id"]
                if tid in new_tracker_ids:
                    new_feats[tid] = self._extract_feature(frame, det["bbox"])
                    new_class[tid] = det["class_id"]

            # Her yeni ID için en iyi kayıp eşleşmesini bul
            # Tampon küçükse O(new × lost) yeterli; büyük sistemlerde
            # Hungarian algoritmasına geçilebilir.
            for tid in new_tracker_ids:
                feat = new_feats[tid]
                cls  = new_class[tid]
                best_sid:  int | None = None
                best_sim: float = self._match_threshold

                for stable_id, info in self._lost.items():
                    # Sınıf filtresi — yanlış eşleşmeyi önler
                    if info["class_id"] != cls:
                        continue
                    sim = float(np.dot(feat, info["feature"]))
                    if sim > best_sim:
                        best_sim = sim
                        best_sid = stable_id

                if best_sid is not None:
                    # ✅ Yeniden tanındı — eski ID geri atandı
                    self._tracker_to_stable[tid] = best_sid
                    del self._lost[best_sid]
                else:
                    # 🆕 Gerçekten yeni nesne
                    self._tracker_to_stable[tid] = self._next_id
                    self._next_id += 1
        else:
            # Kayıp tampon boş → tüm yeniler doğrudan yeni ID alır
            for det in detections:
                tid = det["track_id"]
                if tid is not None and tid not in self._tracker_to_stable:
                    self._tracker_to_stable[tid] = self._next_id
                    self._next_id += 1

        # 4. Görünür track'ların özelliklerini EMA ile güncelle
        #    ve track_id → stable_id ile değiştir
        updated: list[dict] = []
        for det in detections:
            tid = det["track_id"]
            if tid is None:
                updated.append(det)
                continue

            stable_id = self._tracker_to_stable.get(tid, tid)
            feat = self._extract_feature(frame, det["bbox"])

            if stable_id in self._stable_features:
                # EMA: mevcut özelliklere yeni gözlemi karıştır
                blended = (
                    (1.0 - self._ema_alpha) * self._stable_features[stable_id]
                    + self._ema_alpha * feat
                )
                norm = float(np.linalg.norm(blended))
                self._stable_features[stable_id] = blended / norm if norm > 0 else blended
            else:
                self._stable_features[stable_id] = feat

            self._stable_class[stable_id] = det["class_id"]

            updated.append({**det, "track_id": stable_id})

        self._prev_tracker_ids = current_tracker_ids
        return updated

    def reset(self) -> None:
        """Tüm durumu sıfırla (yeni video / senaryo başlangıcı)."""
        self._lost.clear()
        self._tracker_to_stable.clear()
        self._stable_features.clear()
        self._stable_class.clear()
        self._next_id = 1
        self._prev_tracker_ids = set()

    @property
    def lost_count(self) -> int:
        """Tamponda bekleyen kayıp track sayısı."""
        return len(self._lost)

    # ------------------------------------------------------------------
    # Dahili yardımcılar
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_feature(frame: "np.ndarray", bbox: list[int]) -> np.ndarray:
        """
        Bounding-box kesiminden normalize renk histogramı çıkarır.

        Çıktı: 96-boyutlu float32 birim vektör
               (B, G, R her kanal için 32 bin × 3 kanal).
        Ek bağımlılık yok — sadece NumPy.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(96, dtype=np.float32)

        hist_parts: list[np.ndarray] = []
        for ch in range(3):
            h_bins, _ = np.histogram(crop[:, :, ch], bins=32, range=(0, 256))
            hist_parts.append(h_bins.astype(np.float32))

        feature = np.concatenate(hist_parts)
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm > 0 else feature


# ---------------------------------------------------------------------------
# ObjectTracker
# ---------------------------------------------------------------------------

class ObjectTracker:
    """
    YOLOv8n-based detector / tracker — Re-ID destekli.

    * ``detect(frame)``  — Part 1: detection only, no IDs.
    * ``track(frame)``   — Part 2: YOLOv8 + ByteTrack + ReIDBuffer.

    ``track()`` artık kameradan çıkıp giren nesnelere aynı ``track_id``'yi
    yeniden atar; ByteTrack'in sıfırladığı ID'ler kararlı hâle gelir.

    Her iki yöntemin de döndürdüğü dict yapısı::

        {
            "track_id":    int | None,   # None for detect(), int for track()
            "class_id":    int,
            "class_name":  str,
            "bbox":        [x1, y1, x2, y2],   # pixel coords
            "confidence":  float,
            # Danger-integration placeholders (populated by caller):
            "depth_score": None,
            "motion_score": None,
        }

    Parametreler:
        model_path:         Yerel ``.pt`` ağırlık dosyası yolu.
        conf_threshold:     config.YOLO_CONF_THRESHOLD'u geçersiz kılar.
        reid_max_lost:      ReIDBuffer için max kayıp kare sayısı (varsayılan 90).
        reid_match_thresh:  Re-ID eşleme eşiği 0-1 arası (varsayılan 0.75).
        reid_ema_alpha:     Özellik güncelleme hızı 0-1 arası (varsayılan 0.25).
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float | None = None,
        reid_max_lost: int = 90,
        reid_match_thresh: float = 0.75,
        reid_ema_alpha: float = 0.25,
    ) -> None:
        """
        Args:
            model_path:        Path to a local ``.pt`` weights file.
                               Defaults to ``config.YOLO_MODEL_PATH``.
            conf_threshold:    Overrides config.YOLO_CONF_THRESHOLD when set.
            reid_max_lost:     Frames a lost track is kept in the Re-ID buffer
                               before being discarded.  90 ≈ 3 s at 30 fps.
            reid_match_thresh: Cosine-similarity threshold for re-identification.
                               Higher → stricter match (fewer false positives).
            reid_ema_alpha:    EMA weight for appearance feature updates.
                               0 → never update; 1 → use only the latest frame.

        Raises:
            ImportError:       If ``ultralytics`` is not installed.
            FileNotFoundError: If the weights file does not exist locally.
        """
        _require_ultralytics()

        resolved_path: str = model_path if model_path is not None else config.YOLO_MODEL_PATH

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"[ObjectTracker] YOLOv8 weights not found at '{resolved_path}'.\n"
                f"  Place 'yolov8s.pt' inside the '{config.MODEL_DIR}/' directory and retry.\n"
                f"  Download once with:\n"
                f"    python -c \"from ultralytics import YOLO; YOLO('yolov8s.pt')\""
            )

        self.conf_threshold: float = (
            conf_threshold if conf_threshold is not None else config.YOLO_CONF_THRESHOLD
        )
        self.model = YOLO(resolved_path)
        self._id_switch_counter = IDSwitchCounter()

        # Re-ID buffer — kameradan çıkıp giren nesneler için
        self._reid_buffer = ReIDBuffer(
            max_lost_frames=reid_max_lost,
            match_threshold=reid_match_thresh,
            ema_alpha=reid_ema_alpha,
        )

        print(
            f"[ObjectTracker] Model loaded from '{resolved_path}'  "
            f"conf={self.conf_threshold:.2f}  "
            f"re-id(max_lost={reid_max_lost}, thresh={reid_match_thresh:.2f})"
        )

    # ------------------------------------------------------------------
    # Part 1: detection only
    # ------------------------------------------------------------------

    def detect(self, frame: "np.ndarray") -> list[dict]:
        """
        YOLOv8 inference — no tracking, track_id is always None.

        Args:
            frame: BGR image (e.g. from cv2.VideoCapture.read()).

        Returns:
            List of detection dicts (see class docstring).
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            classes=list(config.PRIORITY_CLASSES.keys()),
            verbose=False,
        )
        return self._parse(results, with_ids=False)

    # ------------------------------------------------------------------
    # Part 2: detection + ByteTrack + Re-ID
    # ------------------------------------------------------------------

    def track(self, frame: "np.ndarray") -> list[dict]:
        """
        YOLOv8 + ByteTrack + ReIDBuffer.

        ByteTrack deteksiyonlarını ReIDBuffer üzerinden geçirir:
        kameradan çıkıp giren nesnelere aynı ``track_id`` yeniden atanır.

        Args:
            frame: BGR image.

        Returns:
            List of detection dicts with stable track_id filled.
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            tracker="bytetrack.yaml",
            persist=True,
            classes=list(config.PRIORITY_CLASSES.keys()),
            verbose=False,
        )
        # ByteTrack'ten ham deteksiyonlar
        raw_detections = self._parse(results, with_ids=True)

        # Re-ID: kayıp track'larla eşleştir ve stable_id ata
        stable_detections = self._reid_buffer.update(frame, raw_detections)

        # ID-switch sayacı stable ID'ler üzerinde çalışır
        self._id_switch_counter.update(stable_detections)

        return stable_detections

    # ------------------------------------------------------------------
    # ID-switch counter public API
    # ------------------------------------------------------------------

    def get_id_switch_count(self) -> int:
        """Return the cumulative number of ID switches detected since last reset."""
        return self._id_switch_counter.get()

    def reset_id_switch_count(self) -> None:
        """Reset the ID-switch counter and clear the previous-frame cache."""
        self._id_switch_counter.reset()

    # ------------------------------------------------------------------
    # Re-ID buffer public API
    # ------------------------------------------------------------------

    def reset_reid_buffer(self) -> None:
        """
        Re-ID tamponunu temizle ve tüm stable ID sayacını sıfırla.

        Yeni bir video veya senaryo başlatıldığında çağrılmalıdır;
        aksi hâlde önceki videonun nesneleri yanlışlıkla eşleşebilir.
        """
        self._reid_buffer.reset()

    @property
    def reid_lost_count(self) -> int:
        """Tamponda şu an bekleyen kayıp track sayısı (tanı / hata ayıklama için)."""
        return self._reid_buffer.lost_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse(self, results, *, with_ids: bool) -> list[dict]:
        """Convert ultralytics Results objects into plain dicts."""
        detections: list[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf     = float(box.conf[0])
                cls_id   = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, str(cls_id))

                track_id: int | None = None
                if with_ids and box.id is not None:
                    track_id = int(box.id[0])

                detections.append({
                    "track_id":    track_id,
                    "class_id":    cls_id,
                    "class_name":  cls_name,
                    "bbox":        [x1, y1, x2, y2],
                    "confidence":  float(f"{conf:.4f}"),
                    "depth_score":  None,
                    "motion_score": None,
                })
        return detections