from __future__ import annotations
import collections


class ApproachTracker:
    """
    Her track_id için bbox alan geçmişini tutar.
    Alan büyüyorsa nesne yaklaşıyor, küçülüyorsa uzaklaşıyor.
    """

    def __init__(self, history_len: int = 8, growth_threshold: float = 0.05):
        self._areas: dict[int, collections.deque] = {}
        self._history_len    = history_len
        self._growth_threshold = growth_threshold

    def update(self, detections: list) -> None:
        """Mevcut karedeki tespitlerle alan geçmişini güncelle."""
        seen: set[int] = set()
        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                continue
            seen.add(tid)
            x1, y1, x2, y2 = det["bbox"]
            area = float((x2 - x1) * (y2 - y1))
            if tid not in self._areas:
                self._areas[tid] = collections.deque(maxlen=self._history_len)
            self._areas[tid].append(area)

        # Artık görülmeyen nesneleri temizle
        for tid in list(self._areas):
            if tid not in seen:
                del self._areas[tid]

    def score(self, track_id: int) -> float:
        """
        0-1 arası yaklaşma skoru.
        Geçmişin ilk yarısı ile ikinci yarısının ortalamasını karşılaştırır.
        Alan büyümüşse (büyüme > eşik) → yaklaşıyor.
        """
        hist = self._areas.get(track_id)
        if not hist or len(hist) < 4:
            return 0.0

        vals = list(hist)
        mid      = len(vals) // 2
        recent   = sum(vals[mid:]) / (len(vals) - mid)
        previous = sum(vals[:mid]) / mid

        if previous <= 0:
            return 0.0

        growth = (recent - previous) / previous
        if growth < self._growth_threshold:
            return 0.0

        return float(min((growth - self._growth_threshold) / 0.50, 1.0))
