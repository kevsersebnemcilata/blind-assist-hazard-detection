from __future__ import annotations
import queue
import shutil
import subprocess
import threading
import time

_TR_NAMES: dict[str, str] = {
    "car":           "araç",
    "motorcycle":    "motosiklet",
    "bus":           "otobüs",
    "train":         "tren",
    "truck":         "kamyon",
    "person":        "kişi",
    "bicycle":       "bisiklet",
    "dog":           "köpek",
    "traffic light": "trafik ışığı",
    "stop sign":     "dur işareti",
    "parking meter": "park saati",
    "bench":         "bank",
    "skateboard":    "kaykay",
}

_SPEED    = "180"
_FALLBACK_COOLDOWN = 5.0  # track_id yoksa bu kadar saniyede bir tekrar uyarır


def _direction(cx_norm: float) -> str:
    if cx_norm < 0.33:
        return "solda"
    if cx_norm < 0.67:
        return "önde"
    return "sağda"



def _build_message(critical_dets: list) -> str:
    """risk_score >= 0.75 olan tespitlerden kısa Türkçe uyarı üretir."""
    seen: dict[str, dict] = {}
    for det in critical_dets:
        name = _TR_NAMES.get(det.get("class_name", ""), "")
        if not name:
            continue
        risk = det.get("risk_score", 0.0)
        if name not in seen or risk > seen[name]["risk"]:
            seen[name] = {
                "risk":     risk,
                "cx_norm":  det.get("cx_norm", 0.5),
                "approach": det.get("approach_score", 0.0),
            }

    if not seen:
        return "Dikkat!"

    top = sorted(seen.items(), key=lambda kv: kv[1]["risk"], reverse=True)[:3]
    primary_name, primary_info = top[0]

    dir_str = _direction(primary_info["cx_norm"])
    verb    = " yaklaşıyor" if primary_info["approach"] > 0.15 else ""

    if len(top) == 1:
        return f"{dir_str} {primary_name}{verb}!"

    others     = [name for name, _ in top[1:]]
    others_str = " ve ".join(others) if len(others) == 1 else ", ".join(others[:-1]) + " ve " + others[-1]
    return f"{dir_str} {primary_name} ve {others_str}{verb}!"


class AudioAlertManager:
    def __init__(self):
        self._queue: queue.Queue[str] = queue.Queue()
        self._alerted_ids: set[int]   = set()   # her ID için yalnızca bir kez uyarı
        self._last_fallback: float    = 0.0      # track_id'siz nesneler için
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def update(self, danger_score: float, detections: list | None = None) -> None:
        dets = detections or []

        # Yalnızca risk_score >= 0.65 olan tespitlere bak
        new_critical: list[dict] = []
        for det in dets:
            if det.get("risk_score", 0.0) < 0.65:
                continue

            tid = det.get("track_id")
            if tid is not None:
                if tid in self._alerted_ids:
                    continue          # bu ID için zaten uyarıldı
                self._alerted_ids.add(tid)
            else:
                # ID yoksa zamana göre sınırla
                now = time.monotonic()
                if now - self._last_fallback < _FALLBACK_COOLDOWN:
                    continue
                self._last_fallback = now

            new_critical.append(det)

        if not new_critical:
            return

        # Tüm yeni kritik nesneleri tek mesajda birleştir
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        self._queue.put(_build_message(new_critical))

    def _worker(self) -> None:
        for exe in ("espeak-ng", "espeak"):
            if shutil.which(exe):
                print(f"[AudioAlert] {exe} bulundu, Türkçe TTS aktif.")
                self._run_espeak(exe)
                return

        print("[AudioAlert] espeak bulunamadı, pyttsx3 deneniyor.")
        self._run_pyttsx3()

    def _run_espeak(self, exe: str) -> None:
        while True:
            try:
                message = self._queue.get(timeout=1.0)
                subprocess.run(
                    [exe, "-v", "tr", "-s", _SPEED, message],
                    capture_output=True,
                )
            except queue.Empty:
                pass
            except Exception as exc:
                print(f"[AudioAlert] espeak hatası: {exc}")

    def _run_pyttsx3(self) -> None:
        try:
            import pyttsx3
        except ImportError:
            print("[AudioAlert] pyttsx3 bulunamadı. 'pip install pyttsx3' ile yükleyin.")
            return

        try:
            engine = pyttsx3.init()
        except Exception as exc:
            print(f"[AudioAlert] TTS başlatılamadı: {exc}")
            return

        engine.setProperty("volume", 1.0)
        engine.setProperty("rate", 180)

        for voice in engine.getProperty("voices"):
            if "tr" in voice.id.lower() or "turkish" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break

        while True:
            try:
                message = self._queue.get(timeout=1.0)
                engine.say(message)
                engine.runAndWait()
            except queue.Empty:
                pass
            except Exception as exc:
                print(f"[AudioAlert] Ses hatası: {exc}")
