from __future__ import annotations
import collections
import os
import queue       # still used by _alert_q
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify

from src.utils import config
from src.utils.logger import CSVLogger
from src.utils.risk import compute_risk_score
from src.utils.approach_tracker import ApproachTracker
from src.modules.depth_estimator import DepthEstimator
from src.modules.optical_flow import OpticalFlowEstimator
from src.core.danger_analyzer import DangerAnalyzer
from src.utils.audio_alert import _build_message, _FALLBACK_COOLDOWN

app = Flask(__name__)

# ─── Shared state ─────────────────────────────────────────────────────────────
# Multi-client broadcast: all connected /video_feed clients share the same frame
_frame_cond  = threading.Condition()   # notifies all waiting clients
_last_frame: bytes | None = None       # most recent JPEG (overwritten each frame)

_alert_q: queue.Queue[str] = queue.Queue(maxsize=20)
_stop_evt   = threading.Event()
_state_lock = threading.Lock()
_state      = "idle"   # "idle" | "loading" | "running"
_thread: threading.Thread | None = None

# Live metrics updated every frame from the processing thread
_metrics: dict = {
    "level": "GUVENLI",
    "danger_score": 0.0,
    "fps": 0.0,
    "trend": 0,
    "frame_num": 0,
}


# ─── Alert manager (same logic as AudioAlertManager, pushes to SSE) ───────────
class _WebAlertManager:
    def __init__(self):
        self._alerted_ids: set[int] = set()
        self._last_fallback = 0.0

    def reset(self):
        self._alerted_ids.clear()
        self._last_fallback = 0.0

    def update(self, danger_score: float, detections: list | None = None) -> None:
        dets = detections or []
        new_critical: list[dict] = []
        for det in dets:
            if det.get("risk_score", 0.0) < 0.65:
                continue
            tid = det.get("track_id")
            if tid is not None:
                if tid in self._alerted_ids:
                    continue
                self._alerted_ids.add(tid)
            else:
                now = time.monotonic()
                if now - self._last_fallback < _FALLBACK_COOLDOWN:
                    continue
                self._last_fallback = now
            new_critical.append(det)
        if not new_critical:
            return
        msg = _build_message(new_critical)
        try:
            _alert_q.put_nowait(msg)
        except queue.Full:
            pass


# ─── Drawing helpers (copied from main.py) ────────────────────────────────────
def _draw_yolo(frame, detections, depth_map, motion_map=None):
    fh, fw = frame.shape[:2]
    frame_area = max(fh * fw, 1)
    d_min = d_max = None
    if depth_map is not None:
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())

    for det in detections:
        class_id = det["class_id"]
        cls_info = config.PRIORITY_CLASSES.get(class_id)
        color    = cls_info["color"] if cls_info else (180, 180, 180)
        x1, y1, x2, y2 = det["bbox"]
        tid  = det.get("track_id")
        conf = det["confidence"]

        depth_norm = 0.0
        if depth_map is not None:
            cx = int(max(0, min((x1 + x2) // 2, fw - 1)))
            cy = int(max(0, min((y1 + y2) // 2, fh - 1)))
            depth_val = float(depth_map[cy, cx])
            if d_max > d_min:
                depth_norm = float((depth_val - d_min) / (d_max - d_min))

        obj_motion = 0.0
        if motion_map is not None:
            my1 = max(0, y1); my2 = min(motion_map.shape[0], y2)
            mx1 = max(0, x1); mx2 = min(motion_map.shape[1], x2)
            if my2 > my1 and mx2 > mx1:
                region = motion_map[my1:my2, mx1:mx2]
                obj_motion = min(float(np.percentile(region, 95)) / 15.0, 1.0)

        bbox_area_norm = float(((x2 - x1) * (y2 - y1)) / frame_area)
        risk_score = compute_risk_score(
            class_id=class_id, depth_norm=depth_norm, conf=conf,
            motion_score=obj_motion,
            approach_score=det.get("approach_score", 0.0),
            bbox_area_norm=bbox_area_norm,
        )
        det["risk_score"] = risk_score
        det["cx_norm"]    = (x1 + x2) / 2 / fw
        det["depth_norm"] = depth_norm

        id_str = str(tid) if tid is not None else "?"
        label  = f"{det['class_name']} | ID:{id_str} | Risk:{risk_score:.2f}"
        thickness = 3 if tid is not None else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


# ─── Background processing loop ───────────────────────────────────────────────
def _processing_loop():
    global _state

    with _state_lock:
        _state = "loading"

    os.makedirs(config.LOG_DIR,    exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR,  exist_ok=True)

    logger           = CSVLogger(config.LOG_DIR)
    alert_mgr        = _WebAlertManager()
    approach_tracker = ApproachTracker()
    depth_estimator  = DepthEstimator()
    flow_estimator   = OpticalFlowEstimator()
    analyzer         = DangerAnalyzer()

    detector  = None
    use_track = False
    if config.ENABLE_YOLO:
        try:
            from src.modules.object_tracker import ObjectTracker
            detector  = ObjectTracker()
            use_track = config.ENABLE_BYTETRACK
            mode = "YOLO + ByteTrack" if use_track else "YOLO"
            print(f"[web] {mode} hazır.")
        except ImportError as exc:
            print(f"[web] YOLO yüklenemedi: {exc}")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[web] Kamera açılamadı (index={config.CAMERA_INDEX}).")
        with _state_lock:
            _state = "idle"
        return

    with _state_lock:
        _state = "running"

    _occ_window  = collections.deque(maxlen=10)
    frame_num    = 0
    depth_map    = None
    motion_map   = None
    fps_smoothed = 0.0
    alpha_fps    = 0.1
    danger_score = 0.0
    trend        = 0

    print("[web] İşleme döngüsü başladı.")
    try:
        while not _stop_evt.is_set():
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("[web] Kamera okuma hatası, duruluyor.")
                break

            if config.FRAME_SCALE != 1.0:
                frame = cv2.resize(frame, (0, 0),
                                   fx=config.FRAME_SCALE, fy=config.FRAME_SCALE,
                                   interpolation=cv2.INTER_LINEAR)

            frame_num += 1

            if frame_num % config.DEPTH_EVERY_N_FRAMES == 0:
                depth_map = depth_estimator.estimate(frame)

            if frame_num % config.FLOW_EVERY_N_FRAMES == 0:
                motion_map = flow_estimator.estimate(frame)

            motion_score = depth_score = delta_d = approach_score = 0.0
            if depth_map is not None and motion_map is not None:
                (motion_score, depth_score, delta_d,
                 approach_score, danger_score, trend) = analyzer.analyze(motion_map, depth_map)
                logger.log(frame_num, motion_score, depth_score,
                           delta_d, approach_score, danger_score, trend)
            else:
                danger_score = 0.0
                trend        = 0

            display_frame = frame.copy()
            detections    = []

            if detector is not None:
                detections = (detector.track(display_frame) if use_track
                              else detector.detect(display_frame))
                approach_tracker.update(detections)
                for det in detections:
                    det["depth_score"]  = depth_score
                    det["motion_score"] = motion_score
                    tid = det.get("track_id")
                    det["approach_score"] = (
                        approach_tracker.score(tid) if tid is not None
                        else approach_score
                    )
                _draw_yolo(display_frame, detections, depth_map, motion_map)

                if frame_num % 30 == 0:
                    logger.log_id_switches(frame_num, detector.get_id_switch_count())
                    if detections:
                        logger.log_risk_summary(frame_num, detections)

            alert_mgr.update(danger_score, detections)

            elapsed      = time.perf_counter() - t0
            instant_fps  = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_smoothed = alpha_fps * instant_fps + (1 - alpha_fps) * fps_smoothed

            # Update live metrics for the web UI
            _metrics["level"] = (
                "TEHLIKE" if danger_score >= config.TEHLIKE_THRESHOLD else
                "DIKKAT"  if danger_score >= config.DIKKAT_THRESHOLD  else
                "GUVENLI"
            )
            _metrics["danger_score"] = round(float(danger_score), 3)
            _metrics["fps"]          = round(float(fps_smoothed), 1)
            _metrics["trend"]        = int(trend)
            _metrics["frame_num"]    = frame_num

            _, buf = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg = buf.tobytes()

            # Broadcast frame to all connected clients
            global _last_frame
            with _frame_cond:
                _last_frame = jpg
                _frame_cond.notify_all()

    finally:
        cap.release()
        with _state_lock:
            _state = "idle"
        print("[web] İşleme döngüsü sonlandı.")


# ─── Flask routes ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start")
def start():
    global _thread, _state
    with _state_lock:
        if _state in ("running", "loading"):
            return jsonify(ok=True, state=_state)

    _stop_evt.clear()
    _thread = threading.Thread(target=_processing_loop, daemon=True)
    _thread.start()
    return jsonify(ok=True, state="loading")


@app.route("/stop")
def stop():
    global _state
    _stop_evt.set()

    # Wake all waiting video clients so they can exit
    with _frame_cond:
        _frame_cond.notify_all()

    while not _alert_q.empty():
        try:
            _alert_q.get_nowait()
        except queue.Empty:
            break

    with _state_lock:
        _state = "idle"
    return jsonify(ok=True, state="idle")


@app.route("/status")
def status():
    with _state_lock:
        return jsonify(state=_state)


@app.route("/metrics")
def get_metrics():
    return jsonify(**_metrics)


@app.route("/video_feed")
def video_feed():
    def generate():
        while not _stop_evt.is_set():
            with _frame_cond:
                notified = _frame_cond.wait(timeout=1.0)
            if not notified:
                continue
            jpg = _last_frame
            if jpg is None:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/alerts")
def alerts():
    def generate():
        while True:
            try:
                msg = _alert_q.get(timeout=30.0)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
