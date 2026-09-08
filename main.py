import collections
import os
import sys
import time

import cv2
import numpy as np
from src.utils import config
from src.utils.logger import CSVLogger
from src.utils.overlay import draw_hud
from src.utils.risk import compute_risk_score
from src.utils.audio_alert import AudioAlertManager
from src.utils.approach_tracker import ApproachTracker
from src.modules.depth_estimator import DepthEstimator
from src.modules.optical_flow import OpticalFlowEstimator
from src.core.danger_analyzer import DangerAnalyzer

def _draw_yolo(frame, detections, depth_map, motion_map=None):
    fh, fw = frame.shape[:2]
    frame_area = max(fh * fw, 1)

    # Sahne genelinde min/max bir kez hesapla
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

        depth_val  = 0.0
        depth_norm = 0.0
        if depth_map is not None:
            cx = int(max(0, min((x1 + x2) // 2, fw - 1)))
            cy = int(max(0, min((y1 + y2) // 2, fh - 1)))
            depth_val = float(depth_map[cy, cx])
            if d_max > d_min:
                depth_norm = float((depth_val - d_min) / (d_max - d_min))

        # Nesne bazlı hareket: bbox bölgesindeki optical flow büyüklüğü
        obj_motion = 0.0
        if motion_map is not None:
            my1 = max(0, y1);  my2 = min(motion_map.shape[0], y2)
            mx1 = max(0, x1);  mx2 = min(motion_map.shape[1], x2)
            if my2 > my1 and mx2 > mx1:
                region = motion_map[my1:my2, mx1:mx2]
                obj_motion = min(float(np.percentile(region, 95)) / 15.0, 1.0)

        bbox_area_norm = float(((x2 - x1) * (y2 - y1)) / frame_area)

        risk_score = compute_risk_score(
            class_id       = class_id,
            depth_norm     = depth_norm,
            conf           = conf,
            motion_score   = obj_motion,
            approach_score = det.get("approach_score", 0.0),
            bbox_area_norm = bbox_area_norm,
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

        if risk_score >= 0.75:
            banner       = "! CRITICAL"
            banner_color = (0, 0, 220)
        elif risk_score >= 0.45:
            banner       = "! WARNING"
            banner_color = (0, 140, 255)
        else:
            banner = banner_color = None

        if banner is not None:
            (bw, bh), bbl = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bx = x1
            by = y1 - th - bl - 4 - 6
            cv2.rectangle(frame, (bx, by - bh - bbl - 2), (bx + bw + 4, by), banner_color, -1)
            cv2.putText(frame, banner, (bx + 2, by - bbl - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def _draw_risk_legend(frame):
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.52
    THICKNESS  = 1
    PAD        = 8
    ROW_H      = 22
    ALPHA      = 0.55

    entries = [
        ("CRITICAL  risk >= 0.75", (0,   0, 200)),
        ("WARNING   risk >= 0.45", (0, 140, 255)),
        ("SAFE      risk <  0.45", (0, 200,  80)),
    ]

    max_w = max(
        cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)[0][0]
        for text, _ in entries
    )

    panel_w = max_w + PAD * 2 + 14
    panel_h = len(entries) * ROW_H + PAD * 2

    margin = 10
    fh, fw = frame.shape[:2]
    x0 = fw - panel_w - margin
    y0 = margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (200, 200, 200), 1)
    cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)

    for i, (text, color) in enumerate(entries):
        row_y = y0 + PAD + i * ROW_H
        cv2.rectangle(frame,
                      (x0 + PAD,      row_y),
                      (x0 + PAD + 10, row_y + 12),
                      color, -1)
        cv2.putText(frame, text,
                    (x0 + PAD + 14, row_y + 11),
                    FONT, FONT_SCALE, (230, 230, 230), THICKNESS, cv2.LINE_AA)

def main():
    os.makedirs(config.LOG_DIR,    exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR,  exist_ok=True)

    print("Initializing components ...")
    logger           = CSVLogger(config.LOG_DIR)
    audio_alert      = AudioAlertManager()
    approach_tracker = ApproachTracker()
    depth_estimator  = DepthEstimator()
    flow_estimator  = OpticalFlowEstimator()
    analyzer        = DangerAnalyzer()

    detector  = None
    use_track = False
    if config.ENABLE_YOLO:
        try:
            from src.modules.object_tracker import ObjectTracker
            detector  = ObjectTracker()
            use_track = config.ENABLE_BYTETRACK
            mode_str  = "YOLO + ByteTrack" if use_track else "YOLO (detect-only)"
            print(f"[main] {mode_str} ready.")
        except ImportError as exc:
            print(f"[main] WARNING: {exc}\n       Continuing without YOLO.")

    _occ_window = collections.deque(maxlen=10)

    print(f"Opening camera {config.CAMERA_INDEX} ...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {config.CAMERA_INDEX}.")
        sys.exit(1)

    writer = None
    if config.ENABLE_YOLO:
        fw_cam  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  * config.FRAME_SCALE)
        fh_cam  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * config.FRAME_SCALE)
        fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = (
            config.OUTPUT_TRACKED_VIDEO_PATH if use_track
            else config.OUTPUT_VIDEO_PATH
        )
        writer = cv2.VideoWriter(out_path, fourcc, fps_cam, (fw_cam, fh_cam))
        print(f"[main] Saving annotated output -> {out_path}")

    frame_num    = 0
    fps_smoothed = 0.0
    alpha_fps    = 0.1
    depth_map    = None
    motion_map   = None

    motion_score = depth_score = delta_d = approach_score = danger_score = 0.0
    trend        = 0

    print("Application started. Press 'q' to quit.\n")

    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed - exiting.")
                break

            if config.FRAME_SCALE != 1.0:
                frame = cv2.resize(
                    frame, (0, 0),
                    fx=config.FRAME_SCALE,
                    fy=config.FRAME_SCALE,
                    interpolation=cv2.INTER_LINEAR,
                )

            frame_num += 1

            if frame_num % config.DEPTH_EVERY_N_FRAMES == 0:
                depth_map = depth_estimator.estimate(frame)

            if frame_num % config.FLOW_EVERY_N_FRAMES == 0:
                motion_map = flow_estimator.estimate(frame)

            if depth_map is not None and motion_map is not None:
                (
                    motion_score,
                    depth_score,
                    delta_d,
                    approach_score,
                    danger_score,
                    trend,
                ) = analyzer.analyze(motion_map, depth_map)

                logger.log(
                    frame_num, motion_score, depth_score,
                    delta_d, approach_score, danger_score,
                    trend,
                )
            else:
                motion_score = depth_score = delta_d = approach_score = danger_score = 0.0
                trend = 0

            display_frame = frame.copy()
            detections    = []

            if detector is not None:
                detections = (
                    detector.track(display_frame) if use_track
                    else detector.detect(display_frame)
                )

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
                _draw_risk_legend(display_frame)

            if frame_num % 30 == 0 and detector is not None:
                logger.log_id_switches(frame_num, detector.get_id_switch_count())
                if detections:
                    logger.log_risk_summary(frame_num, detections)

            if config.OCCLUSION_TEST_MODE and use_track and detector is not None:
                visible_ids = {
                    det["track_id"] for det in detections
                    if det.get("track_id") is not None
                }
                _occ_window.append(visible_ids)
                expected_ids = set().union(*_occ_window)
                hidden_count = len(expected_ids - visible_ids)
                logger.log_occlusion(frame_num, hidden_count)

                _yel = (0, 255, 255)
                cv2.putText(display_frame, "OCCLUSION TEST ACTIVE",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _yel, 2, cv2.LINE_AA)
                if hidden_count > 0:
                    cv2.putText(display_frame, f"Hidden objects: {hidden_count}",
                                (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _yel, 2, cv2.LINE_AA)

            audio_alert.update(danger_score, detections)

            display_frame = draw_hud(
                display_frame,
                danger_score,
                config.DANGER_THRESHOLD,
                frame_num,
                trend,
            )

            elapsed      = time.perf_counter() - t0
            instant_fps  = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_smoothed = alpha_fps * instant_fps + (1 - alpha_fps) * fps_smoothed
            fps_color    = (0, 220, 0) if fps_smoothed >= 15 else (0, 140, 255)
            cv2.putText(display_frame, f"FPS {fps_smoothed:5.1f}",
                        (display_frame.shape[1] - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)

            cv2.imshow("Hazard Detection", display_frame)
            if writer is not None:
                writer.write(display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nShutdown complete. {frame_num} frames processed.")
        if writer is not None:
            out_path = (
                config.OUTPUT_TRACKED_VIDEO_PATH if use_track
                else config.OUTPUT_VIDEO_PATH
            )
            print(f"Output saved -> {out_path}")

if __name__ == "__main__":
    main()