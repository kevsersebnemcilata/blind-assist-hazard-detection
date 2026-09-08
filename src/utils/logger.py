import csv
import os
import time


class CSVLogger:

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp     = time.strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"hazard_log_{timestamp}.csv")

        with open(self.log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_num",
                "motion_score",
                "depth_score",
                "delta_d",
                "approach_score",
                "danger_score",
                "trend",
                "timestamp",
            ])

    def log(self, frame_num, motion_score, depth_score, delta_d, approach_score, danger_score, trend=0):
        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_num,
                f"{motion_score:.4f}",
                f"{depth_score:.4f}",
                f"{delta_d:.4f}",
                f"{approach_score:.4f}",
                f"{danger_score:.4f}",
                trend,
                time.time(),
            ])

    def log_id_switches(self, frame_num, id_switch_count):
        print(f"[IDSwitch] frame={frame_num}  cumulative_switches={id_switch_count}")
        path = os.path.join(self.log_dir, "id_switches.csv")
        self._append_or_create(
            path,
            ["frame", "cumulative_switches"],
            [frame_num, id_switch_count],
        )

    def log_occlusion(self, frame_num, hidden_count):
        path = os.path.join(self.log_dir, "occlusion_log.csv")
        self._append_or_create(
            path,
            ["frame", "hidden_object_count"],
            [frame_num, hidden_count],
        )

    def log_risk_summary(self, frame_num, detections):
        path    = os.path.join(self.log_dir, "risk_summary.csv")
        headers = ["frame", "track_id", "class_name",
                   "risk_score", "depth_score", "motion_score"]

        high_risk = [d for d in detections if d.get("risk_score", 0.0) >= 0.45]
        if not high_risk:
            return

        file_exists = os.path.isfile(path)
        with open(path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            for det in high_risk:
                writer.writerow([
                    frame_num,
                    det.get("track_id", "?"),
                    det.get("class_name", "?"),
                    f"{det.get('risk_score',   0.0):.4f}",
                    f"{det.get('depth_score',  0.0):.4f}",
                    f"{det.get('motion_score', 0.0):.4f}",
                ])

    def _append_or_create(self, path, headers, row):
        file_exists = os.path.isfile(path)
        with open(path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)