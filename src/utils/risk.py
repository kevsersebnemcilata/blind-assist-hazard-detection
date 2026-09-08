from __future__ import annotations
from src.utils import config
from src.utils.config import PRIORITY_CLASSES

def compute_risk_score(
    class_id:       int,
    depth_norm:     float,
    conf:           float,
    motion_score:   float = 0.0,
    approach_score: float = 0.0,
    bbox_area_norm: float = 0.0,
) -> float:
    if class_id not in PRIORITY_CLASSES:
        return 0.0

    class_risk     = PRIORITY_CLASSES[class_id]["risk"] / 3.0
    depth_factor   = float(min(max(depth_norm, 0.0), 1.0))
    motion_score   = float(min(max(motion_score,   0.0), 1.0))
    approach_score = float(min(max(approach_score, 0.0), 1.0))
    bbox_area_norm = float(min(max(bbox_area_norm, 0.0), 1.0))
    conf           = float(min(max(conf,           0.0), 1.0))

    score = (
        class_risk     * config.RISK_W_CLASS    +
        depth_factor   * config.RISK_W_DEPTH    +
        motion_score   * config.RISK_W_MOTION   +
        approach_score * config.RISK_W_APPROACH +
        bbox_area_norm * config.RISK_W_BBOX     +
        conf           * config.RISK_W_CONF
    )
    return float(min(score, 1.0))