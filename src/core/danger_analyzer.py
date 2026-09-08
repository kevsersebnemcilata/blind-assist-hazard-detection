import collections
import numpy as np
from src.utils import config


class DangerAnalyzer:
    def __init__(self):
        self.danger_threshold      = config.DANGER_THRESHOLD
        self.motion_weight         = config.MOTION_WEIGHT
        self.depth_weight          = config.DEPTH_WEIGHT
        self.approach_weight       = config.APPROACH_WEIGHT
        self.near_region_threshold = config.NEAR_REGION_THRESHOLD
        self.depth_tau             = config.DEPTH_TAU
        self.smooth_window         = config.DANGER_SMOOTH_WINDOW
        self.trend_threshold       = config.TREND_THRESHOLD
        self.trend_boost           = config.TREND_BOOST

        self.prev_depth_score = None
        self._danger_history  = collections.deque(maxlen=self.smooth_window)

        self.DEPTH_FAR    = 0.70
        self.DEPTH_MEDIUM = 0.82

    def _depth_proximity_factor(self, depth_score):
        if depth_score < self.DEPTH_FAR:
            return (depth_score / self.DEPTH_FAR) * 0.30
        elif depth_score < self.DEPTH_MEDIUM:
            t = (depth_score - self.DEPTH_FAR) / (self.DEPTH_MEDIUM - self.DEPTH_FAR)
            return 0.30 + t * 0.35
        else:
            t = min((depth_score - self.DEPTH_MEDIUM) / (1.0 - self.DEPTH_MEDIUM), 1.0)
            return 0.65 + t * 0.35

    def _weighted_smooth(self):
        if not self._danger_history:
            return 0.0
        n       = len(self._danger_history)
        weights = [float(i + 1) for i in range(n)]
        values  = list(self._danger_history)
        total   = sum(w * v for w, v in zip(weights, values))
        return total / sum(weights)

    def _detect_trend(self):
        if len(self._danger_history) < 6:
            return 0
        vals     = list(self._danger_history)
        recent   = sum(vals[-3:]) / 3
        previous = sum(vals[-6:-3]) / 3
        delta    = recent - previous
        if delta >= self.trend_threshold:
            return 1
        if delta <= -self.trend_threshold:
            return -1
        return 0

    def analyze(self, motion_map, depth_map):
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        if d_max > d_min:
            norm_depth = (depth_map - d_min) / (d_max - d_min)
        else:
            norm_depth = depth_map * 0.0

        depth_score = float(sorted(norm_depth.flatten())[int(len(norm_depth.flatten()) * 0.90)])

        delta_d = 0.0
        if self.prev_depth_score is not None:
            delta_d = depth_score - self.prev_depth_score
        self.prev_depth_score = depth_score

        if delta_d > self.depth_tau:
            approach_score = min((delta_d - self.depth_tau) * 5.0, 1.0)
        else:
            approach_score = 0.0

        near_mask = norm_depth > self.near_region_threshold
        if near_mask.any():
            near_vals  = motion_map[near_mask].flatten()
            near_vals  = sorted(near_vals)
            avg_motion = float(near_vals[int(len(near_vals) * 0.95)])
        else:
            avg_motion = 0.0

        motion_score = min(avg_motion / 15.0, 1.0)

        proximity             = self._depth_proximity_factor(depth_score)
        object_is_near        = depth_score >= self.DEPTH_MEDIUM
        object_is_approaching = delta_d > self.depth_tau and motion_score > 0.1

        if object_is_near and object_is_approaching:
            raw_danger = (
                motion_score   * self.motion_weight  +
                proximity      * self.depth_weight   +
                approach_score * self.approach_weight
            )
        elif object_is_near:
            raw_danger = (
                motion_score * self.motion_weight +
                proximity    * self.depth_weight
            )
        elif proximity > 0.30:
            raw_danger = (
                motion_score * self.motion_weight +
                proximity    * (self.depth_weight * 0.5)
            )
        else:
            raw_danger = motion_score * self.motion_weight

        self._danger_history.append(raw_danger)
        smoothed = self._weighted_smooth()
        trend    = self._detect_trend()

        if trend == 1:
            smoothed = min(smoothed + self.trend_boost * 0.5, 1.0)
        elif trend == -1:
            smoothed = max(smoothed - self.trend_boost * 0.25, 0.0)

        danger_score = max(0.0, min(smoothed, 1.0))

        return motion_score, depth_score, delta_d, approach_score, danger_score, trend