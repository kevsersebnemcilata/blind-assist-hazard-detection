import cv2
from src.utils import config


def draw_hud(frame, danger_score, danger_threshold, frame_count=0, trend=0):

    if danger_score >= config.TEHLIKE_THRESHOLD:
        text  = "TEHLIKE"
        color = (0, 0, 255)
        level = "TEHLIKE"
    elif danger_score >= config.DIKKAT_THRESHOLD:
        text  = "DIKKAT"
        color = (0, 165, 255)
        level = "DIKKAT"
    else:
        text  = "GUVENLI"
        color = (0, 220, 0)
        level = "GUVENLI"

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]

    # Durum metni
    cv2.putText(frame, text, (52, 52), font, 1.5, (0, 0, 0), 5)
    cv2.putText(frame, text, (50, 50), font, 1.5, color, 3)

    # Skor metni
    metrics = f"Danger: {danger_score:.2f} / {danger_threshold:.2f}"
    cv2.putText(frame, metrics, (50, 90), font, 0.7, (255, 255, 255), 2)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 50, 100, 200, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (160, 160, 160), 1)
    fill = int(bar_w * min(max(danger_score, 0.0), 1.0))
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)

    # Trend rozeti
    if trend == 1:
        trend_text  = "^ ARTIYOR"
        trend_color = (0, 60, 220)
    elif trend == -1:
        trend_text  = "v AZALIYOR"
        trend_color = (0, 200, 60)
    else:
        trend_text  = "* STABIL"
        trend_color = (0, 210, 210)
    cv2.putText(frame, trend_text, (50, 128), font, 0.6, trend_color, 2)

    # Kenarlık
    if level == "TEHLIKE":
        if (frame_count // 15) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
    elif level == "DIKKAT":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 165, 255), 6)

    return frame