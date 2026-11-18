"""Conservative temporal confirmation of visual observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

LABELS = frozenset({"Cry", "Happy", "Neutral", "Sleep"})


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    def valid(self) -> bool:
        return (
            self.label in LABELS
            and math.isfinite(self.confidence)
            and 0.0 <= self.confidence <= 1.0
            and len(self.xyxy) == 4
            and all(math.isfinite(v) for v in self.xyxy)
            and self.xyxy[2] > self.xyxy[0] >= 0.0
            and self.xyxy[3] > self.xyxy[1] >= 0.0
        )


@dataclass(frozen=True)
class Observation:
    state: str = "Unknown"
    reason: str = "no_frame"
    candidate: str | None = None
    confidence: float | None = None
    confirmed_frames: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def intersection_over_union(a: Detection, b: Detection) -> float:
    left = max(a.xyxy[0], b.xyxy[0])
    top = max(a.xyxy[1], b.xyxy[1])
    right = min(a.xyxy[2], b.xyxy[2])
    bottom = min(a.xyxy[3], b.xyxy[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (a.xyxy[2] - a.xyxy[0]) * (a.xyxy[3] - a.xyxy[1])
    area_b = (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1])
    return intersection / (area_a + area_b - intersection)


class TemporalState:
    def __init__(
        self,
        confidence: float = 0.60,
        confirm_ms: int = 1000,
        min_frames: int = 3,
        max_gap_ms: int = 1500,
        min_iou: float = 0.30,
    ) -> None:
        if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be in (0, 1]")
        if confirm_ms < 0 or min_frames < 2 or max_gap_ms <= 0:
            raise ValueError("invalid temporal confirmation settings")
        if not math.isfinite(min_iou) or not 0.0 < min_iou <= 1.0:
            raise ValueError("min_iou must be in (0, 1]")
        self.confidence = confidence
        self.confirm_ms = confirm_ms
        self.min_frames = min_frames
        self.max_gap_ms = max_gap_ms
        self.min_iou = min_iou
        self._previous: Detection | None = None
        self._first_ms = 0
        self._last_ms: int | None = None
        self._count = 0
        self._minimum_confidence = 1.0
        self.current = Observation()

    def invalidate(self, reason: str) -> Observation:
        self._previous = None
        self._last_ms = None
        self._count = 0
        self._minimum_confidence = 1.0
        self.current = Observation(reason=reason)
        return self.current

    def expire(self, now_ms: int) -> Observation:
        if self._last_ms is not None and now_ms - self._last_ms > self.max_gap_ms:
            return self.invalidate("stale")
        return self.current

    def update(self, detections: Iterable[Detection], now_ms: int) -> Observation:
        if type(now_ms) is not int or now_ms < 0:
            raise ValueError("timestamp must be a nonnegative integer")
        if self._last_ms is not None and now_ms <= self._last_ms:
            return self.invalidate("non_increasing_timestamp")
        if self._last_ms is not None and now_ms - self._last_ms > self.max_gap_ms:
            self.invalidate("stale")

        items = list(detections)
        if any(not item.valid() for item in items):
            return self.invalidate("invalid_detection")
        accepted = [item for item in items if item.confidence >= self.confidence]
        if not accepted:
            return self.invalidate("low_confidence" if items else "no_detection")
        if len(accepted) != 1:
            overlapping_classes = any(
                a.label != b.label and intersection_over_union(a, b) >= 0.50
                for index, a in enumerate(accepted) for b in accepted[index + 1:]
            )
            return self.invalidate("ambiguous_class" if overlapping_classes else "multiple_detections")

        item = accepted[0]
        continuous = (
            self._previous is not None
            and self._previous.label == item.label
            and intersection_over_union(self._previous, item) >= self.min_iou
        )
        if not continuous:
            self._first_ms = now_ms
            self._count = 0
            self._minimum_confidence = 1.0
        self._previous = item
        self._last_ms = now_ms
        self._count = min(self._count + 1, 1000000)
        self._minimum_confidence = min(self._minimum_confidence, item.confidence)
        confirmed = self._count >= self.min_frames and now_ms - self._first_ms >= self.confirm_ms
        self.current = Observation(
            state=item.label if confirmed else "Unknown",
            reason="confirmed" if confirmed else "confirming",
            candidate=item.label,
            confidence=round(self._minimum_confidence, 6),
            confirmed_frames=self._count,
        )
        return self.current
