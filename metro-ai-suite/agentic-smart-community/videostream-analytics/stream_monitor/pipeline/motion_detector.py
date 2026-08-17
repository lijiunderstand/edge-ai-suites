"""Frame-diff based motion detector."""

from __future__ import annotations

import cv2
import numpy as np

from shared.config import MotionConfig


class MotionDetector:
    def __init__(self, config: MotionConfig):
        self.diff_threshold = config.diff_threshold
        self.area_ratio = config.area_ratio
        self.stable_frames = config.stable_frames
        self._prev_gray: np.ndarray | None = None
        self._static_count = 0

    @property
    def is_static(self) -> bool:
        return self._static_count >= self.stable_frames

    def detect(self, frame: np.ndarray) -> bool:
        """Process a BGR frame. Returns True if motion detected in this frame."""
        return self.detect_gray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def detect_gray(self, gray: np.ndarray) -> bool:
        """Process an already-grayscale frame (single channel uint8).

        Used directly by the GStreamer engine's motion gate, which reads the
        Y plane of NV12 frames. That Y is BT.601/709 LIMITED range (16–235),
        so a luminance step reads ~219/255 (≈14%) smaller here than in the
        full-range COLOR_BGR2GRAY the cv2 path feeds — a MULTIPLICATIVE
        compression, not a small additive offset. Measured on all five bench
        clips (2026-08-17, diff_threshold=25, area_ratio=0.015): frame-level
        agreement with the cv2 path is 96–100% with NO compensation, and both
        candidate compensations (range expansion, threshold ×219/255) make
        agreement slightly WORSE — after the 21×21 blur and area_ratio gate,
        real motion sits far above the threshold, so the compression never
        moves an event boundary. Left uncompensated by design; revisit only
        if a future clip shows boundary drift.
        """
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        changed_ratio = np.count_nonzero(thresh) / thresh.size

        if changed_ratio >= self.area_ratio:
            self._static_count = 0
            return True
        else:
            self._static_count += 1
            return False

    def reset(self):
        self._prev_gray = None
        self._static_count = 0
