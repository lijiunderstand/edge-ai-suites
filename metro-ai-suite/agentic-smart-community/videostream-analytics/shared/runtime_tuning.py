"""Runtime thread tuning — pin library thread counts before any capture starts.

Both engines historically ran with library defaults: `cv2.setNumThreads` at
OpenCV's compile-time default (16 on the bench host) and OpenCV's FFmpeg
capture threading on "auto". Measured on the same 720p60 clip (bench
2026-08-17, §1.1/1.2 of benchmark_parity_plan_2026-08.md):

  - motion detection at 16 threads costs 4.2–4.8× the CPU of 1 thread
    (0.46–0.53 core/stream vs 0.11 @60fps) — thread overhead, not real work;
  - "auto" decode threading differs between cv2 and gst, making cross-engine
    CPU numbers non-comparable.

Pinning both to 1 by default removes thread scheduling as an uncontrolled
variable so CPU figures reflect real compute. Override via env when threaded
behavior is what you want to measure — but both engines must see the same
values:

  VSA_CV2_THREADS    — cv2.setNumThreads() (motion-detect blur/absdiff; used
                       by BOTH engines, gst runs the same MotionDetector)
  VSA_DECODE_THREADS — OPENCV_FFMPEG_THREADS for cv2 captures; the gst engine
                       reads the same var for avdec_h264 max-threads
                       (gst_engine/builder.py).

Call `apply()` before the first VideoCapture / Gst element is created:
OPENCV_FFMPEG_THREADS is read and cached by OpenCV at capture-open time.
"""

from __future__ import annotations

import logging
import os

import cv2

logger = logging.getLogger(__name__)


def apply() -> dict[str, int | str | None]:
    """Pin thread counts from env (idempotent). Returns the effective values."""
    cv2_threads = _env_int("VSA_CV2_THREADS", 1)
    decode_threads = _env_int("VSA_DECODE_THREADS", 1)

    cv2.setNumThreads(cv2_threads)
    # setdefault: an explicit operator setting always wins over the default.
    os.environ.setdefault("OPENCV_FFMPEG_THREADS", str(decode_threads))

    effective = {
        "cv2_threads": cv2.getNumThreads(),
        "opencv_ffmpeg_threads": os.environ.get("OPENCV_FFMPEG_THREADS"),
        "vsa_decode_threads": decode_threads,
    }
    logger.info(
        "Runtime tuning: cv2 threads=%d, OPENCV_FFMPEG_THREADS=%s, VSA_DECODE_THREADS=%d",
        effective["cv2_threads"],  # type: ignore[arg-type]
        effective["opencv_ffmpeg_threads"],
        decode_threads,
    )
    return effective


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default
