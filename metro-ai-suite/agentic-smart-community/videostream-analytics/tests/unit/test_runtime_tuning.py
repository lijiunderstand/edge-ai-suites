"""Tests for shared.runtime_tuning — thread pinning before any capture opens."""

import os

import cv2

from shared import runtime_tuning


def test_apply_pins_cv2_threads(monkeypatch):
    monkeypatch.setenv("VSA_CV2_THREADS", "3")
    monkeypatch.setenv("VSA_DECODE_THREADS", "2")
    monkeypatch.delenv("OPENCV_FFMPEG_THREADS", raising=False)

    eff = runtime_tuning.apply()

    assert cv2.getNumThreads() == 3
    assert eff["cv2_threads"] == 3
    assert eff["opencv_ffmpeg_threads"] == "2"
    assert os.environ["OPENCV_FFMPEG_THREADS"] == "2"


def test_apply_defaults_to_one(monkeypatch):
    monkeypatch.delenv("VSA_CV2_THREADS", raising=False)
    monkeypatch.delenv("VSA_DECODE_THREADS", raising=False)
    monkeypatch.delenv("OPENCV_FFMPEG_THREADS", raising=False)

    eff = runtime_tuning.apply()

    assert eff["cv2_threads"] == 1
    assert eff["opencv_ffmpeg_threads"] == "1"


def test_existing_opencv_ffmpeg_threads_wins(monkeypatch):
    """An explicit operator setting is never overridden by the default."""
    monkeypatch.delenv("VSA_DECODE_THREADS", raising=False)
    monkeypatch.setenv("OPENCV_FFMPEG_THREADS", "4")

    eff = runtime_tuning.apply()

    assert eff["opencv_ffmpeg_threads"] == "4"


def test_invalid_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("VSA_CV2_THREADS", "not-a-number")
    monkeypatch.setenv("VSA_DECODE_THREADS", "abc")
    monkeypatch.delenv("OPENCV_FFMPEG_THREADS", raising=False)

    eff = runtime_tuning.apply()

    assert eff["cv2_threads"] == 1
    assert eff["opencv_ffmpeg_threads"] == "1"
