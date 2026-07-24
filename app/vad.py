"""Voice activity detection using silero-vad (raw ONNX model, no torch dependency).

Speech segmentation adds hysteresis on top of the raw per-frame probability:
 - a run of `min_speech_ms` frames above `threshold` before a segment is considered started
 - a run of `min_silence_ms` frames below `threshold` before a segment is considered ended
 - `speech_pad_ms` of audio is kept from before the detected start / after the detected end,
   so words at the edges of a segment aren't clipped
These three values plus `threshold` are the knobs to retune later for false triggers / cutoffs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import onnxruntime as ort

from audio_capture import FRAME_SIZE, SAMPLE_RATE


@dataclass
class VadConfig:
    threshold: float = 0.5
    min_speech_ms: int = 100
    min_silence_ms: int = 600
    speech_pad_ms: int = 300
    sample_rate: int = SAMPLE_RATE
    frame_size: int = FRAME_SIZE


class SileroVAD:
    """Thin wrapper around the silero-vad ONNX graph: one call in, one probability out.

    The v5 model expects each chunk to be prefixed with the final 64 samples of the
    previous chunk (576 samples total at 16kHz); without this context the output
    probabilities are meaningless.
    """

    CONTEXT_SIZE = 64

    def __init__(self, model_path: str, sample_rate: int = SAMPLE_RATE):
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._sample_rate = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.CONTEXT_SIZE, dtype=np.float32)

    def speech_prob(self, chunk: np.ndarray) -> float:
        chunk = chunk.astype(np.float32)
        inp = np.concatenate([self._context, chunk]).reshape(1, -1)
        out, self._state = self._session.run(
            None, {"input": inp, "state": self._state, "sr": self._sample_rate}
        )
        self._context = chunk[-self.CONTEXT_SIZE:]
        return float(out[0, 0])


class SpeechSegmenter:
    """Feeds fixed-size frames through SileroVAD and emits speech-segment start/end events."""

    def __init__(self, vad: SileroVAD, config: VadConfig,
                 on_speech_start: Callable[[float, np.ndarray], None],
                 on_speech_end: Callable[[float, float, np.ndarray], None]):
        self._vad = vad
        self._cfg = config
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end

        frame_ms = 1000 * config.frame_size / config.sample_rate
        self._min_speech_frames = max(1, round(config.min_speech_ms / frame_ms))
        self._min_silence_frames = max(1, round(config.min_silence_ms / frame_ms))
        pad_frames = max(0, round(config.speech_pad_ms / frame_ms))
        self._pad_frames = pad_frames

        self._pre_roll: deque = deque(maxlen=pad_frames + self._min_speech_frames)
        self._in_speech = False
        self._consec_speech = 0
        self._consec_silence = 0
        self._segment_frames: list[np.ndarray] = []
        self._elapsed_s = 0.0
        self._speech_start_s = 0.0

    def feed(self, frame: np.ndarray) -> None:
        prob = self._vad.speech_prob(frame)
        is_speech = prob >= self._cfg.threshold
        frame_dur = len(frame) / self._cfg.sample_rate

        if not self._in_speech:
            self._pre_roll.append(frame)
            if is_speech:
                self._consec_speech += 1
                if self._consec_speech >= self._min_speech_frames:
                    self._in_speech = True
                    self._consec_silence = 0
                    self._segment_frames = list(self._pre_roll)
                    self._speech_start_s = self._elapsed_s - len(self._segment_frames) * frame_dur
                    initial_audio = np.concatenate(self._segment_frames) if self._segment_frames \
                        else np.zeros(0, dtype=np.float32)
                    self._on_speech_start(max(0.0, self._speech_start_s), initial_audio)
            else:
                self._consec_speech = 0
        else:
            self._segment_frames.append(frame)
            if is_speech:
                self._consec_silence = 0
            else:
                self._consec_silence += 1
                if self._consec_silence >= self._min_silence_frames:
                    trim = max(0, self._consec_silence - self._pad_frames)
                    kept = self._segment_frames[: len(self._segment_frames) - trim] if trim else self._segment_frames
                    audio = np.concatenate(kept) if kept else np.zeros(0, dtype=np.float32)
                    duration_s = len(audio) / self._cfg.sample_rate
                    self._on_speech_end(self._speech_start_s, duration_s, audio)

                    self._in_speech = False
                    self._consec_speech = 0
                    self._consec_silence = 0
                    self._segment_frames = []
                    self._pre_roll.clear()

        self._elapsed_s += frame_dur
