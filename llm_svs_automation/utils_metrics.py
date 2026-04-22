#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估指标工具：F0 RMSE, MCD, Speaker Similarity。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class SpeakerEncoderWrapper:
    """对 resemblyzer 做薄封装，避免主流程硬依赖。"""

    def __init__(self) -> None:
        self._enabled = False
        self._encoder = None
        self._preprocess_wav = None
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav

            self._encoder = VoiceEncoder()
            self._preprocess_wav = preprocess_wav
            self._enabled = True
        except Exception:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def cosine_similarity(self, wav_a: str, wav_b: str) -> float:
        if not self._enabled:
            return float("nan")
        emb_a = self._encoder.embed_utterance(self._preprocess_wav(wav_a))
        emb_b = self._encoder.embed_utterance(self._preprocess_wav(wav_b))
        denom = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        if denom == 0:
            return float("nan")
        return float(np.dot(emb_a, emb_b) / denom)


def load_audio(path: str, sr: int) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


def _interp_to_len(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x
    if len(x) <= 1:
        return np.full((target_len,), x[0] if len(x) else np.nan, dtype=np.float32)
    old_idx = np.linspace(0, 1, num=len(x), endpoint=True)
    new_idx = np.linspace(0, 1, num=target_len, endpoint=True)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def f0_rmse(gt_wav: str, pred_wav: str, sr: int = 24000, hop_length: int = 256, fmin: float = 50.0, fmax: float = 1100.0) -> float:
    """基于 librosa.pyin 计算 F0 RMSE（仅在双方有声帧上统计）。"""
    import librosa

    y_gt = load_audio(gt_wav, sr)
    y_pr = load_audio(pred_wav, sr)

    f0_gt, _, _ = librosa.pyin(y_gt, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    f0_pr, _, _ = librosa.pyin(y_pr, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)

    if f0_gt is None or f0_pr is None:
        return float("nan")

    t = max(len(f0_gt), len(f0_pr))
    f0_gt = _interp_to_len(np.asarray(f0_gt, dtype=np.float32), t)
    f0_pr = _interp_to_len(np.asarray(f0_pr, dtype=np.float32), t)

    valid = np.isfinite(f0_gt) & np.isfinite(f0_pr)
    if valid.sum() < 5:
        return float("nan")

    rmse = math.sqrt(float(np.mean((f0_gt[valid] - f0_pr[valid]) ** 2)))
    return rmse


def mcd(gt_wav: str, pred_wav: str, sr: int = 24000, n_fft: int = 1024, hop_length: int = 256, n_mfcc: int = 13) -> float:
    """计算 Mel-Cepstral Distortion (MCD, dB)。

    公式: (10 / ln(10)) * mean_t sqrt(2 * sum((c_t - c'_t)^2))
    默认使用 MFCC 1..12（去掉第0维能量项）。
    """
    import librosa

    y_gt = load_audio(gt_wav, sr)
    y_pr = load_audio(pred_wav, sr)

    c_gt = librosa.feature.mfcc(y=y_gt, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    c_pr = librosa.feature.mfcc(y=y_pr, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)

    # 去掉 0 维
    c_gt = c_gt[1:, :]
    c_pr = c_pr[1:, :]

    t = min(c_gt.shape[1], c_pr.shape[1])
    if t <= 1:
        return float("nan")

    diff = c_gt[:, :t] - c_pr[:, :t]
    dist = np.sqrt(2.0 * np.sum(diff * diff, axis=0))
    factor = 10.0 / np.log(10.0)
    return float(factor * np.mean(dist))


def check_audio_pair(gt_wav: str, pred_wav: str) -> Tuple[bool, Optional[str]]:
    if not Path(gt_wav).exists():
        return False, f"GT 不存在: {gt_wav}"
    if not Path(pred_wav).exists():
        return False, f"Pred 不存在: {pred_wav}"
    return True, None
