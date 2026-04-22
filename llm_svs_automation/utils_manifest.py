#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manifest 读写工具：支持 .list / .json / .jsonl。

该模块提供统一的样本结构，方便子集划分、推理、评估脚本复用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ManifestSpec:
    """描述 .list 文本格式字段位置。"""

    delimiter: str = "|"
    wav_idx: int = 0
    text_idx: int = 1
    phoneme_idx: int = 2
    pitch_idx: int = 3
    duration_idx: int = 4
    speaker_idx: int = -1
    utt_id_idx: int = -1


@dataclass
class SampleRecord:
    """统一样本结构。"""

    utt_id: str
    wav_path: str
    text: str = ""
    phoneme: str = ""
    pitch: Any = None
    duration: Optional[float] = None
    speaker: str = ""
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        return out


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _normalize_pitch(x: Any) -> Any:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # 优先尝试 JSON 解析，例如 "[110,112,...]"
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s)
                return v if isinstance(v, list) else [v]
            except json.JSONDecodeError:
                pass
        # 其次尝试逗号分隔
        if "," in s:
            vals: List[float] = []
            for token in s.split(","):
                token = token.strip()
                if token:
                    try:
                        vals.append(float(token))
                    except ValueError:
                        continue
            return vals
    return x


def _resolve_utt_id(i: int, wav_path: str, record: Optional[Dict[str, Any]] = None, utt_id_idx: int = -1) -> str:
    if record is not None:
        if "utt_id" in record and str(record["utt_id"]).strip():
            return str(record["utt_id"]).strip()
        if "id" in record and str(record["id"]).strip():
            return str(record["id"]).strip()
    wav_name = Path(wav_path).stem
    return wav_name if wav_name else f"utt_{i:08d}"


def parse_list_line(line: str, i: int, spec: ManifestSpec) -> Optional[SampleRecord]:
    """解析 .list 文件单行，失败时返回 None。"""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split(spec.delimiter)
    if len(parts) <= max(spec.wav_idx, spec.text_idx):
        return None

    def get_field(idx: int, default: Any = "") -> Any:
        if idx < 0 or idx >= len(parts):
            return default
        return parts[idx].strip()

    wav_path = get_field(spec.wav_idx, "")
    text = get_field(spec.text_idx, "")
    phoneme = get_field(spec.phoneme_idx, "")
    pitch = _normalize_pitch(get_field(spec.pitch_idx, ""))
    duration = _safe_float(get_field(spec.duration_idx, None))
    speaker = get_field(spec.speaker_idx, "")
    utt_id_raw = get_field(spec.utt_id_idx, "") if spec.utt_id_idx >= 0 else ""
    utt_id = utt_id_raw if utt_id_raw else _resolve_utt_id(i, wav_path)

    return SampleRecord(
        utt_id=utt_id,
        wav_path=wav_path,
        text=text,
        phoneme=phoneme,
        pitch=pitch,
        duration=duration,
        speaker=speaker,
    )


def _record_to_sample(record: Dict[str, Any], i: int) -> Optional[SampleRecord]:
    wav_path = str(
        record.get("wav_path")
        or record.get("wav")
        or record.get("audio")
        or record.get("audio_path")
        or ""
    ).strip()
    if not wav_path:
        return None

    text = str(record.get("text") or record.get("lyrics") or record.get("sentence") or "")
    phoneme = str(record.get("phoneme") or record.get("phones") or record.get("ph") or "")
    pitch = _normalize_pitch(record.get("pitch") or record.get("f0") or record.get("note_pitch") or [])
    duration = _safe_float(record.get("duration") or record.get("dur") or record.get("audio_duration"))
    speaker = str(record.get("speaker") or record.get("spk") or record.get("singer") or "")
    utt_id = _resolve_utt_id(i, wav_path, record=record)

    return SampleRecord(
        utt_id=utt_id,
        wav_path=wav_path,
        text=text,
        phoneme=phoneme,
        pitch=pitch,
        duration=duration,
        speaker=speaker,
        raw=record,
    )


def load_manifest(path: str, spec: Optional[ManifestSpec] = None) -> List[SampleRecord]:
    """读取 manifest，自动识别 .list / .json / .jsonl。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest 不存在: {p}")

    spec = spec or ManifestSpec()
    suffix = p.suffix.lower()
    samples: List[SampleRecord] = []

    if suffix in {".list", ".txt", ".tsv", ".csv"}:
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                rec = parse_list_line(line, i, spec)
                if rec is not None:
                    samples.append(rec)
        return samples

    if suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sample = _record_to_sample(rec, i)
                if sample is not None:
                    samples.append(sample)
        return samples

    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        items: Iterable[Dict[str, Any]]
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if isinstance(data.get("items"), list):
                items = data["items"]
            elif isinstance(data.get("data"), list):
                items = data["data"]
            elif isinstance(data.get("samples"), list):
                items = data["samples"]
            else:
                raise ValueError(f"无法识别 JSON 顶层结构: {p}")
        else:
            raise ValueError(f"不支持的 JSON 格式: {p}")

        for i, rec in enumerate(items):
            if not isinstance(rec, dict):
                continue
            sample = _record_to_sample(rec, i)
            if sample is not None:
                samples.append(sample)
        return samples

    raise ValueError(f"不支持的 manifest 后缀: {p.suffix}")


def write_jsonl(samples: List[SampleRecord], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def write_list(samples: List[SampleRecord], path: str, mode: str = "phoneme_pitch", delimiter: str = "|") -> None:
    """写出训练 list。

    mode:
    - text_only: wav|text
    - phoneme: wav|text|phoneme
    - phoneme_pitch: wav|text|phoneme|pitch_json
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        for s in samples:
            if mode == "text_only":
                row = [s.wav_path, s.text]
            elif mode == "phoneme":
                row = [s.wav_path, s.text, s.phoneme]
            elif mode == "phoneme_pitch":
                pitch_str = json.dumps(s.pitch if s.pitch is not None else [], ensure_ascii=False)
                row = [s.wav_path, s.text, s.phoneme, pitch_str]
            else:
                raise ValueError(f"未知 mode: {mode}")
            f.write(delimiter.join(str(x) for x in row) + "\n")
