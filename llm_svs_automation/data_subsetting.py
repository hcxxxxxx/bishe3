#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按目标时长构建训练子集，并生成不同标注细粒度版本。"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

from utils_manifest import ManifestSpec, SampleRecord, load_manifest, write_jsonl, write_list


def parse_hours(hours_arg: str) -> List[Tuple[str, float]]:
    """解析如 `1h,5h,20h` 的输入。"""
    results: List[Tuple[str, float]] = []
    for token in hours_arg.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token.endswith("h"):
            value = float(token[:-1])
            label = token
        else:
            value = float(token)
            label = f"{value:g}h"
        results.append((label, value))
    if not results:
        raise ValueError("hours 不能为空，例如: 1h,5h,20h")
    return sorted(results, key=lambda x: x[1])


def build_audio_path(wav_path: str, audio_root: str) -> str:
    p = Path(wav_path)
    if p.is_absolute() or not audio_root:
        return str(p)
    return str(Path(audio_root) / p)


def fill_missing_duration(samples: List[SampleRecord], audio_root: str, strict: bool) -> List[SampleRecord]:
    """当 manifest 中缺失时长时，使用音频文件元信息补齐。"""
    try:
        import soundfile as sf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("缺少 soundfile 依赖，请先安装: pip install soundfile") from e

    out: List[SampleRecord] = []
    dropped = 0
    for s in samples:
        if s.duration is not None and s.duration > 0:
            out.append(s)
            continue

        wav_abspath = build_audio_path(s.wav_path, audio_root)
        try:
            info = sf.info(wav_abspath)
            dur = float(info.frames) / float(info.samplerate)
            out.append(replace(s, duration=dur))
        except Exception:
            if strict:
                raise
            dropped += 1
    if dropped > 0:
        print(f"[WARN] 有 {dropped} 条样本无法获得时长，已跳过。")
    return out


def pick_prefix_subsets(shuffled: List[SampleRecord], targets_hours: List[Tuple[str, float]]) -> Dict[str, List[SampleRecord]]:
    """用同一随机顺序做前缀切片，保证 1h ⊂ 5h ⊂ 20h。"""
    subsets: Dict[str, List[SampleRecord]] = {}
    cumulative = 0.0
    idx = 0

    sorted_targets = sorted(targets_hours, key=lambda x: x[1])
    for label, hour in sorted_targets:
        target_sec = hour * 3600.0
        while idx < len(shuffled) and cumulative < target_sec:
            dur = shuffled[idx].duration or 0.0
            cumulative += dur
            idx += 1
        subsets[label] = shuffled[:idx]
    return subsets


def degrade_annotation(samples: List[SampleRecord], mode: str) -> List[SampleRecord]:
    """标注降级。

    - text_only: 仅保留文本
    - phoneme: 保留文本+音素
    - phoneme_pitch: 文本+音素+音高
    """
    out: List[SampleRecord] = []
    for s in samples:
        if mode == "text_only":
            out.append(replace(s, phoneme="", pitch=[]))
        elif mode == "phoneme":
            out.append(replace(s, pitch=[]))
        elif mode == "phoneme_pitch":
            out.append(s)
        else:
            raise ValueError(f"未知 mode: {mode}")
    return out


def summarize(samples: List[SampleRecord]) -> Dict[str, float]:
    total_dur = sum((x.duration or 0.0) for x in samples)
    return {
        "num_samples": len(samples),
        "total_seconds": total_dur,
        "total_hours": total_dur / 3600.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建不同时长与标注粒度的训练子集")
    parser.add_argument("--input_manifest", required=True, help="输入 manifest: .list/.json/.jsonl")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--hours", default="1h,5h,20h", help="目标时长，例如: 1h,5h,20h")
    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--audio_root", default="", help="当 wav 路径为相对路径时用于拼接")
    parser.add_argument("--compute_missing_duration", action="store_true", help="若缺少时长则从音频读取")
    parser.add_argument("--strict", action="store_true", help="出现坏样本时直接报错")

    # .list 解析参数
    parser.add_argument("--delimiter", default="|", help=".list 分隔符")
    parser.add_argument("--wav_idx", type=int, default=0)
    parser.add_argument("--text_idx", type=int, default=1)
    parser.add_argument("--phoneme_idx", type=int, default=2)
    parser.add_argument("--pitch_idx", type=int, default=3)
    parser.add_argument("--duration_idx", type=int, default=4)
    parser.add_argument("--speaker_idx", type=int, default=-1)
    parser.add_argument("--utt_id_idx", type=int, default=-1)

    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    spec = ManifestSpec(
        delimiter=args.delimiter,
        wav_idx=args.wav_idx,
        text_idx=args.text_idx,
        phoneme_idx=args.phoneme_idx,
        pitch_idx=args.pitch_idx,
        duration_idx=args.duration_idx,
        speaker_idx=args.speaker_idx,
        utt_id_idx=args.utt_id_idx,
    )

    samples = load_manifest(args.input_manifest, spec=spec)
    print(f"[INFO] 读取样本数: {len(samples)}")

    # 补齐时长
    if args.compute_missing_duration:
        samples = fill_missing_duration(samples, audio_root=args.audio_root, strict=args.strict)

    # 过滤无时长样本
    valid = [x for x in samples if x.duration is not None and x.duration > 0]
    if not valid:
        raise RuntimeError("没有可用样本（时长缺失或非正数）。")

    rng = random.Random(args.seed)
    shuffled = valid[:]
    rng.shuffle(shuffled)

    targets = parse_hours(args.hours)
    subsets = pick_prefix_subsets(shuffled, targets)

    modes = ["text_only", "phoneme", "phoneme_pitch"]
    summary = {
        "input_manifest": str(args.input_manifest),
        "seed": args.seed,
        "targets": [k for k, _ in targets],
        "results": {},
    }

    for scale_label, subset in subsets.items():
        summary["results"][scale_label] = summarize(subset)
        print(
            f"[INFO] {scale_label}: {len(subset)} 条, "
            f"{summary['results'][scale_label]['total_hours']:.3f} h"
        )
        for mode in modes:
            degraded = degrade_annotation(subset, mode)
            exp_dir = out_root / scale_label / mode
            exp_dir.mkdir(parents=True, exist_ok=True)

            # 输出 jsonl（保留更完整字段）
            write_jsonl(degraded, str(exp_dir / "train.jsonl"))
            # 输出 list（便于直接对接常见训练脚本）
            write_list(degraded, str(exp_dir / "train.list"), mode=mode, delimiter=args.delimiter)

    with (out_root / "subset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 子集与标注版本已写入: {out_root}")


if __name__ == "__main__":
    main()
