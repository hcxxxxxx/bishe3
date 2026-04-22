#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量推理 + 指标评估。

支持：
1) 按实验表逐个加载 checkpoint/config 并执行推理命令模板。
2) 计算 F0 RMSE / MCD / Speaker Similarity（可选 resemblyzer）。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from utils_manifest import ManifestSpec, load_manifest
from utils_metrics import SpeakerEncoderWrapper, check_audio_pair, f0_rmse, mcd


def read_exp_table(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def render_cmd(template: str, mapping: Dict[str, Any]) -> str:
    cmd = template
    for k, v in mapping.items():
        cmd = cmd.replace("{" + k + "}", str(v))
    return cmd


def run_inference_once(exp: Dict[str, Any], test_manifest: str, pred_dir: str, infer_cmd_template: str, dry_run: bool = False) -> None:
    mapping = {
        "exp_id": exp.get("exp_id", ""),
        "config": exp.get("config_path", ""),
        "ckpt": exp.get("ckpt_path", exp.get("checkpoint", "")),
        "test_manifest": test_manifest,
        "pred_dir": pred_dir,
        "save_dir": exp.get("save_dir", ""),
        "data_scale": exp.get("data_scale", ""),
        "annotation_mode": exp.get("annotation_mode", ""),
    }
    cmd = render_cmd(infer_cmd_template, mapping)
    print(f"[INFER] {exp.get('exp_id', '')}: {cmd}")
    if dry_run:
        return
    subprocess.run(cmd, shell=True, check=True)


def resolve_gt_wav(sample, audio_root: str) -> str:
    p = Path(sample.wav_path)
    if p.is_absolute() or not audio_root:
        return str(p)
    return str(Path(audio_root) / p)


def resolve_pred_wav(sample, pred_dir: str, pred_ext: str, gt_wav: str) -> str:
    pred_root = Path(pred_dir)

    # 约定1: 用 utt_id 命名
    cand1 = pred_root / f"{sample.utt_id}{pred_ext}"
    if cand1.exists():
        return str(cand1)

    # 约定2: 用 GT 文件 basename 命名
    stem = Path(gt_wav).stem
    cand2 = pred_root / f"{stem}{pred_ext}"
    if cand2.exists():
        return str(cand2)

    # 约定3: 保持相对路径结构（仅替换后缀）
    rel = Path(sample.wav_path)
    cand3 = pred_root / rel.with_suffix(pred_ext)
    if cand3.exists():
        return str(cand3)

    # 返回默认路径，供错误信息使用
    return str(cand1)


def aggregate_metric(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    v = arr[valid]
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "count": int(v.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="批量推理并计算指标")
    parser.add_argument("--exp_table", required=True, help="实验索引表 CSV")
    parser.add_argument("--test_manifest", required=True, help="测试集 manifest")
    parser.add_argument("--output_dir", required=True, help="评估输出目录")

    parser.add_argument("--audio_root", default="", help="测试 manifest 中音频相对路径的根目录")
    parser.add_argument("--pred_ext", default=".wav", help="预测音频后缀")

    parser.add_argument("--infer_cmd_template", default="", help="可选，推理命令模板")
    parser.add_argument("--skip_infer", action="store_true", help="跳过推理，仅计算已有输出")
    parser.add_argument("--dry_run", action="store_true")

    # manifest 解析参数（便于兼容 .list）
    parser.add_argument("--delimiter", default="|")
    parser.add_argument("--wav_idx", type=int, default=0)
    parser.add_argument("--text_idx", type=int, default=1)
    parser.add_argument("--phoneme_idx", type=int, default=2)
    parser.add_argument("--pitch_idx", type=int, default=3)
    parser.add_argument("--duration_idx", type=int, default=4)
    parser.add_argument("--speaker_idx", type=int, default=-1)
    parser.add_argument("--utt_id_idx", type=int, default=-1)

    # 指标参数
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--fmin", type=float, default=50.0)
    parser.add_argument("--fmax", type=float, default=1100.0)
    parser.add_argument("--enable_spk_similarity", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    test_samples = load_manifest(args.test_manifest, spec=spec)
    exps = read_exp_table(args.exp_table)

    if not exps:
        raise RuntimeError(f"实验表为空: {args.exp_table}")
    if not test_samples:
        raise RuntimeError(f"测试集为空: {args.test_manifest}")

    spk_encoder = SpeakerEncoderWrapper() if args.enable_spk_similarity else None
    if args.enable_spk_similarity and not spk_encoder.enabled:
        print("[WARN] resemblyzer 不可用，Speaker Similarity 将返回 NaN")

    per_utt_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for exp in exps:
        exp_id = exp.get("exp_id", "unknown_exp")
        pred_dir = exp.get("pred_dir") or str(out_dir / "predictions" / exp_id)
        Path(pred_dir).mkdir(parents=True, exist_ok=True)

        if not args.skip_infer:
            if not args.infer_cmd_template:
                raise ValueError("未设置 --infer_cmd_template，无法执行推理")
            run_inference_once(
                exp=exp,
                test_manifest=args.test_manifest,
                pred_dir=pred_dir,
                infer_cmd_template=args.infer_cmd_template,
                dry_run=args.dry_run,
            )

        f0_vals: List[float] = []
        mcd_vals: List[float] = []
        spk_vals: List[float] = []
        miss_count = 0

        iterator = tqdm(test_samples, desc=f"Eval {exp_id}", ncols=100)
        for sample in iterator:
            gt_wav = resolve_gt_wav(sample, audio_root=args.audio_root)
            pd_wav = resolve_pred_wav(sample, pred_dir=pred_dir, pred_ext=args.pred_ext, gt_wav=gt_wav)

            ok, err = check_audio_pair(gt_wav, pd_wav)
            if not ok:
                miss_count += 1
                per_utt_rows.append(
                    {
                        "exp_id": exp_id,
                        "utt_id": sample.utt_id,
                        "gt_wav": gt_wav,
                        "pred_wav": pd_wav,
                        "f0_rmse": float("nan"),
                        "mcd": float("nan"),
                        "spk_similarity": float("nan"),
                        "error": err,
                    }
                )
                continue

            val_f0 = f0_rmse(gt_wav, pd_wav, sr=args.sr, hop_length=args.hop_length, fmin=args.fmin, fmax=args.fmax)
            val_mcd = mcd(gt_wav, pd_wav, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
            if spk_encoder is not None and spk_encoder.enabled:
                val_spk = spk_encoder.cosine_similarity(gt_wav, pd_wav)
            else:
                val_spk = float("nan")

            f0_vals.append(val_f0)
            mcd_vals.append(val_mcd)
            spk_vals.append(val_spk)

            per_utt_rows.append(
                {
                    "exp_id": exp_id,
                    "utt_id": sample.utt_id,
                    "gt_wav": gt_wav,
                    "pred_wav": pd_wav,
                    "f0_rmse": val_f0,
                    "mcd": val_mcd,
                    "spk_similarity": val_spk,
                    "error": "",
                }
            )

        agg_f0 = aggregate_metric(f0_vals)
        agg_mcd = aggregate_metric(mcd_vals)
        agg_spk = aggregate_metric(spk_vals)

        summary_rows.append(
            {
                "exp_id": exp_id,
                "data_scale": exp.get("data_scale", ""),
                "annotation_mode": exp.get("annotation_mode", ""),
                "pred_dir": pred_dir,
                "num_test_samples": len(test_samples),
                "missing_pairs": miss_count,
                "f0_rmse_mean": agg_f0["mean"],
                "f0_rmse_std": agg_f0["std"],
                "f0_valid_count": agg_f0["count"],
                "mcd_mean": agg_mcd["mean"],
                "mcd_std": agg_mcd["std"],
                "mcd_valid_count": agg_mcd["count"],
                "spk_similarity_mean": agg_spk["mean"],
                "spk_similarity_std": agg_spk["std"],
                "spk_valid_count": agg_spk["count"],
            }
        )

        print(
            f"[SUMMARY] {exp_id}: "
            f"F0={agg_f0['mean']:.4f}, MCD={agg_mcd['mean']:.4f}, SPK={agg_spk['mean']:.4f}, miss={miss_count}"
        )

    per_utt_path = out_dir / "per_utt_metrics.csv"
    with per_utt_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "exp_id",
                "utt_id",
                "gt_wav",
                "pred_wav",
                "f0_rmse",
                "mcd",
                "spk_similarity",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(per_utt_rows)

    summary_path = out_dir / "metrics_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "exp_id",
                "data_scale",
                "annotation_mode",
                "pred_dir",
                "num_test_samples",
                "missing_pairs",
                "f0_rmse_mean",
                "f0_rmse_std",
                "f0_valid_count",
                "mcd_mean",
                "mcd_std",
                "mcd_valid_count",
                "spk_similarity_mean",
                "spk_similarity_std",
                "spk_valid_count",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    (out_dir / "metrics_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] per-utt: {per_utt_path}")
    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    main()
