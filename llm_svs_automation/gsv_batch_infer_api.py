#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调用 GPT-SoVITS API(v2) 批量推理。

用途：
1) 读取测试 manifest（.json/.jsonl/.list）。
2) 调用 HTTP 接口 `/tts` 合成。
3) 按 `utt_id.wav` 写入输出目录，便于对接 batch_infer_and_eval.py。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from utils_manifest import ManifestSpec, load_manifest


def post_json(url: str, payload: Dict[str, Any], timeout: float) -> bytes:
    """向 API 发送 JSON POST 请求并返回二进制响应体。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def looks_like_wav(raw: bytes) -> bool:
    # WAV 文件头一般是 RIFF....WAVE
    return len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"


def main() -> None:
    parser = argparse.ArgumentParser(description="调用 GPT-SoVITS API 批量推理")
    parser.add_argument("--test_manifest", required=True, help="测试集 manifest")
    parser.add_argument("--pred_dir", required=True, help="预测 wav 输出目录")
    parser.add_argument("--api_base", default="http://127.0.0.1:9880", help="API 地址，例如 http://127.0.0.1:9880")
    parser.add_argument("--api_path", default="/tts", help="TTS 接口路径，默认 /tts")

    # GPT-SoVITS 推理参数（最常用）
    parser.add_argument("--ref_audio_path", required=True, help="参考音频路径（克隆音色）")
    parser.add_argument("--prompt_text", default="", help="参考音频对应文本")
    parser.add_argument("--prompt_lang", default="zh", help="参考文本语言，常见 zh/en/ja")
    parser.add_argument("--text_lang", default="zh", help="目标文本语言")
    parser.add_argument("--text_split_method", default="cut5", help="切句策略，例如 cut0/cut5")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1, help="API 侧 batch_size")
    parser.add_argument("--media_type", default="wav", help="输出格式，默认 wav")
    parser.add_argument("--streaming_mode", default="false", help="是否流式，默认 false")

    parser.add_argument("--overwrite", type=int, default=0, help="是否覆盖已有 wav，1=覆盖")
    parser.add_argument("--max_samples", type=int, default=0, help="仅跑前 N 条，0=全部")
    parser.add_argument("--timeout", type=float, default=120.0, help="单条请求超时秒数")
    parser.add_argument("--retry", type=int, default=2, help="失败重试次数")
    parser.add_argument("--sleep", type=float, default=0.05, help="每条请求后的停顿秒数")
    parser.add_argument("--log_csv", default="", help="明细日志 csv，默认 pred_dir/infer_report.csv")

    # manifest 兼容参数（主要用于 .list）
    parser.add_argument("--delimiter", default="|")
    parser.add_argument("--wav_idx", type=int, default=0)
    parser.add_argument("--text_idx", type=int, default=1)
    parser.add_argument("--phoneme_idx", type=int, default=2)
    parser.add_argument("--pitch_idx", type=int, default=3)
    parser.add_argument("--duration_idx", type=int, default=4)
    parser.add_argument("--speaker_idx", type=int, default=-1)
    parser.add_argument("--utt_id_idx", type=int, default=-1)

    args = parser.parse_args()

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
    samples = load_manifest(args.test_manifest, spec=spec)
    if not samples:
        raise RuntimeError(f"测试集为空: {args.test_manifest}")
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv) if args.log_csv else pred_dir / "infer_report.csv"
    tts_url = args.api_base.rstrip("/") + args.api_path

    ok_cnt = 0
    skip_cnt = 0
    fail_cnt = 0

    with log_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utt_id", "text_len", "pred_wav", "status", "message", "elapsed_sec"],
        )
        writer.writeheader()

        for i, s in enumerate(samples, 1):
            utt = (s.utt_id or f"utt_{i:08d}").replace("/", "_").replace("#", "_")
            text = (s.text or "").strip()
            out_wav = pred_dir / f"{utt}.wav"

            if not text:
                fail_cnt += 1
                writer.writerow(
                    {
                        "utt_id": utt,
                        "text_len": 0,
                        "pred_wav": str(out_wav),
                        "status": "fail",
                        "message": "empty text",
                        "elapsed_sec": 0.0,
                    }
                )
                continue

            if out_wav.exists() and args.overwrite != 1:
                skip_cnt += 1
                writer.writerow(
                    {
                        "utt_id": utt,
                        "text_len": len(text),
                        "pred_wav": str(out_wav),
                        "status": "skip",
                        "message": "exists",
                        "elapsed_sec": 0.0,
                    }
                )
                continue

            payload: Dict[str, Any] = {
                "text": text,
                "text_lang": args.text_lang,
                "ref_audio_path": args.ref_audio_path,
                "prompt_text": args.prompt_text,
                "prompt_lang": args.prompt_lang,
                "text_split_method": args.text_split_method,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "speed_factor": args.speed_factor,
                "batch_size": args.batch_size,
                "media_type": args.media_type,
                "streaming_mode": args.streaming_mode,
            }

            start_t = time.time()
            last_err: Optional[str] = None
            raw: Optional[bytes] = None

            for k in range(args.retry + 1):
                try:
                    raw = post_json(tts_url, payload, timeout=args.timeout)
                    break
                except urllib.error.HTTPError as e:
                    body = ""
                    try:
                        body = e.read().decode("utf-8", errors="ignore")
                    except Exception:
                        pass
                    last_err = f"HTTPError {e.code}: {body[:400]}"
                except Exception as e:  # noqa: BLE001
                    last_err = repr(e)

                if k < args.retry:
                    time.sleep(min(2.0, 0.5 * (k + 1)))

            elapsed = round(time.time() - start_t, 3)

            if raw is None:
                fail_cnt += 1
                writer.writerow(
                    {
                        "utt_id": utt,
                        "text_len": len(text),
                        "pred_wav": str(out_wav),
                        "status": "fail",
                        "message": last_err or "unknown error",
                        "elapsed_sec": elapsed,
                    }
                )
                print(f"[{i}/{len(samples)}] FAIL {utt}: {last_err}", file=sys.stderr)
                continue

            if not looks_like_wav(raw):
                # 某些报错会返回 JSON 文本，这里尽量保留可读信息
                msg = raw[:400].decode("utf-8", errors="ignore")
                fail_cnt += 1
                writer.writerow(
                    {
                        "utt_id": utt,
                        "text_len": len(text),
                        "pred_wav": str(out_wav),
                        "status": "fail",
                        "message": f"non-wav response: {msg}",
                        "elapsed_sec": elapsed,
                    }
                )
                print(f"[{i}/{len(samples)}] FAIL {utt}: non-wav response", file=sys.stderr)
                continue

            out_wav.write_bytes(raw)
            ok_cnt += 1
            writer.writerow(
                {
                    "utt_id": utt,
                    "text_len": len(text),
                    "pred_wav": str(out_wav),
                    "status": "ok",
                    "message": "",
                    "elapsed_sec": elapsed,
                }
            )
            if i % 20 == 0 or i == len(samples):
                print(f"[{i}/{len(samples)}] ok={ok_cnt} skip={skip_cnt} fail={fail_cnt}")
            time.sleep(max(0.0, args.sleep))

    print(f"[DONE] 预测完成: ok={ok_cnt}, skip={skip_cnt}, fail={fail_cnt}")
    print(f"[LOG] {log_csv}")


if __name__ == "__main__":
    main()

