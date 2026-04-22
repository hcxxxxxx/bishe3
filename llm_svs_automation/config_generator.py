#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于模板自动生成 GPT-SoVITS 实验配置。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_scales(scales_arg: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for token in scales_arg.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token.endswith("h"):
            value = float(token[:-1])
            label = token
        else:
            value = float(token)
            label = f"{value:g}h"
        out.append((label, value))
    if not out:
        raise ValueError("--scales 不能为空")
    return sorted(out, key=lambda x: x[1])


def load_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("缺少 pyyaml，请安装: pip install pyyaml") from e

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("缺少 pyyaml，请安装: pip install pyyaml") from e

    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """按 a.b.c 的键路径写入字典，不存在时自动创建。"""
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def default_hparams(hours: float, mode: str) -> Dict[str, Any]:
    """默认超参策略（可作为基线，后续可按你论文策略调整）。"""
    if hours <= 1.0:
        batch_size, lr, epochs = 16, 1.0e-4, 320
    elif hours <= 5.0:
        batch_size, lr, epochs = 24, 1.8e-4, 240
    else:
        batch_size, lr, epochs = 32, 2.5e-4, 180

    # 标注越粗，通常需要更稳的学习率
    if mode == "text_only":
        lr *= 0.75
        epochs += 40
    elif mode == "phoneme":
        lr *= 0.9
        epochs += 20

    return {
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "epochs": int(epochs),
    }


def load_override(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("override 文件必须是 JSON 对象")
    return data


def get_override(override: Dict[str, Any], scale: str, mode: str) -> Dict[str, Any]:
    """支持两层覆盖：

    1) override["default"]
    2) override["1h"]["text_only"]
    """
    out: Dict[str, Any] = {}
    if isinstance(override.get("default"), dict):
        out.update(override["default"])
    if isinstance(override.get(scale), dict):
        scale_obj = override[scale]
        if isinstance(scale_obj.get(mode), dict):
            out.update(scale_obj[mode])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="生成实验配置与实验索引表")
    parser.add_argument("--template", required=True, help="基础配置模板 (.yaml/.yml/.json)")
    parser.add_argument("--manifest_root", required=True, help="由 data_subsetting.py 生成的根目录")
    parser.add_argument("--output_dir", required=True, help="配置输出目录")
    parser.add_argument("--save_root", required=True, help="训练输出目录根路径")

    parser.add_argument("--scales", default="1h,5h,20h")
    parser.add_argument("--modes", default="text_only,phoneme,phoneme_pitch")
    parser.add_argument("--manifest_name", default="train.list", choices=["train.list", "train.jsonl"])
    parser.add_argument("--override_json", default="", help="可选超参覆盖 JSON")

    # 配置键路径（按你的 GPT-SoVITS 配置实际字段可直接改）
    parser.add_argument("--train_list_key", default="data.train_list")
    parser.add_argument("--batch_size_key", default="train.batch_size")
    parser.add_argument("--lr_key", default="train.learning_rate")
    parser.add_argument("--epochs_key", default="train.max_epoch")
    parser.add_argument("--save_dir_key", default="train.output_dir")

    # DDP 参数
    parser.add_argument("--ddp_enable_key", default="train.use_ddp")
    parser.add_argument("--world_size_key", default="train.world_size")
    parser.add_argument("--nproc_key", default="train.nproc_per_node")
    parser.add_argument("--backend_key", default="train.dist_backend")
    parser.add_argument("--master_addr_key", default="train.master_addr")
    parser.add_argument("--master_port_key", default="train.master_port")
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--nproc_per_node", type=int, default=8)
    parser.add_argument("--master_addr", default="127.0.0.1")
    parser.add_argument("--master_port_base", type=int, default=29500)

    parser.add_argument("--default_ngpu", type=int, default=2, help="写入 experiments.csv 的默认 ngpu")
    parser.add_argument("--out_format", default="yaml", choices=["yaml", "json"])

    args = parser.parse_args()

    template_path = Path(args.template)
    template_cfg = load_config(template_path)

    scales = parse_scales(args.scales)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    override = load_override(args.override_json)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for idx_s, (scale_label, scale_hours) in enumerate(scales):
        for idx_m, mode in enumerate(modes):
            exp_id = f"{scale_label}_{mode}"
            train_manifest = Path(args.manifest_root) / scale_label / mode / args.manifest_name
            if not train_manifest.exists():
                raise FileNotFoundError(f"未找到训练列表: {train_manifest}")

            hparams = default_hparams(scale_hours, mode)
            hparams.update(get_override(override, scale_label, mode))

            cfg = copy.deepcopy(template_cfg)
            set_nested(cfg, args.train_list_key, str(train_manifest))
            set_nested(cfg, args.batch_size_key, hparams["batch_size"])
            set_nested(cfg, args.lr_key, hparams["learning_rate"])
            set_nested(cfg, args.epochs_key, hparams["epochs"])

            exp_save_dir = save_root / exp_id
            set_nested(cfg, args.save_dir_key, str(exp_save_dir))

            # DDP 写入
            set_nested(cfg, args.ddp_enable_key, True)
            set_nested(cfg, args.world_size_key, args.world_size)
            set_nested(cfg, args.nproc_key, args.nproc_per_node)
            set_nested(cfg, args.backend_key, "nccl")
            set_nested(cfg, args.master_addr_key, args.master_addr)
            set_nested(cfg, args.master_port_key, args.master_port_base + idx_s * 10 + idx_m)

            ext = ".json" if args.out_format == "json" else ".yaml"
            cfg_path = out_root / f"{exp_id}{ext}"
            dump_config(cfg, cfg_path)

            rows.append(
                {
                    "exp_id": exp_id,
                    "data_scale": scale_label,
                    "annotation_mode": mode,
                    "train_manifest": str(train_manifest),
                    "config_path": str(cfg_path),
                    "save_dir": str(exp_save_dir),
                    "ngpu": args.default_ngpu,
                    "batch_size": hparams["batch_size"],
                    "learning_rate": hparams["learning_rate"],
                    "epochs": hparams["epochs"],
                }
            )

    csv_path = out_root / "experiments.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "exp_id",
                "data_scale",
                "annotation_mode",
                "train_manifest",
                "config_path",
                "save_dir",
                "ngpu",
                "batch_size",
                "learning_rate",
                "epochs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    (out_root / "experiments.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] 生成配置数量: {len(rows)}")
    print(f"[DONE] 索引表: {csv_path}")


if __name__ == "__main__":
    main()
