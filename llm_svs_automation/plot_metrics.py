#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取指标汇总 CSV，自动绘制折线图和消融柱状图。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_hours(scale: str) -> float:
    s = str(scale).strip().lower()
    if s.endswith("h"):
        s = s[:-1]
    return float(s)


def ensure_metrics(df: pd.DataFrame, metrics: List[str]) -> List[str]:
    valid: List[str] = []
    for m in metrics:
        if m in df.columns:
            valid.append(m)
        else:
            print(f"[WARN] 指标列不存在，已跳过: {m}")
    if not valid:
        raise ValueError("没有可用指标列可绘图")
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制实验结果图")
    parser.add_argument("--summary_csv", required=True, help="batch_infer_and_eval.py 产出的 metrics_summary.csv")
    parser.add_argument("--output_dir", required=True, help="图像输出目录")
    parser.add_argument("--metrics", default="f0_rmse_mean,mcd_mean,spk_similarity_mean")
    parser.add_argument("--style", default="whitegrid")
    parser.add_argument("--dpi", type=int, default=180)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style=args.style)

    df = pd.read_csv(args.summary_csv)
    if "data_scale" not in df.columns or "annotation_mode" not in df.columns:
        raise ValueError("summary_csv 缺少必须列: data_scale / annotation_mode")

    df["data_hours"] = df["data_scale"].map(parse_hours)
    df = df.sort_values(["data_hours", "annotation_mode"]).reset_index(drop=True)

    metric_list = [x.strip() for x in args.metrics.split(",") if x.strip()]
    metric_list = ensure_metrics(df, metric_list)

    # 1) 折线图：横轴数据规模，纵轴指标
    for metric in metric_list:
        plt.figure(figsize=(7.2, 4.6))
        sns.lineplot(
            data=df,
            x="data_hours",
            y=metric,
            hue="annotation_mode",
            marker="o",
            linewidth=2,
        )
        plt.xlabel("Data Scale (hours)")
        plt.ylabel(metric)
        plt.title(f"Data Scale vs {metric}")
        plt.tight_layout()
        save_path = out_dir / f"line_{metric}.png"
        plt.savefig(save_path, dpi=args.dpi)
        plt.close()
        print(f"[DONE] {save_path}")

    # 2) 消融柱状图：每个指标一个子图，比较不同标注粒度
    n = len(metric_list)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.0), squeeze=False)

    for i, metric in enumerate(metric_list):
        ax = axes[0][i]
        sns.barplot(
            data=df,
            x="data_scale",
            y=metric,
            hue="annotation_mode",
            ax=ax,
            errorbar=None,
        )
        ax.set_title(f"Ablation: {metric}")
        ax.set_xlabel("Data Scale")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=0)

    handles, labels = axes[0][0].get_legend_handles_labels()
    for i in range(n):
        if axes[0][i].get_legend() is not None:
            axes[0][i].get_legend().remove()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    bar_path = out_dir / "ablation_bar_metrics.png"
    fig.savefig(bar_path, dpi=args.dpi)
    plt.close(fig)
    print(f"[DONE] {bar_path}")


if __name__ == "__main__":
    main()
