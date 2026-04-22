#!/usr/bin/env bash
set -euo pipefail

# 一键串联示例（请先按你环境修改这些路径）
DATASET_ROOT="/path/to/dataset_root"
TRAIN_MANIFEST="/path/to/train.list"
TEST_MANIFEST="/path/to/test.list"
WORK_ROOT="/path/to/llm_svs_runs"
TRAIN_ENTRY="/path/to/train.py"
INFER_ENTRY="/path/to/infer.py"

EXP_DATA_DIR="$WORK_ROOT/exp_data"
EXP_CFG_DIR="$WORK_ROOT/exp_configs"
EXP_SAVE_DIR="$WORK_ROOT/exp_runs"
EVAL_OUT_DIR="$WORK_ROOT/eval"

mkdir -p "$WORK_ROOT"

# 1) 子集构建
python llm_svs_automation/data_subsetting.py \
  --input_manifest "$TRAIN_MANIFEST" \
  --output_dir "$EXP_DATA_DIR" \
  --hours 1h,5h,20h \
  --seed 3407 \
  --compute_missing_duration \
  --audio_root "$DATASET_ROOT"

# 2) 配置生成
python llm_svs_automation/config_generator.py \
  --template llm_svs_automation/templates/base_config.example.yaml \
  --manifest_root "$EXP_DATA_DIR" \
  --output_dir "$EXP_CFG_DIR" \
  --save_root "$EXP_SAVE_DIR" \
  --scales 1h,5h,20h \
  --modes text_only,phoneme,phoneme_pitch \
  --manifest_name train.list \
  --world_size 8 \
  --nproc_per_node 8

# 3) 训练调度（4组x2卡）
bash llm_svs_automation/train_launcher.sh \
  --exp_table "$EXP_CFG_DIR/experiments.csv" \
  --mode split2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --cmd_template "torchrun --nproc_per_node={ngpu} --master_port={port} $TRAIN_ENTRY --config {config}"

# 4) 评估（注意：你需要保证 experiments.csv 中有 ckpt_path，或 infer.py 不依赖该参数）
python llm_svs_automation/batch_infer_and_eval.py \
  --exp_table "$EXP_CFG_DIR/experiments.csv" \
  --test_manifest "$TEST_MANIFEST" \
  --audio_root "$DATASET_ROOT" \
  --output_dir "$EVAL_OUT_DIR" \
  --infer_cmd_template "python $INFER_ENTRY --config {config} --ckpt {ckpt} --test_list {test_manifest} --out_dir {pred_dir}" \
  --enable_spk_similarity

# 5) 画图
python llm_svs_automation/plot_metrics.py \
  --summary_csv "$EVAL_OUT_DIR/metrics_summary.csv" \
  --output_dir "$EVAL_OUT_DIR/plots"

