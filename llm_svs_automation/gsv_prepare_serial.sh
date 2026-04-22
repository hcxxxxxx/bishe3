#!/usr/bin/env bash
set -Eeuo pipefail

# GPT-SoVITS 数据预处理（串行稳健版）
# 功能：
# 1) 串行运行 1-get-text.py / 2-get-hubert-wav32k.py / 3-get-semantic.py
# 2) 每一步单独日志 + 总日志，失败自动退出
# 3) 生成 done 标记，支持 resume（默认开启）
#
# 典型用法：
# bash llm_svs_automation/gsv_prepare_serial.sh \
#   --gsv_root /mnt/workspace/hongchengxun/GPT-SoVITS \
#   --exp_dir /mnt/workspace/hongchengxun/bishe3/data/runs/gsv/1h_text_only \
#   --gpu 0

usage() {
  cat <<'EOF'
Usage:
  bash gsv_prepare_serial.sh --gsv_root PATH --exp_dir PATH [options]

Required:
  --gsv_root PATH        GPT-SoVITS 仓库根目录（例如 /mnt/workspace/.../GPT-SoVITS）
  --exp_dir PATH         实验目录（需包含 train_gpt.list 与 wavs/）

Options:
  --gpu ID               步骤2/3使用的 GPU（默认 0）
  --steps LIST           运行步骤，逗号分隔：1,2,3（默认 1,2,3）
  --resume 0|1           是否跳过已完成步骤（默认 1）
  --is_half 0|1          步骤2/3的 is_half（默认 1）
  --force_clean 0|1      是否清理旧中间文件（默认 0）
  --log_root PATH        日志根目录（默认 <exp_dir>/logs）
  --dry_run 0|1          仅打印命令不执行（默认 0）
  -h, --help             显示帮助

Inputs expected in <exp_dir>:
  - train_gpt.list
  - wavs/

Outputs:
  - 2-name2text.txt
  - 6-name2semantic.tsv
  - logs/prepare_YYYYmmdd_HHMMSS/*
  - .prep_status/step{1,2,3}.done
EOF
}

GSV_ROOT=""
EXP_DIR=""
GPU_ID="0"
STEPS="1,2,3"
RESUME="1"
IS_HALF="1"
FORCE_CLEAN="0"
LOG_ROOT=""
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gsv_root) GSV_ROOT="$2"; shift 2 ;;
    --exp_dir) EXP_DIR="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --is_half) IS_HALF="$2"; shift 2 ;;
    --force_clean) FORCE_CLEAN="$2"; shift 2 ;;
    --log_root) LOG_ROOT="$2"; shift 2 ;;
    --dry_run) DRY_RUN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$GSV_ROOT" || -z "$EXP_DIR" ]]; then
  echo "[ERROR] --gsv_root 与 --exp_dir 必填"
  usage
  exit 1
fi

if [[ ! -d "$GSV_ROOT" ]]; then
  echo "[ERROR] gsv_root 不存在: $GSV_ROOT"
  exit 1
fi
if [[ ! -f "$EXP_DIR/train_gpt.list" ]]; then
  echo "[ERROR] 缺少输入文件: $EXP_DIR/train_gpt.list"
  exit 1
fi
if [[ ! -d "$EXP_DIR/wavs" ]]; then
  echo "[ERROR] 缺少输入目录: $EXP_DIR/wavs"
  exit 1
fi

if [[ -z "$LOG_ROOT" ]]; then
  LOG_ROOT="$EXP_DIR/logs"
fi
mkdir -p "$LOG_ROOT"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$LOG_ROOT/prepare_$RUN_TS"
mkdir -p "$RUN_LOG_DIR"
ln -sfn "$RUN_LOG_DIR" "$LOG_ROOT/latest_prepare"

STATUS_DIR="$EXP_DIR/.prep_status"
mkdir -p "$STATUS_DIR"

SUMMARY_LOG="$RUN_LOG_DIR/summary.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY_LOG"
}

run_cmd() {
  local name="$1"
  shift
  local cmd="$*"
  local step_log="$RUN_LOG_DIR/${name}.log"
  log "[RUN] $name"
  log "[CMD] $cmd"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  (
    set -o pipefail
    eval "$cmd" 2>&1 | tee -a "$step_log"
  )
}

has_step() {
  local target="$1"
  [[ ",$STEPS," == *",$target,"* ]]
}

maybe_skip_done() {
  local step="$1"
  local done_file="$STATUS_DIR/step${step}.done"
  if [[ "$RESUME" == "1" && -f "$done_file" ]]; then
    log "[SKIP] step${step} 已完成（$done_file）"
    return 0
  fi
  return 1
}

mark_done() {
  local step="$1"
  local done_file="$STATUS_DIR/step${step}.done"
  date '+%F %T' > "$done_file"
  log "[DONE] step${step} -> $done_file"
}

trap 'log "[FAIL] 脚本异常退出，见日志：$RUN_LOG_DIR"' ERR

log "=== GPT-SoVITS 串行预处理启动 ==="
log "GSV_ROOT=$GSV_ROOT"
log "EXP_DIR=$EXP_DIR"
log "GPU_ID=$GPU_ID"
log "STEPS=$STEPS"
log "RESUME=$RESUME IS_HALF=$IS_HALF FORCE_CLEAN=$FORCE_CLEAN"
log "LOG_DIR=$RUN_LOG_DIR"

export inp_text="$EXP_DIR/train_gpt.list"
export inp_wav_dir="$EXP_DIR/wavs"
export exp_name="$(basename "$EXP_DIR")"
export opt_dir="$EXP_DIR"
export bert_pretrained_dir="$GSV_ROOT/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
export cnhubert_base_dir="$GSV_ROOT/GPT_SoVITS/pretrained_models/chinese-hubert-base"
export pretrained_s2G="$GSV_ROOT/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
export s2config_path="$GSV_ROOT/GPT_SoVITS/configs/s2.json"
export PYTHONPATH="$GSV_ROOT/GPT_SoVITS:$GSV_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "$GSV_ROOT"

run_cmd "env_check" "pwd; python -V; which python; nvidia-smi || true"
run_cmd "input_check" "wc -l \"$inp_text\"; ls -ld \"$inp_wav_dir\""

if [[ "$FORCE_CLEAN" == "1" ]]; then
  run_cmd "cleanup" "rm -f \"$opt_dir\"/2-name2text*.txt \"$opt_dir\"/6-name2semantic*.tsv"
fi

# Step 1: text/phoneme 提取（CPU 串行，最稳）
if has_step "1"; then
  if ! maybe_skip_done "1"; then
    export is_half="False"
    run_cmd "step1_get_text" "CUDA_VISIBLE_DEVICES='' i_part=0 all_parts=1 python -u GPT_SoVITS/prepare_datasets/1-get-text.py"
    run_cmd "step1_merge" "test -f \"$opt_dir/2-name2text-0.txt\"; cp \"$opt_dir/2-name2text-0.txt\" \"$opt_dir/2-name2text.txt\"; wc -l \"$opt_dir/2-name2text.txt\""
    mark_done "1"
  fi
fi

# Step 2: hubert + wav32k（单卡串行）
if has_step "2"; then
  if ! maybe_skip_done "2"; then
    export is_half=$([[ "$IS_HALF" == "1" ]] && echo "True" || echo "False")
    run_cmd "step2_get_hubert" "CUDA_VISIBLE_DEVICES=$GPU_ID i_part=0 all_parts=1 python -u GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"
    run_cmd "step2_check" "test -d \"$opt_dir/4-cnhubert\"; test -d \"$opt_dir/5-wav32k\"; find \"$opt_dir/4-cnhubert\" -type f | wc -l; find \"$opt_dir/5-wav32k\" -type f | wc -l"
    mark_done "2"
  fi
fi

# Step 3: semantic 提取（单卡串行）
if has_step "3"; then
  if ! maybe_skip_done "3"; then
    export is_half=$([[ "$IS_HALF" == "1" ]] && echo "True" || echo "False")
    run_cmd "step3_get_semantic" "CUDA_VISIBLE_DEVICES=$GPU_ID i_part=0 all_parts=1 python -u GPT_SoVITS/prepare_datasets/3-get-semantic.py"
    run_cmd "step3_merge" "test -f \"$opt_dir/6-name2semantic-0.tsv\"; cp \"$opt_dir/6-name2semantic-0.tsv\" \"$opt_dir/6-name2semantic.tsv\"; wc -l \"$opt_dir/6-name2semantic.tsv\""
    mark_done "3"
  fi
fi

run_cmd "final_check" "ls -lh \"$opt_dir\"/2-name2text.txt \"$opt_dir\"/6-name2semantic.tsv; nvidia-smi || true"
log "=== 全部完成 ==="
log "日志目录: $RUN_LOG_DIR"
