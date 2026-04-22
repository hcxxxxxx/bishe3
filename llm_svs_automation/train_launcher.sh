#!/usr/bin/env bash
set -euo pipefail

# 8 卡训练调度器
# - split2: 8 卡分为 4 组（每组2卡），可并行跑4个实验
# - full8: 单实验占满8卡，实验按队列串行

EXP_TABLE=""
MODE="split2"                # split2 | full8
GPUS="0,1,2,3,4,5,6,7"
LOG_DIR="./logs/train"
MASTER_PORT_BASE=29500
STOP_ON_ERROR=0
DRY_RUN=0

# 训练命令模板（可按你的 GPT-SoVITS 入口替换）
# 可用变量: {config} {ngpu} {gpus} {port} {exp_id}
CMD_TEMPLATE='torchrun --nproc_per_node={ngpu} --master_port={port} train.py --config {config}'

usage() {
  cat <<EOF
Usage:
  bash train_launcher.sh --exp_table experiments.csv [options]

Options:
  --exp_table PATH         实验索引表（config_generator.py 生成）
  --mode MODE              split2 或 full8（默认 split2）
  --gpus LIST              GPU 列表，逗号分隔（默认 0,1,2,3,4,5,6,7）
  --log_dir DIR            日志目录（默认 ./logs/train）
  --master_port_base INT   master_port 起始值（默认 29500）
  --cmd_template STRING    训练命令模板
  --stop_on_error 0|1      遇到失败是否立即停止（默认 0）
  --dry_run 0|1            仅打印调度计划，不执行（默认 0）
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp_table) EXP_TABLE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --master_port_base) MASTER_PORT_BASE="$2"; shift 2 ;;
    --cmd_template) CMD_TEMPLATE="$2"; shift 2 ;;
    --stop_on_error) STOP_ON_ERROR="$2"; shift 2 ;;
    --dry_run) DRY_RUN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$EXP_TABLE" ]]; then
  echo "[ERROR] --exp_table 必填"
  usage
  exit 1
fi
if [[ ! -f "$EXP_TABLE" ]]; then
  echo "[ERROR] 找不到实验表: $EXP_TABLE"
  exit 1
fi
if [[ "$MODE" != "split2" && "$MODE" != "full8" ]]; then
  echo "[ERROR] --mode 仅支持 split2/full8"
  exit 1
fi

mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
GPU_COUNT=${#GPU_IDS[@]}
if (( GPU_COUNT < 2 )); then
  echo "[ERROR] 至少需要2张卡"
  exit 1
fi

# 构建 GPU 分组
GPU_GROUPS=()
if [[ "$MODE" == "split2" ]]; then
  if (( GPU_COUNT % 2 != 0 )); then
    echo "[ERROR] split2 模式要求 GPU 数量为偶数"
    exit 1
  fi
  for ((i=0; i<GPU_COUNT; i+=2)); do
    GPU_GROUPS+=("${GPU_IDS[$i]},${GPU_IDS[$i+1]}")
  done
else
  GPU_GROUPS+=("$GPUS")
fi
NUM_SLOTS=${#GPU_GROUPS[@]}

echo "[INFO] 模式: $MODE"
echo "[INFO] GPU 分组: ${GPU_GROUPS[*]}"
echo "[INFO] 槽位数: $NUM_SLOTS"

# 读取实验表
EXP_IDS=()
CFG_PATHS=()
NGPUS=()

while IFS=',' read -r exp_id data_scale annotation_mode train_manifest config_path save_dir ngpu batch_size lr epochs; do
  # 跳过表头/空行
  if [[ "$exp_id" == "exp_id" || -z "$exp_id" ]]; then
    continue
  fi
  EXP_IDS+=("$exp_id")
  CFG_PATHS+=("$config_path")
  if [[ "$MODE" == "split2" ]]; then
    NGPUS+=("2")
  else
    NGPUS+=("$GPU_COUNT")
  fi
done < "$EXP_TABLE"

TOTAL=${#EXP_IDS[@]}
if (( TOTAL == 0 )); then
  echo "[ERROR] 实验表为空: $EXP_TABLE"
  exit 1
fi

echo "[INFO] 实验总数: $TOTAL"

render_cmd() {
  local tmpl="$1"
  local cfg="$2"
  local ngpu="$3"
  local gpus="$4"
  local port="$5"
  local exp_id="$6"

  local cmd="$tmpl"
  cmd="${cmd//\{config\}/$cfg}"
  cmd="${cmd//\{ngpu\}/$ngpu}"
  cmd="${cmd//\{gpus\}/$gpus}"
  cmd="${cmd//\{port\}/$port}"
  cmd="${cmd//\{exp_id\}/$exp_id}"
  echo "$cmd"
}

# 槽位状态
SLOT_PID=()
SLOT_EXP=()
SLOT_LOG=()
for ((i=0; i<NUM_SLOTS; i++)); do
  SLOT_PID+=("0")
  SLOT_EXP+=("")
  SLOT_LOG+=("")
done

next_idx=0
completed=0
failed=0

launch_on_slot() {
  local slot="$1"
  local idx="$2"

  local exp_id="${EXP_IDS[$idx]}"
  local cfg="${CFG_PATHS[$idx]}"
  local ngpu="${NGPUS[$idx]}"
  local gpus="${GPU_GROUPS[$slot]}"
  local port=$((MASTER_PORT_BASE + idx))

  local log_file="$LOG_DIR/${exp_id}_$(date +%Y%m%d_%H%M%S).log"
  local cmd
  cmd=$(render_cmd "$CMD_TEMPLATE" "$cfg" "$ngpu" "$gpus" "$port" "$exp_id")

  echo "[LAUNCH] slot=$slot exp=$exp_id gpus=$gpus port=$port"
  echo "[CMD] $cmd"
  echo "[LOG] $log_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    SLOT_PID[$slot]="-1"
    SLOT_EXP[$slot]="$exp_id"
    SLOT_LOG[$slot]="$log_file"
    return
  fi

  CUDA_VISIBLE_DEVICES="$gpus" bash -lc "$cmd" > "$log_file" 2>&1 &
  SLOT_PID[$slot]="$!"
  SLOT_EXP[$slot]="$exp_id"
  SLOT_LOG[$slot]="$log_file"
}

reap_slot_if_done() {
  local slot="$1"
  local pid="${SLOT_PID[$slot]}"

  if [[ "$pid" == "0" ]]; then
    return
  fi

  # dry-run 直接标记完成
  if [[ "$pid" == "-1" ]]; then
    echo "[DONE] exp=${SLOT_EXP[$slot]} (dry-run)"
    SLOT_PID[$slot]="0"
    SLOT_EXP[$slot]=""
    SLOT_LOG[$slot]=""
    completed=$((completed + 1))
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    return
  fi

  set +e
  wait "$pid"
  local status=$?
  set -e

  if (( status == 0 )); then
    echo "[DONE] exp=${SLOT_EXP[$slot]}"
  else
    echo "[FAIL] exp=${SLOT_EXP[$slot]} status=$status log=${SLOT_LOG[$slot]}"
    failed=$((failed + 1))
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      echo "[STOP] stop_on_error=1，终止其余任务..."
      for p in "${SLOT_PID[@]}"; do
        if [[ "$p" != "0" && "$p" != "-1" ]]; then
          kill "$p" 2>/dev/null || true
        fi
      done
      exit "$status"
    fi
  fi

  SLOT_PID[$slot]="0"
  SLOT_EXP[$slot]=""
  SLOT_LOG[$slot]=""
  completed=$((completed + 1))
}

while (( completed < TOTAL )); do
  # 先回收已完成任务
  for ((slot=0; slot<NUM_SLOTS; slot++)); do
    reap_slot_if_done "$slot"
  done

  # 再在空槽位启动新任务
  for ((slot=0; slot<NUM_SLOTS; slot++)); do
    if (( next_idx >= TOTAL )); then
      break
    fi
    if [[ "${SLOT_PID[$slot]}" == "0" ]]; then
      launch_on_slot "$slot" "$next_idx"
      next_idx=$((next_idx + 1))
    fi
  done

  sleep 3
done

echo "[SUMMARY] total=$TOTAL completed=$completed failed=$failed"
if (( failed > 0 )); then
  exit 1
fi
