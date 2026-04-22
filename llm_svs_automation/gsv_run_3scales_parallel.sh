#!/usr/bin/env bash
set -Eeuo pipefail

# 仅按数据规模(1h/5h/20h)并行训练脚本
# - 默认统一使用 text_only（不区分标注粒度）
# - 默认使用 GPU 0,3,6 分别跑 1h,5h,20h
# - 每个规模流程：构建 train_gpt.list -> 预处理(1/2/3) -> S1 -> S2
#
# 用法示例：
# bash llm_svs_automation/gsv_run_3scales_parallel.sh \
#   --root /mnt/workspace/hongchengxun/bishe3 \
#   --gsv_root /mnt/workspace/hongchengxun/GPT-SoVITS \
#   --mode text_only \
#   --gpus 0,3,6

usage() {
  cat <<'EOF'
Usage:
  bash gsv_run_3scales_parallel.sh [options]

Options:
  --root PATH           实验仓库根目录（默认 /mnt/workspace/hongchengxun/bishe3）
  --gsv_root PATH       GPT-SoVITS 根目录（默认 /mnt/workspace/hongchengxun/GPT-SoVITS）
  --mode NAME           标注目录名（默认 phoneme_pitch；可改 text_only/phoneme）
  --gpus LIST           3张卡，逗号分隔（默认 0,3,6）
  --scales LIST         3个规模，逗号分隔（默认 1h,5h,20h）
  --skip_prepare 0|1    是否跳过预处理（默认 0）
  --skip_s1 0|1         是否跳过 S1（默认 0）
  --skip_s2 0|1         是否跳过 S2（默认 0）
  --dry_run 0|1         仅打印不执行（默认 0）

  --s1_epochs N         S1 epoch（默认 20）
  --s1_batch N          S1 batch（默认 8）
  --s2_epochs N         S2 epoch（默认 100）
  --s2_batch N          S2 batch（默认 4）
  --s2_segment N        S2 segment_size（默认 10240）

Outputs:
  - data/runs/gsv_3scales/<scale>_<mode>/
  - data/runs/gsv_3scales/_batch_logs/
EOF
}

ROOT="/mnt/workspace/hongchengxun/bishe3"
GSV_ROOT="/mnt/workspace/hongchengxun/GPT-SoVITS"
MODE="text_only"
GPUS="0,3,6"
SCALES="1h,5h,20h"
SKIP_PREPARE="0"
SKIP_S1="0"
SKIP_S2="0"
DRY_RUN="0"

S1_EPOCHS="20"
S1_BATCH="8"
S2_EPOCHS="100"
S2_BATCH="4"
S2_SEGMENT="10240"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --gsv_root) GSV_ROOT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --scales) SCALES="$2"; shift 2 ;;
    --skip_prepare) SKIP_PREPARE="$2"; shift 2 ;;
    --skip_s1) SKIP_S1="$2"; shift 2 ;;
    --skip_s2) SKIP_S2="$2"; shift 2 ;;
    --dry_run) DRY_RUN="$2"; shift 2 ;;
    --s1_epochs) S1_EPOCHS="$2"; shift 2 ;;
    --s1_batch) S1_BATCH="$2"; shift 2 ;;
    --s2_epochs) S2_EPOCHS="$2"; shift 2 ;;
    --s2_batch) S2_BATCH="$2"; shift 2 ;;
    --s2_segment) S2_SEGMENT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1 ;;
  esac
done

DATA_ROOT="$ROOT/data"
SUBSET_ROOT="$DATA_ROOT/subsets/m4singer"
RUNS_ROOT="$DATA_ROOT/runs/gsv_3scales"
AUTO_ROOT="$ROOT/llm_svs_automation"
PREP_SCRIPT="$AUTO_ROOT/gsv_prepare_serial.sh"
LOG_ROOT="$RUNS_ROOT/_batch_logs"
mkdir -p "$RUNS_ROOT" "$LOG_ROOT"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_ROOT/run_$RUN_TS.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

run_cmd() {
  local cmd="$*"
  log "[CMD] $cmd"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  eval "$cmd"
}

if [[ ! -d "$ROOT" ]]; then
  echo "[ERROR] ROOT not found: $ROOT"
  exit 1
fi
if [[ ! -d "$GSV_ROOT" ]]; then
  echo "[ERROR] GSV_ROOT not found: $GSV_ROOT"
  exit 1
fi
if [[ ! -x "$PREP_SCRIPT" ]]; then
  echo "[ERROR] PREP_SCRIPT not executable: $PREP_SCRIPT"
  exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
IFS=',' read -r -a SCALE_ARR <<< "$SCALES"
if [[ ${#GPU_ARR[@]} -ne 3 ]]; then
  echo "[ERROR] --gpus 必须给3张卡，例如 0,3,6"
  exit 1
fi
if [[ ${#SCALE_ARR[@]} -ne 3 ]]; then
  echo "[ERROR] --scales 必须给3个规模，例如 1h,5h,20h"
  exit 1
fi

export PYTHONPATH="$GSV_ROOT:$GSV_ROOT/GPT_SoVITS:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

prepare_train_list() {
  local subset_json="$1"
  local exp_dir="$2"

  SUBSET_JSON="$subset_json" EXP_DIR="$exp_dir" DATA_ROOT="$DATA_ROOT" python - <<'PY'
import json, os, shutil
from pathlib import Path

subset_json = Path(os.environ["SUBSET_JSON"])
exp_dir = Path(os.environ["EXP_DIR"])
data_root = Path(os.environ["DATA_ROOT"])
audio_root = data_root / "M4Singer" / "m4singer"

wav_dir = exp_dir / "wavs"
wav_dir.mkdir(parents=True, exist_ok=True)
out = exp_dir / "train_gpt.list"

count = 0
with subset_json.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
    for line in f:
        r = json.loads(line)
        src = Path(r["wav_path"])
        if not src.is_absolute():
            src = audio_root / src
        if not src.exists():
            continue
        utt = str(r.get("utt_id") or src.stem).replace("/", "_").replace("#", "_")
        dst = wav_dir / f"{utt}.wav"
        if not dst.exists():
            try:
                os.symlink(src, dst)
            except Exception:
                shutil.copy2(src, dst)
        text = str(r.get("text") or "").strip()
        if not text:
            continue
        spk = str(r.get("speaker") or "m4singer")
        w.write(f"{dst.name}|{spk}|zh|{text}\n")
        count += 1

print(f"train_gpt.list: {out}")
print(f"samples: {count}")
PY
}

build_tmp_configs() {
  local exp_id="$1"
  local exp_dir="$2"
  local gpu_logic="0"

  EXP_ID="$exp_id" EXP_DIR="$exp_dir" GSV_ROOT="$GSV_ROOT" \
  S1_EPOCHS="$S1_EPOCHS" S1_BATCH="$S1_BATCH" \
  S2_EPOCHS="$S2_EPOCHS" S2_BATCH="$S2_BATCH" S2_SEGMENT="$S2_SEGMENT" \
  GPU_LOGIC="$gpu_logic" \
  python - <<'PY'
import json, os, yaml
from pathlib import Path

exp_id = os.environ["EXP_ID"]
exp_dir = Path(os.environ["EXP_DIR"])
gsv = Path(os.environ["GSV_ROOT"])
gpu_logic = os.environ["GPU_LOGIC"]

s1_epochs = int(os.environ["S1_EPOCHS"])
s1_batch = int(os.environ["S1_BATCH"])
s2_epochs = int(os.environ["S2_EPOCHS"])
s2_batch = int(os.environ["S2_BATCH"])
s2_segment = int(os.environ["S2_SEGMENT"])

# S1
s1 = yaml.safe_load((gsv / "GPT_SoVITS/configs/s1longer-v2.yaml").read_text(encoding="utf-8"))
s1.setdefault("train", {})
s1["train"]["exp_name"] = exp_id
s1["train"]["epochs"] = s1_epochs
s1["train"]["batch_size"] = s1_batch
s1["train"]["save_every_n_epoch"] = 2
s1["train"]["if_save_latest"] = True
s1["train"]["if_save_every_weights"] = True
s1["train"]["if_dpo"] = False
s1["train"]["half_weights_save_dir"] = str(exp_dir / "GPT_weights_v2")
s1["train_semantic_path"] = str(exp_dir / "6-name2semantic.tsv")
s1["train_phoneme_path"] = str(exp_dir / "2-name2text.txt")
s1["output_dir"] = str(exp_dir / "logs_s1_v2")
s1_pre = gsv / "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
if s1_pre.exists():
    s1["pretrained_s1"] = str(s1_pre)
(exp_dir / "GPT_weights_v2").mkdir(parents=True, exist_ok=True)
(exp_dir / "logs_s1_v2").mkdir(parents=True, exist_ok=True)
(exp_dir / "tmp_s1.yaml").write_text(yaml.safe_dump(s1, allow_unicode=True, sort_keys=False), encoding="utf-8")

# S2
s2 = json.loads((gsv / "GPT_SoVITS/configs/s2.json").read_text(encoding="utf-8"))
s2.setdefault("train", {})
s2.setdefault("data", {})
s2.setdefault("model", {})
s2["name"] = exp_id
s2["version"] = "v2"
s2["model"]["version"] = "v2"
s2["data"]["exp_dir"] = str(exp_dir)
s2["s2_ckpt_dir"] = str(exp_dir / "logs_s2_v2")
s2["save_weight_dir"] = str(exp_dir / "SoVITS_weights_v2")
s2["train"]["epochs"] = s2_epochs
s2["train"]["batch_size"] = s2_batch
s2["train"]["segment_size"] = s2_segment
s2["train"]["grad_ckpt"] = True
s2["train"]["save_every_epoch"] = 2
s2["train"]["if_save_latest"] = True
s2["train"]["if_save_every_weights"] = True
s2["train"]["gpu_numbers"] = gpu_logic

s2_g = gsv / "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
s2_d = gsv / "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
if s2_g.exists():
    s2["train"]["pretrained_s2G"] = str(s2_g)
if s2_d.exists():
    s2["train"]["pretrained_s2D"] = str(s2_d)

(exp_dir / "logs_s2_v2").mkdir(parents=True, exist_ok=True)
(exp_dir / "SoVITS_weights_v2").mkdir(parents=True, exist_ok=True)
(exp_dir / "tmp_s2.json").write_text(json.dumps(s2, ensure_ascii=False, indent=2), encoding="utf-8")
print("written:", exp_dir / "tmp_s1.yaml")
print("written:", exp_dir / "tmp_s2.json")
PY
}

run_scale_job() {
  local scale="$1"
  local gpu="$2"
  local exp_id="${scale}_${MODE}"
  local exp_dir="$RUNS_ROOT/$exp_id"
  local job_log="$exp_dir/logs/job_${RUN_TS}.log"
  local subset_json="$SUBSET_ROOT/$scale/$MODE/train.jsonl"

  mkdir -p "$exp_dir/logs"
  {
    echo "[$(date '+%F %T')] [START] exp=$exp_id gpu=$gpu"
    echo "[$(date '+%F %T')] [INFO] subset=$subset_json"
    if [[ ! -f "$subset_json" ]]; then
      echo "[$(date '+%F %T')] [ERROR] missing subset: $subset_json"
      return 1
    fi

    prepare_train_list "$subset_json" "$exp_dir"

    if [[ "$SKIP_PREPARE" != "1" ]]; then
      bash "$PREP_SCRIPT" \
        --gsv_root "$GSV_ROOT" \
        --exp_dir "$exp_dir" \
        --gpu "$gpu" \
        --resume 1 \
        --is_half 1
    else
      echo "[$(date '+%F %T')] [SKIP] prepare"
    fi

    build_tmp_configs "$exp_id" "$exp_dir"

    if [[ "$SKIP_S1" != "1" ]]; then
      (cd "$GSV_ROOT" && CUDA_VISIBLE_DEVICES="$gpu" python -u GPT_SoVITS/s1_train.py -c "$exp_dir/tmp_s1.yaml")
    else
      echo "[$(date '+%F %T')] [SKIP] s1"
    fi

    if [[ "$SKIP_S2" != "1" ]]; then
      (cd "$GSV_ROOT" && CUDA_VISIBLE_DEVICES="$gpu" python -u GPT_SoVITS/s2_train.py --config "$exp_dir/tmp_s2.json")
    else
      echo "[$(date '+%F %T')] [SKIP] s2"
    fi

    local s1_ckpt=""
    local s2_ckpt=""
    s1_ckpt="$(ls -t "$exp_dir"/GPT_weights_v2/*.ckpt 2>/dev/null | head -n 1 || true)"
    s2_ckpt="$(ls -t "$exp_dir"/SoVITS_weights_v2/*.pth 2>/dev/null | head -n 1 || true)"
    echo "[$(date '+%F %T')] [DONE] exp=$exp_id"
    echo "[$(date '+%F %T')] [CKPT] S1=${s1_ckpt:-N/A}"
    echo "[$(date '+%F %T')] [CKPT] S2=${s2_ckpt:-N/A}"
  } 2>&1 | tee -a "$job_log"
}

log "===== 3规模并行任务启动 ====="
log "ROOT=$ROOT"
log "GSV_ROOT=$GSV_ROOT"
log "MODE=$MODE"
log "SCALES=$SCALES"
log "GPUS=$GPUS"
log "MASTER_LOG=$MASTER_LOG"

PIDS=()
for idx in 0 1 2; do
  scale="${SCALE_ARR[$idx]}"
  gpu="${GPU_ARR[$idx]}"
  exp_id="${scale}_${MODE}"
  log "[LAUNCH] exp=$exp_id gpu=$gpu"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY_RUN] skip actual launch for $exp_id"
    continue
  fi
  run_scale_job "$scale" "$gpu" &
  PIDS+=("$!")
done

FAIL=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then
    FAIL=1
  fi
done

if [[ "$FAIL" == "1" ]]; then
  log "[FAIL] 至少一个规模任务失败"
  exit 1
fi

log "[DONE] 三个规模任务全部完成"
