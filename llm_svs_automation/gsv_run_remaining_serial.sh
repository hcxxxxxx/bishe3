#!/usr/bin/env bash
set -Eeuo pipefail

# 一键串行跑 M4Singer 9组实验（默认自动跳过已完成组）
# 流程：构建 train_gpt.list -> gsv_prepare_serial.sh -> S1 -> S2
#
# 说明：
# - 需要在 GPTSoVits 环境中执行
# - 默认单卡串行（--gpu 0）
# - 已完成判定：<exp_dir>/SoVITS_weights_v2/*.pth 存在
#
# 示例：
# bash llm_svs_automation/gsv_run_remaining_serial.sh \
#   --root /mnt/workspace/hongchengxun/bishe3 \
#   --gsv_root /mnt/workspace/hongchengxun/GPT-SoVITS \
#   --gpu 0

usage() {
  cat <<'EOF'
Usage:
  bash gsv_run_remaining_serial.sh [options]

Options:
  --root PATH               实验仓库根目录（默认 /mnt/workspace/hongchengxun/bishe3）
  --gsv_root PATH           GPT-SoVITS 根目录（默认 /mnt/workspace/hongchengxun/GPT-SoVITS）
  --gpu ID                  使用哪张卡（默认 0）
  --exclude CSV             额外排除实验，如 1h_text_only,5h_text_only
  --continue_on_error 0|1   失败后是否继续后续实验（默认 0）
  --dry_run 0|1             仅打印不执行（默认 0）

  --s1_epochs N             S1 训练轮数（默认 20）
  --s1_batch N              S1 batch（默认 8）
  --s2_epochs N             S2 训练轮数（默认 100）
  --s2_batch N              S2 batch（默认 4）
  --s2_segment N            S2 segment_size（默认 10240）

Default paths resolved from --root:
  data/subsets/m4singer/<scale>/<mode>/train.jsonl
  data/runs/gsv/<exp_id>/
  llm_svs_automation/gsv_prepare_serial.sh
EOF
}

ROOT="/mnt/workspace/hongchengxun/bishe3"
GSV_ROOT="/mnt/workspace/hongchengxun/GPT-SoVITS"
GPU_ID="0"
EXCLUDE_CSV=""
CONTINUE_ON_ERROR="0"
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
    --gpu) GPU_ID="$2"; shift 2 ;;
    --exclude) EXCLUDE_CSV="$2"; shift 2 ;;
    --continue_on_error) CONTINUE_ON_ERROR="$2"; shift 2 ;;
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
RUNS_ROOT="$DATA_ROOT/runs/gsv"
AUTO_ROOT="$ROOT/llm_svs_automation"
PREP_SCRIPT="$AUTO_ROOT/gsv_prepare_serial.sh"
GLOBAL_LOG_DIR="$RUNS_ROOT/_batch_logs"
mkdir -p "$GLOBAL_LOG_DIR" "$RUNS_ROOT"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$GLOBAL_LOG_DIR/run_$RUN_TS.log"
SUMMARY_CSV="$RUNS_ROOT/finished_weights.csv"

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
  echo "[ERROR] ROOT 不存在: $ROOT"
  exit 1
fi
if [[ ! -d "$GSV_ROOT" ]]; then
  echo "[ERROR] GSV_ROOT 不存在: $GSV_ROOT"
  exit 1
fi
if [[ ! -x "$PREP_SCRIPT" ]]; then
  echo "[ERROR] 预处理脚本不可执行: $PREP_SCRIPT"
  exit 1
fi

export PYTHONPATH="$GSV_ROOT:$GSV_ROOT/GPT_SoVITS:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "exp_id,s1_ckpt,s2_ckpt,status,updated_at" > "$SUMMARY_CSV"
fi

should_exclude() {
  local exp="$1"
  if [[ -z "$EXCLUDE_CSV" ]]; then
    return 1
  fi
  [[ ",$EXCLUDE_CSV," == *",$exp,"* ]]
}

is_done() {
  local exp="$1"
  local exp_dir="$RUNS_ROOT/$exp"
  compgen -G "$exp_dir/SoVITS_weights_v2/*.pth" > /dev/null
}

append_summary() {
  local exp="$1"
  local s1="$2"
  local s2="$3"
  local status="$4"
  echo "$exp,$s1,$s2,$status,$(date '+%F %T')" >> "$SUMMARY_CSV"
}

prepare_train_list() {
  local exp="$1"
  local scale="$2"
  local mode="$3"
  local exp_dir="$4"

  local subset_json="$SUBSET_ROOT/$scale/$mode/train.jsonl"
  if [[ ! -f "$subset_json" ]]; then
    log "[ERROR] 缺少子集文件: $subset_json"
    return 1
  fi

  mkdir -p "$exp_dir/wavs" "$exp_dir/logs"
  log "[INFO] 生成 train_gpt.list: $exp"
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
  local exp="$1"
  local exp_dir="$2"
  local s1_yaml="$exp_dir/tmp_s1.yaml"
  local s2_json="$exp_dir/tmp_s2.json"

  EXP_ID="$exp" EXP_DIR="$exp_dir" GSV_ROOT="$GSV_ROOT" \
  S1_EPOCHS="$S1_EPOCHS" S1_BATCH="$S1_BATCH" \
  S2_EPOCHS="$S2_EPOCHS" S2_BATCH="$S2_BATCH" S2_SEGMENT="$S2_SEGMENT" \
  python - <<'PY'
import json, os, yaml
from pathlib import Path

exp_id = os.environ["EXP_ID"]
exp_dir = Path(os.environ["EXP_DIR"])
gsv = Path(os.environ["GSV_ROOT"])

s1_epochs = int(os.environ["S1_EPOCHS"])
s1_batch = int(os.environ["S1_BATCH"])
s2_epochs = int(os.environ["S2_EPOCHS"])
s2_batch = int(os.environ["S2_BATCH"])
s2_segment = int(os.environ["S2_SEGMENT"])

# -------- S1 --------
s1_tpl = gsv / "GPT_SoVITS/configs/s1longer-v2.yaml"
s1 = yaml.safe_load(s1_tpl.read_text(encoding="utf-8"))
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
(exp_dir / "tmp_s1.yaml").write_text(
    yaml.safe_dump(s1, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)

# -------- S2 --------
s2_tpl = gsv / "GPT_SoVITS/configs/s2.json"
s2 = json.loads(s2_tpl.read_text(encoding="utf-8"))
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
s2["train"]["gpu_numbers"] = "0"

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

  [[ -f "$s1_yaml" && -f "$s2_json" ]]
}

run_one_exp() {
  local exp="$1"
  local scale="${exp%%_*}"
  local mode="${exp#${scale}_}"
  local exp_dir="$RUNS_ROOT/$exp"
  mkdir -p "$exp_dir/logs"

  log "=============================="
  log "[START] $exp (scale=$scale mode=$mode)"

  prepare_train_list "$exp" "$scale" "$mode" "$exp_dir"

  run_cmd "bash \"$PREP_SCRIPT\" \
    --gsv_root \"$GSV_ROOT\" \
    --exp_dir \"$exp_dir\" \
    --gpu \"$GPU_ID\" \
    --resume 1 \
    --is_half 1"

  build_tmp_configs "$exp" "$exp_dir"

  run_cmd "cd \"$GSV_ROOT\" && CUDA_VISIBLE_DEVICES=\"$GPU_ID\" python -u GPT_SoVITS/s1_train.py \
    -c \"$exp_dir/tmp_s1.yaml\" 2>&1 | tee \"$exp_dir/logs/s1_train.log\""

  run_cmd "cd \"$GSV_ROOT\" && CUDA_VISIBLE_DEVICES=\"$GPU_ID\" python -u GPT_SoVITS/s2_train.py \
    --config \"$exp_dir/tmp_s2.json\" 2>&1 | tee \"$exp_dir/logs/s2_train.log\""

  local s1_ckpt=""
  local s2_ckpt=""
  s1_ckpt="$(ls -t "$exp_dir"/GPT_weights_v2/*.ckpt 2>/dev/null | head -n 1 || true)"
  s2_ckpt="$(ls -t "$exp_dir"/SoVITS_weights_v2/*.pth 2>/dev/null | head -n 1 || true)"

  append_summary "$exp" "$s1_ckpt" "$s2_ckpt" "success"
  log "[DONE] $exp"
  log "[CKPT] S1: ${s1_ckpt:-N/A}"
  log "[CKPT] S2: ${s2_ckpt:-N/A}"
}

EXPERIMENTS=(
  "1h_text_only"
  "1h_phoneme"
  "1h_phoneme_pitch"
  "5h_text_only"
  "5h_phoneme"
  "5h_phoneme_pitch"
  "20h_text_only"
  "20h_phoneme"
  "20h_phoneme_pitch"
)

log "===== 批量串行训练开始 ====="
log "ROOT=$ROOT"
log "GSV_ROOT=$GSV_ROOT"
log "GPU_ID=$GPU_ID"
log "SUMMARY_CSV=$SUMMARY_CSV"
log "MASTER_LOG=$MASTER_LOG"

for exp in "${EXPERIMENTS[@]}"; do
  if should_exclude "$exp"; then
    log "[SKIP] 在 exclude 列表中: $exp"
    continue
  fi
  if is_done "$exp"; then
    log "[SKIP] 已完成(检测到 SoVITS 权重): $exp"
    continue
  fi

  if ! run_one_exp "$exp"; then
    append_summary "$exp" "" "" "failed"
    log "[FAIL] $exp"
    if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
      log "[CONTINUE] continue_on_error=1，继续下一个实验"
      continue
    else
      log "[STOP] continue_on_error=0，停止"
      exit 1
    fi
  fi
done

log "===== 批量任务结束 ====="
log "结果汇总: $SUMMARY_CSV"
