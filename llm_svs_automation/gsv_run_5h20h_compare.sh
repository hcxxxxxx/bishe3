#!/usr/bin/env bash
set -Eeuo pipefail

# 只跑 5h/20h（text_only）用于和 1h 快速对照
# - 调用现有 gsv_run_remaining_serial.sh，避免重复维护主流程
# - 默认使用单卡串行，优先保证稳定
#
# 推荐：
# 1) 先用 --profile fast 跑出可对照结果
# 2) 确认趋势后，再改 --profile full 做最终版

usage() {
  cat <<'EOF'
Usage:
  bash gsv_run_5h20h_compare.sh [options]

Options:
  --root PATH         项目根目录（默认 /mnt/workspace/hongchengxun/bishe3）
  --gsv_root PATH     GPT-SoVITS 根目录（默认 /mnt/workspace/hongchengxun/GPT-SoVITS）
  --gpu ID            训练卡号（默认 6）
  --profile NAME      fast 或 full（默认 fast）
  --dry_run 0|1       仅打印不执行（默认 0）

Profiles:
  fast:
    S1: epochs=8,  batch=8
    S2: epochs=30, batch=1, segment=5120
  full:
    S1: epochs=20, batch=8
    S2: epochs=100,batch=2, segment=7680

Outputs:
  - /data/runs/gsv/5h_text_only/
  - /data/runs/gsv/20h_text_only/
  - /data/runs/gsv/finished_weights.csv
EOF
}

ROOT="/mnt/workspace/hongchengxun/bishe3"
GSV_ROOT="/mnt/workspace/hongchengxun/GPT-SoVITS"
GPU_ID="6"
PROFILE="fast"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --gsv_root) GSV_ROOT="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --dry_run) DRY_RUN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1 ;;
  esac
done

AUTO_ROOT="$ROOT/llm_svs_automation"
BASE_SCRIPT="$AUTO_ROOT/gsv_run_remaining_serial.sh"

if [[ ! -x "$BASE_SCRIPT" ]]; then
  echo "[ERROR] 未找到可执行脚本: $BASE_SCRIPT"
  exit 1
fi

case "$PROFILE" in
  fast)
    S1_EPOCHS=8
    S1_BATCH=8
    S2_EPOCHS=30
    S2_BATCH=1
    S2_SEGMENT=5120
    ;;
  full)
    S1_EPOCHS=20
    S1_BATCH=8
    S2_EPOCHS=100
    S2_BATCH=2
    S2_SEGMENT=7680
    ;;
  *)
    echo "[ERROR] --profile 仅支持 fast/full，当前: $PROFILE"
    exit 1
    ;;
esac

# 排除除了 5h_text_only 和 20h_text_only 之外的 7 组
EXCLUDE_LIST="1h_text_only,1h_phoneme,1h_phoneme_pitch,5h_phoneme,5h_phoneme_pitch,20h_phoneme,20h_phoneme_pitch"

echo "[INFO] root=$ROOT"
echo "[INFO] gsv_root=$GSV_ROOT"
echo "[INFO] gpu=$GPU_ID"
echo "[INFO] profile=$PROFILE"
echo "[INFO] dry_run=$DRY_RUN"
echo "[INFO] exclude=$EXCLUDE_LIST"
echo "[INFO] S1: epochs=$S1_EPOCHS batch=$S1_BATCH"
echo "[INFO] S2: epochs=$S2_EPOCHS batch=$S2_BATCH segment=$S2_SEGMENT"

CMD=(
  bash "$BASE_SCRIPT"
  --root "$ROOT"
  --gsv_root "$GSV_ROOT"
  --gpu "$GPU_ID"
  --exclude "$EXCLUDE_LIST"
  --continue_on_error 1
  --dry_run "$DRY_RUN"
  --s1_epochs "$S1_EPOCHS"
  --s1_batch "$S1_BATCH"
  --s2_epochs "$S2_EPOCHS"
  --s2_batch "$S2_BATCH"
  --s2_segment "$S2_SEGMENT"
)

echo "[RUN] ${CMD[*]}"
"${CMD[@]}"

RUNS_ROOT="$ROOT/data/runs/gsv"
echo
echo "[DONE] 训练流程结束。"
echo "[CHECK] 最新权重："
ls -t "$RUNS_ROOT"/5h_text_only/SoVITS_weights_v2/*.pth 2>/dev/null | head -n 1 || true
ls -t "$RUNS_ROOT"/20h_text_only/SoVITS_weights_v2/*.pth 2>/dev/null | head -n 1 || true
echo
echo "[NEXT] 若都产出 .pth，即可按 1h 同样流程做推理+评估对照。"

