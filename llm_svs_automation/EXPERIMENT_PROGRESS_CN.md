# LLM-SVS 实验进展记录（截至 2026-04-23）

本文档记录当前已完成工作、成功命令、关键结果与后续计划。  
服务器路径基准：`/mnt/workspace/hongchengxun/bishe3`

## 1. 环境与路径（已确认）

- GPU: `8 x RTX 4090 D`
- Driver/CUDA: `570.133.07 / CUDA 12.8`
- 训练主环境: `GPTSoVits`
- 自动化/评估环境: `llm_svs`

常用变量：

```bash
export ROOT=/mnt/workspace/hongchengxun/bishe3
export DATA=$ROOT/data
export AUTO=$ROOT/llm_svs_automation
export GSV=/mnt/workspace/hongchengxun/GPT-SoVITS
export EVAL=$DATA/eval/1h_text_only
```

---

## 2. 数据与子集（已完成）

### 2.1 生成 M4Singer 统一 manifest（已完成）
- 路径：`$DATA/manifests/m4singer_manifest.jsonl`

### 2.2 按时长采样 1h/5h/20h（已完成）

成功命令：

```bash
python $AUTO/data_subsetting.py \
  --input_manifest $DATA/manifests/m4singer_manifest.jsonl \
  --output_dir $DATA/subsets/m4singer \
  --hours 1h,5h,20h \
  --seed 3407 \
  --compute_missing_duration \
  --audio_root $DATA/M4Singer/m4singer
```

成功输出（示例）：
- `1h: 718 条`
- `5h: 3541 条`
- `20h: 14058 条`

---

## 3. 配置生成（已完成）

成功命令：

```bash
python $AUTO/config_generator.py \
  --template $AUTO/templates/base_config.example.yaml \
  --manifest_root $DATA/subsets/m4singer \
  --output_dir $DATA/configs/m4singer \
  --save_root $DATA/runs/m4singer \
  --scales 1h,5h,20h \
  --modes text_only,phoneme,phoneme_pitch \
  --manifest_name train.list \
  --out_format json
```

成功输出：
- `9` 份实验配置
- 索引表：`$DATA/configs/m4singer/experiments.csv`

---

## 4. 训练与预处理进展

### 4.1 1h_text_only（已完整跑通）

- 预处理：成功
- S1：成功（`max_epochs=20` 到达）
- S2：成功（`epoch=100` 完成）

权重产物（示例）：
- S1: `$DATA/runs/gsv_3scales/1h_text_only/GPT_weights_v2/*.ckpt`
- S2: `$DATA/runs/gsv_3scales/1h_text_only/SoVITS_weights_v2/*.pth`

### 4.2 5h/20h（进行中）

- 主要问题：共享 GPU 场景下 `CUDA OOM`
- 当前策略：单卡串行、降低 `s2_batch` 与 `segment_size`

---

## 5. 推理与评估（1h 已完成）

### 5.1 权重切换（API）

```bash
curl "http://127.0.0.1:9880/set_gpt_weights?weights_path=$S1_CKPT"
curl "http://127.0.0.1:9880/set_sovits_weights?weights_path=$S2_CKPT"
```

### 5.2 批量推理（已成功，20/20）

```bash
python $AUTO/gsv_batch_infer_api.py \
  --test_manifest $DATA/manifests/m4singer_manifest.jsonl \
  --pred_dir $EVAL/predictions/1h_text_only \
  --api_base http://127.0.0.1:9880 \
  --ref_audio_path "$REF_WAV" \
  --prompt_text "$PROMPT_TEXT" \
  --prompt_lang zh \
  --text_lang zh \
  --max_samples 20 \
  --overwrite 1
```

结果：
- `ok=20, skip=0, fail=0`

### 5.3 指标评估（已成功）

说明：需使用规范化 `utt_id` 的测试清单（`test_20_eval.jsonl`）。

```bash
python $AUTO/batch_infer_and_eval.py \
  --exp_table $EVAL/exp_1h.csv \
  --test_manifest $EVAL/test_20_eval.jsonl \
  --audio_root $DATA/M4Singer/m4singer \
  --output_dir $EVAL \
  --skip_infer
```

结果（20 条）：
- `F0 RMSE = 102.2748`
- `MCD = 767.3510`
- `miss = 0`
- `SPK = NaN`（未启用/不可用 resemblyzer）

---

## 6. 已确认的常见现象与结论

1. `Falling back to ['CPUExecutionProvider'] and retrying.`  
   - 出现在 `step1_get_text`，因为脚本主动设置 `CUDA_VISIBLE_DEVICES=''`，属于正常现象。

2. `torch.load` 报 `weights_only`  
   - PyTorch 2.6+ 默认行为变化，加载旧格式权重时需 `weights_only=False`（可信来源前提）。

3. `Could not load libtorchcodec`  
   - 发生在 API 服务端环境（`GPTSoVits`），不是 `llm_svs` 客户端脚本本身。

---

## 7. 已新增自动化脚本

- `gsv_batch_infer_api.py`  
  批量调用 GPT-SoVITS API `/tts`，输出 `utt_id.wav` 供评估脚本直接读取。

- `gsv_run_5h20h_compare.sh`  
  仅跑 `5h_text_only` 与 `20h_text_only` 的快速对照脚本（`fast/full` 两档）。

---

## 8. 下一步（建议执行顺序）

1. 先用 `gsv_run_5h20h_compare.sh --profile fast` 跑出 5h/20h 可对照结果。  
2. 对 5h/20h 复用 1h 同口径推理与评估流程，汇总到同一 `metrics_summary.csv`。  
3. 画对比图：`plot_metrics.py`。  
4. 若趋势成立，再跑 `--profile full` 形成最终版结果。

