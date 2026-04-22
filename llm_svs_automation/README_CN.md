# LLM-SVS 实验自动化脚本（GPT-SoVITS）

这套脚本用于支持你做“数据规模 x 标注细粒度”实验：
- 数据规模：`1h / 5h / 20h`
- 标注粒度：`text_only / phoneme / phoneme_pitch`

## 1. 建议目录结构

```text
llm_svs_automation/
├── data_subsetting.py               # 数据子集划分 + 标注降级
├── config_generator.py              # 自动生成实验配置（含 DDP 参数）
├── train_launcher.sh                # 8卡调度（split2/full8 + 队列）
├── batch_infer_and_eval.py          # 批量推理 + 指标评估
├── plot_metrics.py                  # 结果可视化
├── run_pipeline_example.sh          # 一键串联示例脚本
├── utils_manifest.py                # manifest 读写工具
├── utils_metrics.py                 # F0/MCD/声纹相似度工具
├── requirements.txt
└── templates/
    ├── base_config.example.yaml
    └── hparam_override.example.json
```

## 2. 环境搭建 Checklist（Conda）

```bash
conda create -n llm_svs python=3.10 -y
conda activate llm_svs

# 按你的 CUDA/PyTorch 版本替换下行（示例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r llm_svs_automation/requirements.txt
```

可选校验：
```bash
python -c "import torch; print(torch.cuda.device_count())"
python -c "import librosa, soundfile, yaml, pandas, seaborn"
```

## 3. 使用流程

### Step A: 数据子集划分 + 标注降级

```bash
python llm_svs_automation/data_subsetting.py \
  --input_manifest /path/to/train.list \
  --output_dir /path/to/exp_data \
  --hours 1h,5h,20h \
  --seed 3407 \
  --compute_missing_duration \
  --audio_root /path/to/dataset_root
```

输出示例：
- `/path/to/exp_data/1h/text_only/train.list`
- `/path/to/exp_data/1h/phoneme/train.list`
- `/path/to/exp_data/1h/phoneme_pitch/train.list`
- 以及 `train.jsonl` 和 `subset_summary.json`

### Step B: 自动生成配置

```bash
python llm_svs_automation/config_generator.py \
  --template llm_svs_automation/templates/base_config.example.yaml \
  --manifest_root /path/to/exp_data \
  --output_dir /path/to/exp_configs \
  --save_root /path/to/exp_runs \
  --scales 1h,5h,20h \
  --modes text_only,phoneme,phoneme_pitch \
  --manifest_name train.list \
  --world_size 8 \
  --nproc_per_node 8
```

会生成：
- 多个配置文件：`/path/to/exp_configs/*.yaml`
- 实验索引表：`/path/to/exp_configs/experiments.csv`

### Step C: 8 卡训练调度

`split2`（4组 x 2卡并行）：
```bash
bash llm_svs_automation/train_launcher.sh \
  --exp_table /path/to/exp_configs/experiments.csv \
  --mode split2 \
  --gpus 0,1,2,3,4,5,6,7 \
  --cmd_template 'torchrun --nproc_per_node={ngpu} --master_port={port} /path/to/train.py --config {config}'
```

`full8`（单实验占满8卡，按队列串行）：
```bash
bash llm_svs_automation/train_launcher.sh \
  --exp_table /path/to/exp_configs/experiments.csv \
  --mode full8 \
  --gpus 0,1,2,3,4,5,6,7 \
  --cmd_template 'torchrun --nproc_per_node={ngpu} --master_port={port} /path/to/train.py --config {config}'
```

### Step D: 批处理推理与指标计算

先在 `experiments.csv` 中补 `ckpt_path`（可选，若推理命令需要）。

```bash
python llm_svs_automation/batch_infer_and_eval.py \
  --exp_table /path/to/exp_configs/experiments.csv \
  --test_manifest /path/to/test.list \
  --audio_root /path/to/dataset_root \
  --output_dir /path/to/eval_out \
  --infer_cmd_template 'python /path/to/infer.py --config {config} --ckpt {ckpt} --test_list {test_manifest} --out_dir {pred_dir}' \
  --enable_spk_similarity
```

输出：
- `metrics_summary.csv`
- `metrics_summary.json`
- `per_utt_metrics.csv`

### Step E: 画图

```bash
python llm_svs_automation/plot_metrics.py \
  --summary_csv /path/to/eval_out/metrics_summary.csv \
  --output_dir /path/to/eval_out/plots
```

## 4. 说明

1. `data_subsetting.py` 使用“同一随机顺序前缀切片”，天然保证 `1h ⊂ 5h ⊂ 20h`。  
2. 指标脚本默认按文件名匹配预测音频：`utt_id.wav` 或 `basename.wav`。  
3. `resemblyzer` 若不可用，不会中断评估，只会输出 `spk_similarity=NaN`。  
4. 若你的 GPT-SoVITS 配置键名与默认不同，可通过 `config_generator.py` 的 `*_key` 参数适配。
