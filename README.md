# English-Swedish Machine Translation Project

This repository contains experiments for English<->Swedish machine translation using:

- A custom Transformer model trained with PyTorch
- A LoRA-finetuned `tencent/HY-MT1.5-7B` model
- Baseline comparisons against OPUS and GPT-SW3 translators

The code supports training, qualitative translation checks, and metric-based evaluation (BLEU, chrF++, COMET).

## Project Layout

- `mt/train.py` - train a custom seq2seq Transformer with Hydra config.
- `mt/translate.py` - load a trained custom checkpoint and run greedy translation.
- `mt/finetune_model.py` - LoRA finetuning pipeline for `tencent/HY-MT1.5-7B`.
- `mt/eval.py` - metric evaluation across multiple models.
- `mt/create_samples.py` - manual sample translations for quick quality checks.
- `mt/test_finetune.py` - sanity tests for finetuned adapters and baselines.
- `run.sh`, `train_finetune.sh`, `eval.sh`, `translate.sh`, `test_model.sh` - SLURM helper scripts.
- `data/tokenizer/` - SentencePiece tokenizer assets for the custom model.
- `outputs/` - Hydra run outputs and logs from previous runs.

## Requirements

- Python 3.10+ recommended
- CUDA GPU recommended (code falls back to CPU where possible)
- Access to Hugging Face model downloads
- (Optional) SLURM for cluster execution using included shell scripts

Main Python packages used by the code:

- `torch`
- `transformers`
- `datasets`
- `hydra-core`
- `omegaconf`
- `sentencepiece`
- `peft`
- `trl`
- `evaluate`
- `tqdm`

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers datasets hydra-core omegaconf sentencepiece peft trl evaluate tqdm numpy
```

If you need private model access:

```bash
huggingface-cli login
```

## Configuration (Hydra)

`mt/train.py`, `mt/translate.py`, and `mt/eval.py` expect a Hydra config at:

- `conf/config.yaml`

The repository currently contains run-time snapshots under `outputs/*/.hydra/config.yaml`, which can be used as a template. A typical config includes fields like:

- dataset: `hf_name`, `hf_config`, `src_lang`, `tgt_lang`
- tokenizer: `sp_model_path`, token IDs, `vocab_size`, `max_len`
- training: `batch_size`, `epochs`, `lr`, scheduler and regularization fields
- architecture: `d_model`, `nhead`, encoder/decoder layers, `ffn_dim`
- paths: `save_dir`, `log_dir`

## Running

### 1) Train custom Transformer

Local:

```bash
python -u mt/train.py
```

SLURM:

```bash
sbatch run.sh
```

Checkpoints are saved to the configured `save_dir` (for example `checkpoints/en-sv/`).

### 2) Run greedy translation with custom model

Local:

```bash
python -u mt/translate.py
```

SLURM:

```bash
sbatch translate.sh
```

### 3) Finetune Tencent model with LoRA

Local:

```bash
python -u mt/finetune_model.py
```

SLURM:

```bash
sbatch train_finetune.sh
```

By default, adapter outputs are written to `./tencent_opus_finetune_16r`.

### 4) Evaluate models (BLEU, chrF++, COMET)

Local:

```bash
python -u mt/eval.py
```

SLURM:

```bash
sbatch eval.sh
```

Evaluation artifacts are serialized under `eval/`.

### 5) Manual sample checks

```bash
python -u mt/create_samples.py
```

or (SLURM):

```bash
sbatch test_model.sh
```

## Notes

- Some scripts use hardcoded paths (for example adapter checkpoints). Update constants in the scripts to match your environment.
- Several shell scripts assume specific local environments (`conda activate theenv` or `~/.envs/en-sv-mt`). Adjust these before running.
- `outputs/` contains historical run logs and Hydra snapshots; you can keep or clean it as needed.

## Quick Start Checklist

1. Install dependencies in a clean environment.
2. Create `conf/config.yaml` (or copy from `outputs/*/.hydra/config.yaml`).
3. Verify tokenizer path (`sp_model_path`) and output directories.
4. Run `python -u mt/train.py` or `sbatch run.sh`.
5. Run `python -u mt/eval.py` after training/finetuning.
