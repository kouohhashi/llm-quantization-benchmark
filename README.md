# LLM Quantization Benchmark: GGUF vs AWQ for Japanese Business Documents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Benchmarking **GGUF Q4_K_M vs AWQ INT4** for Japanese business document tasks
(meeting summarization & structured JSON extraction) on RTX 4060 8GB VRAM.

> 📄 **Technical Paper**: [日本語ビジネス文書タスクにおけるLLM量子化方式の比較評価](paper/quantization_paper_ja.md)
>
> 著者: 大橋 功 / 株式会社喋ラボ

---

## Key Findings

| Metric | Qwen3-8B GGUF Q4_K_M | Gemma4-E4B GGUF Q4_K_M | AWQ INT4 (vLLM) |
|--------|----------------------|------------------------|-----------------|
| Speed @ 8K | 28.6 tok/s | 41.0 tok/s | ❌ OOM |
| Speed @ 32K | 8.0 tok/s | 35.8 tok/s | ❌ OOM |
| JSON validity | 100% | 100% | — |
| Action item F1 | **0.615** | 0.319 | — |
| VRAM @ 32K | 7.46 GB | 6.49 GB | — |

**AWQ INT4 (vLLM) could not launch on 8GB VRAM** — even at the minimum 8K context.
Model weights occupy 5.71 GiB, leaving insufficient space for the KV cache (requires 1.12 GiB for 8K).

---

## Hardware

- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM)
- CUDA: 12.8 / Driver: 580
- OS: Ubuntu Linux
- Python: 3.11.15

---

## Models Evaluated

| Model | Format | Size | Japanese |
|-------|--------|------|----------|
| Qwen3-8B | GGUF Q4_K_M | ~5.0 GB | ◎ |
| Gemma4-E4B-it | GGUF Q4_K_M | ~5.4 GB | △ |
| Qwen3-8B-AWQ | AWQ INT4 (vLLM) | ~5.7 GB | ❌ (launch failed) |

---

## Tasks

### Task A: Meeting Summarization
Generate a 200–300 character Japanese summary from a meeting transcript.

### Task B: Action Item Extraction (JSON)
Extract structured action items in JSON format:

```json
{
  "action_items": [
    {
      "task": "string",
      "assignee": "string or null",
      "due_date": "YYYY-MM-DD or null",
      "priority": "high | medium | low"
    }
  ]
}
```

---

## Test Data

Synthetic Japanese meeting transcripts (no real data used):

| ID | Type | Tokens | Action Items |
|----|------|--------|--------------|
| T1 | Weekly standup | 2,591 | 7 |
| T2 | Monthly review | 6,515 | 12 |
| T3 | Quarterly planning | 12,681 | 20 |

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── benchmark.py          # Inference + measurement script
├── evaluate.py           # F1, JSON validity, ROUGE-L evaluation
├── run_experiments.sh    # Experiment runner (all conditions)
├── experiment_instructions.md  # Full experiment design doc
├── data/
│   ├── test_data_T1.txt
│   ├── test_data_T2.txt
│   └── test_data_T3.txt
├── ground_truth_T1.json
├── ground_truth_T2.json
├── ground_truth_T3.json
├── results/
│   ├── results.csv           # All 27 trial results (raw)
│   ├── summary_report.csv    # Aggregated results
│   ├── summary_report.md
│   └── awq/
│       └── AWQ_limit_summary.md
├── docs/
│   └── progress.md
└── paper/
    └── quantization_paper_ja.md
```

---

## Reproducing the Experiments

### Setup

```bash
# Create Python 3.11 environment
uv venv --python 3.11
source .venv/bin/activate

# GGUF runtime (CUDA)
pip install llama-cpp-python --extra-index-url \
  https://abetlen.github.io/llama-cpp-python/whl/cu121

# AWQ runtime
pip install vllm

# Evaluation tools
pip install rouge-score pynvml psutil
```

### Download Models

```bash
# GGUF models (via Hugging Face)
huggingface-cli download bartowski/Qwen3-8B-GGUF \
  --include "Qwen3-8B-Q4_K_M.gguf" --local-dir models/gguf/

huggingface-cli download bartowski/gemma-4-e4b-it-GGUF \
  --include "gemma-4-E4B-it-Q4_K_M.gguf" --local-dir models/gguf/
```

### Run GGUF Benchmark

```bash
bash run_experiments.sh
```

### Evaluate

```bash
python evaluate.py --input results/results.csv --output results/summary_report.md
```

---

## AWQ on 8GB VRAM: Failure Analysis

AWQ INT4 requires more KV cache headroom than GGUF.
With `gpu_memory_utilization=0.97`, only **0.73 GiB** remains for KV cache after loading
5.71 GiB of model weights — insufficient for 8K context (needs 1.12 GiB).

| util | KV available | Max context | 8K launch |
|------|-------------|-------------|-----------|
| 0.90 | 0.19 GiB | ~1,392 tok | ❌ |
| 0.93 | 0.42 GiB | ~3,056 tok | ❌ |
| 0.95 | 0.57 GiB | ~4,160 tok | ❌ |
| 0.97 | 0.73 GiB | ~5,280 tok | ❌ |

AWQ itself works fine — a reduced context (4K, util=0.97) launched successfully at 22.67 tok/s.
The limitation is purely hardware memory capacity.

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Citation

```bibtex
@techreport{ohashi2026quantization,
  title     = {日本語ビジネス文書タスクにおけるLLM量子化方式の比較評価:
               GGUF Q4\_K\_M と AWQ INT4 の速度・品質・メモリ効率},
  author    = {大橋 功},
  institution = {株式会社喋ラボ},
  year      = {2026},
  url       = {https://github.com/kouohhashi/llm-quantization-benchmark}
}
```
