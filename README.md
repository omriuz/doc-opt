<h1 align="center">DocOpt</h1>

<p align="center">
  <img src="assets/doc-opt-logo.png" alt="DocOpt logo" width="260"/>
</p>

<p align="center">
  Document Optimization for Black-Box Retrieval via Reinforcement Learning
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.05087">
    <img src="https://img.shields.io/badge/arXiv-2604.05087-B31B1B.svg" alt="arXiv"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"/>
  </a>
</p>

Official repository for the paper [Document Optimization for Black-Box Retrieval via Reinforcement Learning](https://arxiv.org/abs/2604.05087).

DocOpt optimizes document text to improve retrieval ranking using GRPO reinforcement learning. It supports text and image document collections, multiple retriever types, and any dataset in the expected layout.

**Retriever support**
- **OpenAI** — `text-embedding-3-small`, `text-embedding-3-large`, etc.
- **Local dense** — any Hugging Face embedding model (e.g. `Qwen/Qwen3-Embedding-0.6B`)
- **Multi-vector (ColBERT)** — e.g. `jinaai/jina-colbert-v2`, scored via MaxSim late interaction

**Document type support**
- **Text** — documents are embedded directly and optimized by a causal LM
- **Image** — a vision-language model describes each image; descriptions are embedded and optimized

Multiple retrievers can be listed in a single config; the pipeline runs end-to-end for each.

## Prerequisites

- Python `3.13`
- A GPU for the vLLM and GRPO stages. Development was done using a single H200. A100/H100s should work with reduced batch sizes.
- `OPENAI_API_KEY` when using OpenAI embedding models (not required for local retrievers)

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## Dataset

Datasets are expected in this layout:

**Text datasets**
```
data/<dataset>/texts/*.txt
data/<dataset>/benchmark/benchmark.csv
```

**Image datasets**
```
data/<dataset>/images/*.{jpg,png,webp}
data/<dataset>/benchmark/benchmark.csv
```

`benchmark.csv` has columns `question` and `correct_answer_document_ids` (a JSON-encoded dict of `{doc_id: relevance}`).

To download and convert any MTEB/BEIR retrieval dataset:
```bash
doc-opt prepare-dataset --dataset-id mteb/DS1000Retrieval --output-dir data/DS1000Retrieval
```

## Run

```bash
cp .env.example .env   # add OPENAI_API_KEY if using OpenAI embeddings
doc-opt run --config configs/default.yaml
```

The CLI loads `configs/default.yaml` by default. Switch configs or override individual fields via CLI flags:

```bash
doc-opt run --config configs/freshstack.yaml --embedding-model Qwen/Qwen3-Embedding-0.6B
```

**Reward functions** — set `reward_function` in the config:
- `ranking` — counterfactual NDCG@5 delta (default)
- `dense` — counterfactual similarity-score delta
- `hybrid` — average of ranking and dense

Each run writes a `run_report.json` with per-stage metrics (`ndcg@{1,5,10}`, `recall@{1,5,10}`) for `direct retrieval`, `direct transformation`, `refresh`, and `document optimization`.

## Reproduce

The checked-in default config is `configs/default.yaml`. It writes outputs under `artifacts/ds10k/`.

A sample report is checked in at `artifacts/ds10k/run_report.json`. It captures an optimization run on `DS1000Retrieval` through 600 GRPO steps.

In that run:
- baseline retrieval reaches `ndcg@5 = 0.4669` and `recall@5 = 0.6785`
- direct document transformation reaches `ndcg@5 = 0.4873` and `recall@5 = 0.7217`
- after 600 optimization steps, performance improves to `ndcg@5 = 0.5686` and `recall@5 = 0.8280`

![Sample results chart](artifacts/ds10k/run_report_plot.svg)

_Figure generated from `artifacts/ds10k/run_report.json` using `scripts/plot_run_report.py`._

![Sample step chart](artifacts/ds10k/run_report_steps_plot.svg)

_Figure generated from `artifacts/ds10k/run_report.json` using `scripts/plot_run_report_steps.py`._

## Citation

If you use this repository, please cite:

```bibtex
@misc{uzan2026documentoptimizationblackboxretrieval,
      title={Document Optimization for Black-Box Retrieval via Reinforcement Learning}, 
      author={Omri Uzan and Ron Polonsky and Douwe Kiela and Christopher Potts},
      year={2026},
      eprint={2604.05087},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.05087}, 
}
```
