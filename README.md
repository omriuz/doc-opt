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

**Tested retrievers**
- **OpenAI** — `text-embedding-3-small`, `text-embedding-3-large`
- **Qwen** — `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Embedding-4B`
- **Jina ColBERT** — `jinaai/jina-colbert-v2` (multi-vector, scored via MaxSim late interaction)

**Tested document types**
- **Text** — documents are embedded directly and optimized by a causal LM (tested with `Qwen/Qwen3-4B-Instruct`)
- **Image** — a VLM describes each image; descriptions are embedded and optimized (tested with `Qwen/Qwen3-VL-2B-Instruct` and OpenAI / Qwen retrievers)

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

The default config is `configs/default.yaml` (DS1000Retrieval, `text-embedding-3-small`). Run outputs are written under `artifacts/ds1000/text-embedding-3-small/`.

A sample result on DS1000Retrieval with `text-embedding-3-small` over 600 GRPO steps:

| Stage | NDCG@5 | Recall@5 |
|---|---|---|
| Baseline retrieval | 0.4669 | 0.6785 |
| Direct transformation | 0.4873 | 0.7217 |
| After optimization | 0.5686 | 0.8280 |

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
