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

This repository contains the current reproduction code for document optimization on the [DS1000Retrieval dataset](https://huggingface.co/datasets/mteb/DS1000Retrieval). The implementation uses OpenAI embeddings for retrieval and a vLLM-backed causal language model for document rewriting and GRPO training.

We are activley working on extending the support of this repo to additional retrievers, datasets and models. Stay tuned!

## Prerequisites

- Python `3.13`
- A GPU for the vLLM and GRPO stages. Development was done using a single H200. A100/H100s should work with reduced batch sizes.
- `OPENAI_API_KEY` for embedding generation

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## Dataset

The reproduction config expects a local dataset root at `data/DS1000Retrieval` with this layout:

- `data/DS1000Retrieval/texts/*.txt`
- `data/DS1000Retrieval/benchmark/benchmark.csv`

You can also run custom experiments with another dataset prepared in the same format.

## Reproduce

```bash
cp .env.example .env
doc-opt run
```

By default, the CLI loads `configs/default.yaml`. The default config writes outputs under `artifacts/reproduce/`. The run report is saved to `artifacts/reproduce/run_report.json` and records the `direct retrieval`, `direct transformation`, `refresh`, and `document optimization` stages with `ndcg@{1,5,10}` and `recall@{1,5,10}` metrics.

A sample report is checked in at `artifacts/reproduce/run_report.json`. It captures a short run after only 50 optimization steps on `DS1000Retrieval`, rather than a full training run.

In that sample run:
- baseline retrieval reaches `ndcg@5 = 0.4776` and `recall@5 = 0.6867`
- direct document transformation reaches `ndcg@5 = 0.4899` and `recall@5 = 0.7317`
- after the first refresh at 50 optimization steps, performance improves to `ndcg@5 = 0.5359` and `recall@5 = 0.7805`

![Sample results chart](artifacts/reproduce/run_report_plot.svg)

_Figure generated from `artifacts/reproduce/run_report.json` using `scripts/plot_run_report.py`._

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
