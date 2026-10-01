# LD-SPO

Paper-aligned implementation of **LD-SPO: LLM-Distilled Strategy-Level Policy Optimization for Adaptive Financial Trading**.

[中文说明](README_zh-CN.md) · [Method](docs/METHOD.md) · [Strategies](docs/STRATEGIES.md) · [Data](docs/DATA.md)

## Included components

All Python source and experiment entry points are kept in `scripts/`. The release includes:

- a Pre-LN Transformer for the numerical market sequence;
- token and article encoders for aligned financial news;
- learned-query attention pooling for both modalities;
- a learned no-news representation;
- gated numerical-text fusion;
- an 11-strategy policy head and scalar value head;
- action sampling and PPO-compatible action evaluation.
- Stage-1 teacher distillation and value warm-up;
- Stage-2 PPO with stage-aware KL anchoring;
- the strategy-level trading environment and parallel sampler;
- CSI 300 and NASDAQ-100 experiment launchers;
- a deterministic public sample dataset and generator.

Numerical-only ablation models and embedded diagnostic test functions remain excluded so the released model implementation stays focused on the paper's multimodal architecture.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Public sample experiment

```bash
python scripts/run_sample_experiment.py
```

The sample is synthetic and is intended to validate the software path, not reproduce the paper's financial results.

## Full experiments

After preparing licensed datasets described in [docs/DATA.md](docs/DATA.md):

```bash
bash scripts/run_csi300.sh linear
bash scripts/run_nasdaq100.sh linear
```

## Input and output

The model accepts a dictionary containing:

- `numerical`: float tensor shaped `(batch, time, 21)`;
- `news_token_ids`: integer tensor shaped `(batch, articles, tokens)`;
- `news_mask`: boolean tensor shaped `(batch, articles)`.

It returns 11 policy logits and one scalar value estimate per sample.

```python
from scripts.ldspo_model import LDSPOMultimodalStudent

model = LDSPOMultimodalStudent()
policy_logits, state_value = model(observations)
```

## Strategy order

`Weave`, `Inertia`, `Breach`, `Flux`, `Envelope`, `Reversion`, `Correction`, `Oscillation`, `Long`, `Short`, `Cash`.

## License and disclaimer

Released under the [MIT License](LICENSE) for research and educational use. This software does not constitute investment advice.
