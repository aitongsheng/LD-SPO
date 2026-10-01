# Data availability

The repository contains `data/sample/ldspo_sample.csv`, a deterministic synthetic dataset with 360 rows, two symbols, the 21 numerical inputs, aligned news fields, 11 teacher-policy probabilities, and a value target.

The complete CSI 300 and NASDAQ-100 market-news datasets are not redistributed because their market records and news text originate from third-party providers and remain subject to licensing, terms-of-service, and copyright restrictions. The full artifacts are also too large for an ordinary source repository.

The sample preserves the public schema and date semantics needed to exercise loading, multimodal batching, teacher distillation, the trading environment, and PPO integration. It is not suitable for measuring or reproducing financial performance.

Regenerate the sample with:

```bash
python scripts/generate_sample_data.py
```

For full experiments, provide licensed CSV or Parquet data with the same feature, news, and teacher-distribution columns, then set `DATA_PATH` when invoking the shell launchers.
