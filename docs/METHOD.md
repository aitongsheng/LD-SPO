# Method summary

## Multimodal state

At decision time `t`, the state is `s_t = (X_t, N_t)`. `X_t` is a `T × F` numerical lookback sequence and `N_t` is the set of eligible stock-, sector-, and market-level news published no later than the decision cutoff.

The default paper setting uses `T=30`, `F=21`, and news from the preceding three days.

## Student architecture

1. A linear projection, LayerNorm, and GeLU map numerical observations to dimension `d_model`.
2. A Pre-LN Transformer encodes the temporal sequence.
3. Learnable-query attention pooling produces the numerical vector.
4. A lightweight token encoder maps each news item to an article vector.
5. Article attention pooling produces the textual vector; missing news uses a learned embedding.
6. A sigmoid gate controls the textual contribution before LayerNorm.
7. Two-layer policy and value heads output an 11-way categorical policy and scalar value.

## Stage 1

The student minimizes forward KL from the teacher distribution to the student policy. The value head is initially frozen, then warmed up against teacher-weighted execution-window rewards with a reduced learning rate.

## Stage 2

PPO optimizes completed non-overlapping `H`-day transitions. The current student is regularized toward a frozen copy of its Stage-1 initialization. The KL coefficient is strong early and anneals toward its final value so that reward feedback progressively takes control.

## Evaluation

- **Greedy:** execute the highest-probability strategy.
- **Top-3 fusion:** allocate capital among the three highest-probability strategies in proportion to their normalized policy probabilities.

No online LLM query is required at deployment.
