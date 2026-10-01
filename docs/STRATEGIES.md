# Strategy library

The 11 actions are option-like macro-actions executed over a fixed forward window.

| Strategy | Behavior |
|---|---|
| Weave | Moving-average-cross trend following |
| Inertia | Multi-horizon momentum trend following |
| Breach | Turtle-style range breakout |
| Flux | Joint price-volume breakout |
| Envelope | ATR volatility-channel breakout |
| Reversion | Bollinger-band mean reversion |
| Correction | RSI overreaction correction |
| Oscillation | KDJ oscillator mean reversion |
| Long | Persistent long directional baseline |
| Short | Persistent short directional baseline; degenerates to Cash when shorting is disabled |
| Cash | No risky exposure during the execution window |

Paper names are used in logs, documentation, and public sample columns. `legacy_name` in `scripts/ldspo_config.py` connects them to the original executor implementation without changing action indices or trading semantics.
