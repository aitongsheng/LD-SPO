from __future__ import annotations
import ast
import hashlib
import json
import re
from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd
from ldspo_config import MAX_NEWS_ITEMS, MAX_NEWS_TOKENS, NEWS_LOOKBACK_DAYS, NEWS_TEXT_COLS, TEXT_VOCAB_SIZE
_TOKEN_RE = re.compile('\\w+|[^\\w\\s]', flags=re.UNICODE)
_SPLIT_RE = re.compile('(?:\\r?\\n)+|\\s*\\|\\|\\s*|\\s*<NEWS>\\s*', flags=re.IGNORECASE)

def _flatten_text(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        result: List[str] = []
        for item in value:
            result.extend(_flatten_text(item))
        return result
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'null', '[]', '{}'}:
        return []
    if text[:1] in '[{' and text[-1:] in ']}':
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if parsed != text:
                    return _flatten_text(list(parsed.values()) if isinstance(parsed, dict) else parsed)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                pass
    return [part.strip() for part in _SPLIT_RE.split(text) if part.strip()]

def extract_aligned_news(df: pd.DataFrame, decision_idx: int, news_cols: Sequence[str]=NEWS_TEXT_COLS, lookback_days: int=NEWS_LOOKBACK_DAYS) -> List[str]:
    if len(df) == 0:
        return []
    decision_idx = int(np.clip(decision_idx, 0, len(df) - 1))
    if 'trade_date' in df.columns:
        dates = pd.to_datetime(df['trade_date'], errors='coerce')
        cutoff = dates.iloc[decision_idx]
        if pd.notna(cutoff):
            start = cutoff - pd.Timedelta(days=max(lookback_days - 1, 0))
            row_ids = np.flatnonzero((dates >= start) & (dates <= cutoff))
        else:
            row_ids = np.arange(max(0, decision_idx - lookback_days + 1), decision_idx + 1)
    else:
        row_ids = np.arange(max(0, decision_idx - lookback_days + 1), decision_idx + 1)
    items: List[str] = []
    seen = set()
    for row_idx in row_ids:
        for col in news_cols:
            if col not in df.columns:
                continue
            for article in _flatten_text(df.iloc[int(row_idx)][col]):
                normalized = ' '.join(article.split())
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    items.append(normalized)
    return items

def _token_id(token: str, vocab_size: int) -> int:
    digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
    return 2 + int.from_bytes(digest, 'little') % max(vocab_size - 2, 1)

def encode_news_items(items: Iterable[str], max_items: int=MAX_NEWS_ITEMS, max_tokens: int=MAX_NEWS_TOKENS, vocab_size: int=TEXT_VOCAB_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    token_ids = np.zeros((max_items, max_tokens), dtype=np.int64)
    item_mask = np.zeros(max_items, dtype=np.bool_)
    for item_idx, article in enumerate(list(items)[:max_items]):
        tokens = _TOKEN_RE.findall(article.lower())[:max_tokens]
        if not tokens:
            continue
        token_ids[item_idx, :len(tokens)] = [_token_id(tok, vocab_size) for tok in tokens]
        item_mask[item_idx] = True
    return (token_ids, item_mask)

def build_news_observation(df: pd.DataFrame, decision_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    return encode_news_items(extract_aligned_news(df, decision_idx))
