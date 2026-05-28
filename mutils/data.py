from __future__ import annotations

from pathlib import Path

import datasets as ds
import pandas as pd

__all__ = ["get_wildchat_prompts"]


def get_wildchat_prompts(
    n: int = 100,
    max_chars: int = 1500,
    language: str = "English",
    seed: int = 42,
    cache_path: str | Path | None = None,
) -> list[str]:
    """Retrieve N random first-turn user prompts from allenai/WildChat-1M.

    Streams the dataset (no full download) with a 10K shuffle buffer, filters to
    `language` and `len(content) < max_chars`, returns the first n matches.

    Args:
        n: Number of prompts to return.
        max_chars: Maximum content length in characters.
        language: WildChat 'language' field value to filter on.
        seed: Shuffle seed.
        cache_path: If set and the file exists, load from it; otherwise fetch from HF
            and write to it for next time.
    Returns:
        List of n prompt strings.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return pd.read_parquet(cache_path).prompt.tolist()

    raw = ds.load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    raw = raw.shuffle(seed=seed, buffer_size=10_000)
    prompts = []
    for r in raw:
        c = r["conversation"][0]["content"]
        if r.get("language") == language and len(c) < max_chars:
            prompts.append(c)
        if len(prompts) >= n:
            break

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"prompt_idx": range(len(prompts)), "prompt": prompts}).to_parquet(cache_path)

    return prompts
