# -*- coding: utf-8 -*-
"""datasets_loader.py — Dataset loading and sampling for intervention_analysis.py.

Three standard benchmarks are used:
  - SST-2       Natural language sentences (Stanford Sentiment Treebank).
  - GSM8K       Grade-school math word problems.
  - HumanEval   Python programming prompts.

Examples with fewer than `cut_length` tokens are discarded.
All kept examples are truncated to exactly `cut_length` tokens.
`sample_size` examples are then drawn with a fixed random seed.
"""

import random
import numpy as np
from datasets import load_dataset

# ─── Dataset registry ────────────────────────────────────────────────────────

DATASET_SPECS = [
    {
        "name": "sst2",
        "hf_path": "stanfordnlp/sst2",
        "config": None,
        "split": "train",
        "text_field": "sentence",
        "kind": "natural_language",
    },
    {
        "name": "gsm8k",
        "hf_path": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "text_field": "question",
        "kind": "math",
    },
    {
        "name": "humaneval",
        "hf_path": "openai/openai_humaneval",
        "config": None,
        "split": "test",
        "text_field": "prompt",
        "kind": "code",
    },
]

DEFAULT_SAMPLE_SIZE = 100
DEFAULT_CUT_LENGTH  = 40
DEFAULT_SEED        = 42

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_one_dataset(spec):
    kwargs = {}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    return load_dataset(spec["hf_path"], **kwargs, split=spec["split"])


def _normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def _truncate(tokenizer, text, cut_length):
    """Return *text* truncated to exactly *cut_length* GPT-2 tokens."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids[:cut_length])


# ─── Public API ──────────────────────────────────────────────────────────────

def verify_datasets(tokenizer,
                    sample_size=DEFAULT_SAMPLE_SIZE,
                    cut_length=DEFAULT_CUT_LENGTH):
    """Check that every dataset has at least `sample_size` examples with
    >= `cut_length` tokens (i.e. that can be truncated to the target length).

    Prints a summary table and raises ``ValueError`` for any shortfall.
    """
    print("Verifying datasets...")
    print(f"  Cut length: {cut_length} tokens (examples shorter than this are discarded)")
    print(f"  Required per dataset: {sample_size}\n")

    ok = True
    for spec in DATASET_SPECS:
        print(f"  Loading {spec['name']} ({spec['hf_path']}) ...")
        ds = _load_one_dataset(spec)
        count = 0
        for ex in ds:
            text = _normalize_text(ex.get(spec["text_field"], ""))
            if not text:
                continue
            tlen = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            if tlen >= cut_length:
                count += 1

        status = "OK" if count >= sample_size else "FAIL"
        print(f"    [{status}] {count} usable examples  (need {sample_size})")

        if count < sample_size:
            ok = False

    if not ok:
        raise ValueError(
            "One or more datasets do not have enough usable examples. "
            "Reduce --sample-size or --cut-length."
        )
    print("\nAll datasets verified OK.\n")


def sample_benchmark_datasets(tokenizer,
                               sample_size=DEFAULT_SAMPLE_SIZE,
                               cut_length=DEFAULT_CUT_LENGTH,
                               seed=DEFAULT_SEED):
    """Load, filter, truncate, and sample from all three benchmark datasets.

    Only examples with at least `cut_length` GPT-2 tokens are kept.
    All selected examples are then decoded back to text after truncation to
    exactly `cut_length` tokens, so every example in the returned corpus has
    the same sequence length.

    Parameters
    ----------
    tokenizer   : GPT2Tokenizer
    sample_size : int  — examples to draw from each dataset.
    cut_length  : int  — token count threshold and truncation target.
    seed        : int  — random seed for reproducibility.

    Returns
    -------
    sampled       : dict[str, list[str]]
        dataset_name → list of truncated text strings (each == cut_length tokens).
    manifest_rows : list[dict]
        One row per sampled example (for audit / reproducibility CSV).
    """
    rng = random.Random(seed)
    sampled = {}
    manifest_rows = []

    for spec in DATASET_SPECS:
        print(f"Sampling {spec['name']} ({spec['hf_path']}) ...")
        ds = _load_one_dataset(spec)

        candidates = []
        for idx, ex in enumerate(ds):
            text = _normalize_text(ex.get(spec["text_field"], ""))
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= cut_length:
                truncated = tokenizer.decode(ids[:cut_length])
                candidates.append({
                    "dataset":               spec["name"],
                    "kind":                  spec["kind"],
                    "hf_path":               spec["hf_path"],
                    "split":                 spec["split"],
                    "source_index":          idx,
                    "text":                  truncated,
                    "original_token_length": len(ids),
                    "token_length":          cut_length,
                })

        if len(candidates) < sample_size:
            raise ValueError(
                f"Dataset '{spec['name']}' only has {len(candidates)} usable examples "
                f"(need {sample_size}). Run verify_datasets() first."
            )

        rng.shuffle(candidates)
        chosen = candidates[:sample_size]

        print(f"  Sampled {len(chosen)} examples  "
              f"(all truncated to {cut_length} tokens; "
              f"mean original length={np.mean([r['original_token_length'] for r in chosen]):.1f})")

        sampled[spec["name"]] = [r["text"] for r in chosen]
        manifest_rows.extend(chosen)

    return sampled, manifest_rows
