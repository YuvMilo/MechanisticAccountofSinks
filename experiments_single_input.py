# -*- coding: utf-8 -*-
"""experiments_single_input.py — Single-sentence and model-weight analyses.


Modes
-----
  --mode epe-bias-proj    Per-head EPE alignment density histograms (weight-only,
                          no input needed).  For each layer, plots two overlapping
                          density histograms: cos(bq_h, wk_h @ EPE_1) vs all
                          other positions.

  --mode massive-activations   EPE_1 coordinate values (no input needed).
"""

import math
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_SENTENCE = "It was the best of times, it was the worst of times, it was the age of wisdom."

LAYER_RANGE_START = 3   # 0-indexed inclusive  (paper layer 4)
LAYER_RANGE_END = 11    # 0-indexed exclusive  (paper layer 11)

PURE_POSITIONAL_INDEX = 0

from datasets_loader import DEFAULT_CUT_LENGTH


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def get_initial_embeddings(model, inputs):
    captured = {}

    def _wpe_hook(_module, _input, output):
        captured["pos_enc"] = output.clone()
        return output

    def _wte_hook(_module, _input, output):
        captured["token_embeddings"] = output.clone()
        return output

    hooks = [
        model.transformer.wpe.register_forward_hook(_wpe_hook),
        model.transformer.wte.register_forward_hook(_wte_hook),
    ]
    with torch.no_grad():
        model(**inputs, output_attentions=False)
    for h in hooks:
        h.remove()

    return captured["pos_enc"], captured["token_embeddings"]


# ═══════════════════════════════════════════════════════════════════════════════
# Per-layer, per-head similarity extraction
# ═══════════════════════════════════════════════════════════════════════════════

def compute_layer_similarities(hidden_states, layer, ppes):
    """Compute per-head bq·k and bq·ppe similarity for one layer.

    Returns
    -------
    similarities_bq : np.ndarray, shape [num_heads, seq_len]
        ``dot(bq_h, wk_h @ hidden[pos])`` per head h and position.
    similarities_ppes_bq : np.ndarray, shape [num_heads, num_positions] or empty
        ``cosine_similarity(bq_h, wk_h @ ppe[pos])`` per head h and position.
    attention_output : Tensor
        Layer attention output (needed to continue the forward pass).
    """
    attn_layer = layer.attn
    hidden_size = hidden_states.size(-1)

    qkv_weight = attn_layer.c_attn.weight.t()
    qkv_bias = attn_layer.c_attn.bias
    wq, wk, wv = qkv_weight.chunk(3, dim=0)
    bq, bk, bv = qkv_bias.chunk(3, dim=0)

    num_heads = attn_layer.num_heads
    head_dim = hidden_size // num_heads
    scale = head_dim ** -0.5

    bq_heads = bq.view(num_heads, head_dim)            # [H, D]
    wk_heads = wk.view(num_heads, head_dim, hidden_size)  # [H, D, hidden]

    seq_len = hidden_states.size(1)
    h0 = hidden_states[0]  # [seq_len, hidden]

    # Per-head bq·k: [H, seq_len]
    keys_no_bias = torch.matmul(wk_heads, h0.T)  # [H, D, seq_len]
    similarities_bq = torch.einsum("hd,hds->hs", bq_heads, keys_no_bias)
    similarities_bq = similarities_bq.detach().cpu().numpy()

    # Per-head bq·ppe cosine: [H, num_positions]
    if ppes is not None:
        num_pos = len(ppes)
        ppes_stack = torch.stack(list(ppes))  # [num_pos, hidden]
        keys_ppe = torch.matmul(wk_heads, ppes_stack.T)  # [H, D, num_pos]

        bq_norm = bq_heads / (bq_heads.norm(dim=1, keepdim=True) + 1e-12)  # [H, D]
        keys_ppe_t = keys_ppe.permute(0, 2, 1)  # [H, num_pos, D]
        keys_ppe_norm = keys_ppe_t / (keys_ppe_t.norm(dim=2, keepdim=True) + 1e-12)
        cos_sim = torch.einsum("hd,hpd->hp", bq_norm, keys_ppe_norm)  # [H, num_pos]
        similarities_ppes_bq = cos_sim.detach().cpu().numpy()
    else:
        similarities_ppes_bq = np.array([])

    # Standard multi-head attention forward (unchanged)
    query_proj = F.linear(hidden_states, wq, bq)
    key_proj = F.linear(hidden_states, wk, bk)
    value_proj = F.linear(hidden_states, wv, bv)

    def reshape(x):
        return x.view(x.size(0), x.size(1), num_heads, head_dim).transpose(1, 2)

    query = reshape(query_proj)
    key = reshape(key_proj)
    value = reshape(value_proj)

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=hidden_states.device)
    ).view(1, 1, seq_len, seq_len)
    attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
    attn_weights = F.softmax(attn_scores, dim=-1)

    context = torch.matmul(attn_weights, value)
    context = context.transpose(1, 2).contiguous().view(
        hidden_states.size(0), seq_len, hidden_size
    )
    output_weight = attn_layer.c_proj.weight.t()
    output_bias = attn_layer.c_proj.bias
    attention_output = F.linear(context, output_weight, output_bias)

    return similarities_bq, similarities_ppes_bq, attention_output


def collect_similarities_for_sentence(model, token_embeddings, pos_enc):
    """Run a full manual forward pass and return per-layer, per-head similarities.

    Returns
    -------
    all_bq_k   : list[np.ndarray]  — each shape [num_heads, seq_len]
    all_bq_ppe : list[np.ndarray]  — each shape [num_heads, num_positions]
    """
    layer_input = token_embeddings.clone() + pos_enc.clone()
    ppes = pos_enc.clone()[0] + model.transformer.h[0].mlp(pos_enc.clone())[0]

    all_bq_k = []
    all_bq_ppe = []

    for i in range(len(model.transformer.h)):
        layer = model.transformer.h[i]
        normalized = layer.ln_1(layer_input.clone())

        sim_bq, sim_ppe, attn_out = compute_layer_similarities(
            normalized, layer, ppes
        )
        all_bq_k.append(sim_bq)
        all_bq_ppe.append(sim_ppe)

        attn_and_res = layer_input + attn_out
        normed = layer.ln_2(attn_and_res)
        mlp_out = layer.mlp(normed)
        layer_input = attn_and_res + mlp_out

    return all_bq_k, all_bq_ppe


# ═══════════════════════════════════════════════════════════════════════════════
# EPE similarity computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_epe_similarities(model, token_embeddings, pos_enc):
    """Compute cos(EPE_i, R_i - O_i) for each token position.

    Returns list[float] of length seq_len.
    """
    mlp = model.transformer.h[0].mlp

    x = token_embeddings.clone() + pos_enc.clone()
    e = token_embeddings.clone()
    p = pos_enc.clone()

    R = x + mlp(x)
    O = e + mlp(e)
    epe = p + mlp(p)

    diff = R - O

    sims = []
    for i in range(x.size(1)):
        epe_np = epe[0, i].detach().cpu().numpy().reshape(1, -1)
        diff_np = diff[0, i].detach().cpu().numpy().reshape(1, -1)
        sim = cosine_similarity(epe_np, diff_np)[0][0]
        sims.append(float(sim))
    return sims


def forward_first_block(model, hidden_states):
    """GPT-2 block 0: LN1 → ``layer.attn`` → residual → LN2 → MLP → residual."""
    layer = model.transformer.h[0]
    normalized = layer.ln_1(hidden_states.clone())
    attn_out, _ = layer.attn(normalized, output_attentions=False)
    attention_and_residual = hidden_states + attn_out
    normalized_ar = layer.ln_2(attention_and_residual)
    mlp_out = layer.mlp(normalized_ar)
    return attention_and_residual + mlp_out


def compute_epe_full_first_layer_similarities(model, token_embeddings, pos_enc):
    """cos(EPE_i, R_i - O_i) at every position; R/O from full block 0.

    Same as :func:`compute_epe_similarities` but ``R``/``O`` are the outputs of
    the entire first transformer block (LN1 → attn → residual → LN2 → MLP → residual)
    instead of just the MLP.

    Returns list[float] of length seq_len.
    """
    mlp = model.transformer.h[0].mlp

    x = token_embeddings.clone() + pos_enc.clone()
    e = token_embeddings.clone()
    p = pos_enc.clone()

    R = forward_first_block(model, x)
    O = forward_first_block(model, e)
    epe = p + mlp(p)

    diff = R - O

    sims = []
    for i in range(x.size(1)):
        epe_np = epe[0, i].detach().cpu().numpy().reshape(1, -1)
        diff_np = diff[0, i].detach().cpu().numpy().reshape(1, -1)
        sim = cosine_similarity(epe_np, diff_np)[0][0]
        sims.append(float(sim))
    return sims


# ═══════════════════════════════════════════════════════════════════════════════
# EPE-Bias Projection Alignment — per-head density histograms (weight-only)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_perhead_epe_cosine(model, num_positions):
    """Compute per-head cos(bq_h, wk_h @ EPE_j) for all layers and positions.

    Returns
    -------
    result : np.ndarray, shape [num_layers, num_heads, num_positions]
    """
    num_layers = len(model.transformer.h)
    num_heads = model.config.n_head
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads

    with torch.no_grad():
        pos_ids = torch.arange(num_positions).unsqueeze(0)  # [1, num_pos]
        pe = model.transformer.wpe(pos_ids)  # [1, num_pos, hidden]
        epes = pe[0] + model.transformer.h[0].mlp(pe)[0]  # [num_pos, hidden]

    result = np.zeros((num_layers, num_heads, num_positions))

    for layer_idx in range(num_layers):
        layer = model.transformer.h[layer_idx]
        attn = layer.attn
        qkv_weight = attn.c_attn.weight.t()
        qkv_bias = attn.c_attn.bias
        bq = qkv_bias.chunk(3, dim=0)[0]
        wk = qkv_weight.chunk(3, dim=0)[1]

        bq_heads = bq.view(num_heads, head_dim)
        wk_heads = wk.view(num_heads, head_dim, hidden_size)

        keys = torch.matmul(wk_heads, epes.T)  # [H, D, num_pos]

        bq_norm = bq_heads / (bq_heads.norm(dim=1, keepdim=True) + 1e-12)
        keys_t = keys.permute(0, 2, 1)  # [H, num_pos, D]
        keys_norm = keys_t / (keys_t.norm(dim=2, keepdim=True) + 1e-12)
        cos_sim = torch.einsum("hd,hpd->hp", bq_norm, keys_norm)

        result[layer_idx] = cos_sim.detach().cpu().numpy()

    return result


def _plot_epe_dual_histogram(epe1_vals, other_vals, save_path,
                              figsize=(8, 3.3), n_bins=40):
    """Save one dual density histogram: EPE_1 (red) vs other positions (blue)."""
    all_vals = np.concatenate([epe1_vals, other_vals])
    bins = np.linspace(all_vals.min() - 0.05, all_vals.max() + 0.05, n_bins)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(other_vals, bins=bins, density=True, alpha=0.6,
            color="steelblue", edgecolor="black", linewidth=0.8,
            label="Other positions")
    ax.hist(epe1_vals, bins=bins, density=True, alpha=0.7,
            color="tab:red", edgecolor="black", linewidth=0.8,
            label="Position 1 (EPE$_1$)")
    ax.set_xlabel("Cosine similarity", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.tick_params(labelsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(save_path).name}")


def bq_ppe_analysis(model, output_dir, num_positions=DEFAULT_CUT_LENGTH):
    """Per-head EPE alignment density histograms.

    Produces:
      1. ``epe_alignment.png`` — pooled across all relevant layers (main figure).
      2. ``per_layer/epe_alignment_layer_XX.png`` — one plot per relevant layer.

    Each plot shows two overlapping probability-density histograms:
      - EPE_1 (red): cos(bq_h, wk_h @ EPE_1) across all heads (and layers for pooled)
      - Other positions (blue): same for positions 2..num_positions
    """
    output_path = Path(output_dir) / "epe_bias_proj"
    output_path.mkdir(parents=True, exist_ok=True)
    per_layer_dir = output_path / "per_layer"
    per_layer_dir.mkdir(parents=True, exist_ok=True)

    cos_data = _compute_perhead_epe_cosine(model, num_positions)
    # cos_data shape: [num_layers, num_heads, num_positions]

    num_layers = cos_data.shape[0]
    layer_end = min(LAYER_RANGE_END, num_layers)

    # --- 1. Pooled figure across all relevant layers ---
    selected = cos_data[LAYER_RANGE_START:layer_end, :, :]  # [L_sel, H, P]
    epe1_vals = selected[:, :, 0].ravel()    # [L_sel * H]
    other_vals = selected[:, :, 1:].ravel()  # [L_sel * H * (P-1)]
    _plot_epe_dual_histogram(epe1_vals, other_vals,
                             save_path=output_path / "epe_alignment.png")

    # --- 2. Per-layer figures ---
    for layer_idx in range(LAYER_RANGE_START, layer_end):
        epe1_vals_l = cos_data[layer_idx, :, 0]       # [H]
        other_vals_l = cos_data[layer_idx, :, 1:].ravel()  # [H * (P-1)]
        _plot_epe_dual_histogram(
            epe1_vals_l, other_vals_l,
            save_path=per_layer_dir / f"epe_alignment_layer_{layer_idx+1:02d}.png",
        )

    print(f"\nOutputs saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Massive activations in EPE_1
# ═══════════════════════════════════════════════════════════════════════════════

def massive_activations_analysis(model, output_dir):
    """Plot the coordinate values of EPE_1 showing massive activations."""
    output_path = Path(output_dir) / "massive_activations"
    output_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        pe_first = model.transformer.wpe(torch.tensor([[0]]))
        epe_first = pe_first[0, 0] + model.transformer.h[0].mlp(pe_first)[0, 0]

    values = epe_first.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(values, linewidth=1.0, color="steelblue")
    ax.set_xlabel("Dimension index", fontsize=20)
    ax.set_ylabel("Value", fontsize=20)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "massive_activations_in_ppe.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    abs_vals = np.abs(values)
    mean_abs = abs_vals.mean()
    std_abs = abs_vals.std()
    massive_mask = abs_vals > (mean_abs + 3 * std_abs)
    massive_indices = np.where(massive_mask)[0].tolist()
    print(f"Massive activation indices: {massive_indices}")
    for idx in massive_indices:
        print(f"  dim {idx}: {values[idx]:.2f}")
    print(f"\nOutputs saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Single-sentence and model-weight analyses for GPT-2 (per-head).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["epe-bias-proj", "massive-activations"],
        required=True,
        help="'epe-bias-proj' — per-head EPE alignment density histograms (weight-only); "
             "'massive-activations' — EPE_1 coordinate values (no input).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Root directory for all outputs.",
    )
    args = parser.parse_args()

    print("Loading GPT-2…")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    print("Model loaded.\n")

    if args.mode == "massive-activations":
        massive_activations_analysis(model, args.output_dir)
    elif args.mode == "epe-bias-proj":
        bq_ppe_analysis(model, args.output_dir)


if __name__ == "__main__":
    main()
