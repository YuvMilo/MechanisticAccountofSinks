# -*- coding: utf-8 -*-
"""intervention_analysis.py — Attention intervention experiments on GPT-2.

Two modes
---------
  --mode sentence  (default)  Per-(layer, head) 2×5 attention-heatmap grids for one sentence,
                               plus a head-averaged grid per layer.
                               Files: layer_{l}_head_{h}.png and layer_{l}_avg.png.

  --mode dataset               BOS-attention statistics over three standard benchmarks
                               (SST-2, GSM8K, HumanEval).  Each is sampled to --sample-size
                               examples filtered to [--min-tokens, --max-tokens] tokens.
                               Outputs under ``<output-dir>/dataset_analysis/``:
                                 sample_manifest.csv
                                 bos_attention_stats_by_dataset.csv
                                 bos_attention_stats_overall.csv
                                 bos_attention_summary_{all,mid}_layers.{csv,txt}
                                 <dataset>/bos_attention_summary_{scope}.{csv,txt}

Metric (dataset mode)
---------------------
  Average attention to the BOS token (position 0) from the second half of the sequence,
  over all heads, in layers 4-11 (1-indexed, excluding first 3 and last 1).

Interventions (labelled a–h)
-----------------------------
  (a) Baseline          Regular transformer: token + positional embeddings, all layers intact.
  (b) No Query Bias     Sets bq = 0 in every layer, isolating content-only queries.
  (c) Remove First PE   Position 0 receives PE[1] instead of PE[0] as input.
  (d) Swap EPE          After layer-0 MLP, subtract EPE[0] from pos-0 and add EPE[1],
                        and vice-versa at pos-1.  Both vectors are rescaled to the EPE
                        component of the layer-0 residual stream at position 0.
  (e) Swap PE           Same swap as (d) but using the raw PE vectors instead of EPEs.
  (f) Nullify BOS Token Zero out the BOS token embedding before processing; PE still added.
  (g) No MLP            Skip the MLP block in every layer (attention + residual only).
  (h) No PE             Input = token embeddings only; no positional signal anywhere.
  (i) Zero Top-3 Wk     In every layer zero out the 3 columns of Wk whose indices match
                        the top-3 absolute-value dimensions of EPE[0] (ppes[0]).
  (j) Zero Random Wk    In every layer zero out 3 randomly chosen columns of Wk.
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import argparse
import random
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets_loader import (
    sample_benchmark_datasets,
    verify_datasets,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_CUT_LENGTH,
    DEFAULT_SEED,
)
# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SENTENCE = "It was the best of times, it was the worst of times, it was the age of wisdom."

# ═══════════════════════════════════════════════════════════════════════════════
# Embedding retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def get_initial_embeddings(model, inputs):
    """Run one forward pass to capture positional and token embeddings via hooks."""
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
# Manual self-attention  (UNCHANGED from reconstruct_whole_model.py)
# ═══════════════════════════════════════════════════════════════════════════════

def manual_self_attention_new(hidden_states, layer,
                              ppes=None,
                              intervene_query_bias=False,
                              fixed_wk_zero_indices=None,
                              random_wk_zero_rows=False):
    """
    Manual implementation of the self-attention mechanism for a single GPT-2 layer.

    Args:
        hidden_states (torch.Tensor): The input tensor to the attention layer (batch_size, sequence_length, hidden_size).
        layer (torch.nn.Module): The transformer layer module containing the attention sub-layer.
        ppes (torch.Tensor, optional): Positional embeddings (effective positional embeddings) for similarity calculations. Defaults to None.
        intervene_query_bias (bool): If True, nullify the query bias.
        fixed_wk_zero_indices (list or None): If a list of indices is provided, zero out these columns in Wk.
        random_wk_zero_rows (bool): If True, zero out 3 random columns in Wk.

    Returns:
        tuple: A tuple containing:
            - attention_output (torch.Tensor): The output of the attention layer (batch_size, sequence_length, hidden_size).
            - attention_weights (torch.Tensor): The attention weights (batch_size, num_heads, sequence_length, sequence_length).
            - query_before_reshape (torch.Tensor): The query tensor before reshaping for multi-head attention.
            - key_before_reshape (torch.Tensor): The key tensor before reshaping for multi-head attention.
            - attention_scores (torch.Tensor): The raw attention scores before softmax (batch_size, num_heads, sequence_length, sequence_length).
            - result_vector (torch.Tensor): The result of (Wq @ Wk^T) @ first_token_only_massive (or similar, check use).
            - similarities_bq (list): Cosine similarities related to bq.
            - value_of_dot_product_against_attention_unmasked (list): Dot product values.
            - similarities_ppes_bq (list): Cosine similarities related to ppes and bq.
            - similarities_rows_of_wk (list): Similarities related to rows of Wk.
            - bq_Wk (list): bq and wk (modified if interventions applied).
    """
    # Get attention layer parameters
    attn_layer = layer.attn
    num_heads = attn_layer.num_heads
    hidden_size = hidden_states.size(-1)
    head_dim = hidden_size // num_heads
    scale = head_dim ** -0.5 # 1 / sqrt(head_dim)

    # Get QKV weights and biases from the combined linear layer (c_attn)
    # The weight is transposed by Hugging Face for Conv1D, so we transpose it back for standard linear op
    qkv_weight = attn_layer.c_attn.weight.t()
    qkv_bias = attn_layer.c_attn.bias

    # Split the combined weight tensor into Wq, Wk, and Wv
    wq, wk, wv = qkv_weight.chunk(3, dim=0)
    bq, bk, bv = qkv_bias.chunk(3, dim=0)

    # --- INTERVENTION A: Nullify query bias ---
    if intervene_query_bias:
        bq = torch.zeros_like(bq)

    # Calculate Wq multiplied by Wk transposed
    wq_wk_t_product = F.linear(wq, wk.t())

    # Zero out all values except the top 3 in the first token of layer_input
    first_token_vector = abs(hidden_states[0][0].clone())
    topk_values, topk_indices_local = torch.topk(first_token_vector, k=3) # Get top 3 absolute values and their indices
    mask = torch.zeros_like(first_token_vector, dtype=torch.bool)
    mask[topk_indices_local] = True
    first_token_vector_masked = torch.zeros_like(first_token_vector)
    first_token_vector_masked[mask] = first_token_vector[mask]
    first_token_only_massive = first_token_vector_masked
    result_vector = torch.matmul(wq_wk_t_product, first_token_only_massive).view(1, num_heads, 1, head_dim)

    # --- INTERVENTION E/F: Nullify massive/random activation columns in Wk ---
    wk_active = wk.clone() # Use a temporary variable for modifications
    if fixed_wk_zero_indices is not None:
        # User specified indices to zero out
        wk_active[:, fixed_wk_zero_indices] = 0.0
    elif random_wk_zero_rows:
        # Generate random indices to zero out
        random_indices_to_zero = [random.randint(0, hidden_size - 1) for _ in range(3)]
        wk_active[:, random_indices_to_zero] = 0.0


    # Project input to QKV
    query_proj = F.linear(hidden_states, wq, bq)
    key_proj = F.linear(hidden_states, wk_active, bk) # Use wk_active
    value_proj = F.linear(hidden_states, wv, bv)

    # Store query and key before reshaping
    query_before_reshape = query_proj.clone()
    key_before_reshape = key_proj.clone()
    value_reshaped = value_proj.clone()

    # Recalculate k1, similarities_bq, etc. with the possibly modified wk_active
    k1 = F.linear(hidden_states[0][0], wk_active, torch.zeros_like(bk))
    dot_product_bq_k1 = torch.dot(bq, k1).detach()

    similarities_bq = []
    for l in range(len(hidden_states[0])):
      kl = F.linear(hidden_states[0][l], wk_active, torch.zeros_like(bk))
      similarity = torch.dot(bq, kl).detach()
      similarities_bq.append(similarity)

    value_of_dot_product_against_attention_unmasked = []
    attention_scores_unsplit = torch.matmul(query_proj, key_proj.transpose(-2, -1))
    for i_seq in range(attention_scores_unsplit.shape[1]):
        for j_seq in range(attention_scores_unsplit.shape[2]):
            value_of_dot_product_against_attention_unmasked.append(attention_scores_unsplit[0, i_seq, j_seq].detach())

    # Calculate similarities_ppes_bq using the passed ppes argument
    similarities_ppes_bq = []
    if ppes is not None:
        for l in range(len(ppes)):
            kl = torch.matmul(ppes[l], wk_active.T)
            similarity = cosine_similarity(bq.detach().cpu().numpy().reshape(1, -1), kl.detach().cpu().numpy().reshape(1, -1))[0][0]
            similarities_ppes_bq.append(similarity)


    # Reshape Q, K, V for multi-head attention
    def reshape_for_multihead(x):
        return x.view(x.size(0), x.size(1), num_heads, head_dim).transpose(1, 2)

    query = reshape_for_multihead(query_proj)
    key = reshape_for_multihead(key_proj)
    value = reshape_for_multihead(value_proj)

    # Calculate attention scores (Q @ K^T)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_scores_before_mask=attention_scores.clone()

    # Apply causal mask
    sequence_length = hidden_states.size(1)
    causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, device=hidden_states.device)).view(1, 1, sequence_length, sequence_length)
    attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))

    # Apply softmax to get attention probabilities (weights)
    attention_weights = F.softmax(attention_scores, dim=-1)


    # Calculate weighted sum of values (attention_weights @ V)
    context = torch.matmul(attention_weights, value)

    # Concatenate heads and apply final linear projection (c_proj)
    context = context.transpose(1, 2).contiguous().view(hidden_states.size(0), sequence_length, hidden_size)

    # Apply the output projection
    output_weight = attn_layer.c_proj.weight.t() # Transpose
    output_bias = attn_layer.c_proj.bias
    attention_output = F.linear(context, output_weight, output_bias)

    # Recalculate similarities_rows_of_wk and bq_Wk with current bq and wk_active
    all_indexes = list(range(hidden_size))
    similarities_rows_of_wk = []
    for index in all_indexes:
        one_hot_vector = torch.zeros(hidden_size, dtype=torch.float32, device=hidden_states.device)
        one_hot_vector[index] = 1.0
        wk_i = torch.matmul(one_hot_vector, wk_active.T)
        similarity = torch.abs(torch.dot(bq, wk_i).detach())
        similarities_rows_of_wk.append(similarity)
    bq_Wk = [bq, wk_active] # Return the possibly modified wk_active

    return attention_output, attention_weights, query_before_reshape, key_before_reshape, \
           attention_scores_before_mask, result_vector, similarities_bq, \
           value_of_dot_product_against_attention_unmasked, similarities_ppes_bq, \
           similarities_rows_of_wk, bq_Wk

# ═══════════════════════════════════════════════════════════════════════════════
# Generic intervention loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_intervention_loop(model, layer_input, ppes,
                          attn_kwargs=None, modify_mlp_fn=None, skip_mlp=False):
    """Run through all transformer layers, collecting per-layer attention weights.

    Parameters
    ----------
    model : GPT2LMHeadModel
    layer_input : Tensor [batch, seq, hidden]
        Initial input to the first layer.
    ppes : Tensor or None
        Effective positional embeddings (passed to manual_self_attention_new for
        diagnostic similarity calculations only — does not affect attention output).
    attn_kwargs : dict or None
        Extra keyword arguments forwarded to ``manual_self_attention_new``
        (e.g. ``intervene_query_bias=True``).
    modify_mlp_fn : callable(layer_idx, mlp_output) -> mlp_output, or None
        Hook applied to the MLP output before adding the second residual.
    skip_mlp : bool
        If True, MLP is skipped entirely; layer output = input + attn_output.

    Returns
    -------
    list[Tensor]
        Per-layer attention weights with all heads kept, each of
        shape ``[num_heads, seq_len, seq_len]``.
    """
    attn_kwargs = attn_kwargs or {}
    attention_weights_per_layer = []

    for i in range(len(model.transformer.h)):
        layer = model.transformer.h[i]

        normalized = layer.ln_1(layer_input.clone())

        attention_output, attention_weights, *_ = manual_self_attention_new(
            normalized, layer, ppes=ppes, **attn_kwargs
        )

        # Keep all heads: shape [num_heads, seq_len, seq_len] (batch dim squeezed).
        attention_weights_per_layer.append(
            attention_weights.cpu().detach()[0]
        )

        if skip_mlp:
            layer_input = layer_input + attention_output
        else:
            attention_and_residual = layer_input + attention_output
            normalized_ar = layer.ln_2(attention_and_residual)
            mlp_output = layer.mlp(normalized_ar)
            if modify_mlp_fn is not None:
                mlp_output = modify_mlp_fn(i, mlp_output)
            layer_input = attention_and_residual + mlp_output

    return attention_weights_per_layer

# ═══════════════════════════════════════════════════════════════════════════════
# Intervention functions
#
# Each takes (model, token_embeddings, pos_enc) and returns a list of per-layer
# attention-weight tensors [seq_len, seq_len].
# ═══════════════════════════════════════════════════════════════════════════════

def intervention_a_baseline(model, token_embeddings, pos_enc):
    """(a) Baseline: regular transformer with PE added as usual."""
    layer_input = token_embeddings.clone() + pos_enc.clone()
    ppes = pos_enc.clone()[0] + model.transformer.h[0].mlp(pos_enc.clone())[0]
    return run_intervention_loop(model, layer_input, ppes)


def intervention_b_nullify_query_bias(model, token_embeddings, pos_enc):
    """(b) Nullify Query Bias: sets bq = 0 in every layer."""
    layer_input = token_embeddings.clone() + pos_enc.clone()
    ppes = pos_enc.clone()[0] + model.transformer.h[0].mlp(pos_enc.clone())[0]
    return run_intervention_loop(
        model, layer_input, ppes,
        attn_kwargs={"intervene_query_bias": True},
    )


def intervention_c_remove_first_pe(model, token_embeddings, pos_enc):
    """(c) Remove First PE: position 0 receives PE[1] instead of PE[0]."""
    pos_enc_copy = pos_enc.clone()
    pos_enc_copy[0][0] = pos_enc_copy[0][1].clone()
    layer_input = token_embeddings.clone() + pos_enc_copy
    ppes = pos_enc_copy[0] + model.transformer.h[0].mlp(pos_enc_copy)[0]
    return run_intervention_loop(model, layer_input, ppes)


def intervention_d_swap_epe_normalized(model, token_embeddings, pos_enc):
    """(d) Swap EPE: after layer-0 MLP, swap the EPE component between pos 0 and pos 1.

    True component swap — pos 0 loses its EPE[0] component and gains EPE[1] in its
    place (same magnitude), and pos 1 gets the mirror:

        alpha = (mlp_out[0] · EPE[0]) / ‖EPE[0]‖
        mlp_out[0]  -= alpha * EPE[0]_hat  (remove EPE[0] component)
        mlp_out[0]  += alpha * EPE[1]_hat  (replace with EPE[1] direction)
        mlp_out[1]  += alpha * EPE[0]_hat  (give pos 1 the EPE[0] component)
        mlp_out[1]  -= alpha * EPE[1]_hat  (remove EPE[1] from pos 1)

    After the swap pos 0 looks positionally like pos 1 and vice-versa.
    """
    pos_enc_copy = pos_enc.clone()
    layer_input = token_embeddings.clone() + pos_enc_copy
    ppes = pos_enc_copy[0] + model.transformer.h[0].mlp(pos_enc_copy)[0]

    epe0_hat = ppes[0] / torch.linalg.norm(ppes[0])
    epe1_hat = ppes[1] / torch.linalg.norm(ppes[1])

    def _modify(layer_idx, mlp_output):
        if layer_idx == 0:
            alpha = torch.dot(mlp_output[0][0], epe0_hat)
            mlp_output[0][0] = mlp_output[0][0] - alpha * epe0_hat + alpha * epe1_hat
            mlp_output[0][1] = mlp_output[0][1] + alpha * epe0_hat - alpha * epe1_hat
        return mlp_output

    return run_intervention_loop(model, layer_input, ppes, modify_mlp_fn=_modify)


def intervention_e_swap_pe(model, token_embeddings, pos_enc):
    """(e) Swap PE: after layer-0 MLP, swap the PE component between pos 0 and pos 1.

    Same true component swap as int_d but using raw PE vectors as the direction:

        alpha = (mlp_out[0] · PE[0]) / ‖PE[0]‖
        mlp_out[0]  -= alpha * PE[0]_hat
        mlp_out[0]  += alpha * PE[1]_hat
        mlp_out[1]  += alpha * PE[0]_hat
        mlp_out[1]  -= alpha * PE[1]_hat
    """
    pos_enc_copy = pos_enc.clone()
    layer_input = token_embeddings.clone() + pos_enc_copy
    ppes = pos_enc_copy[0] + model.transformer.h[0].mlp(pos_enc_copy)[0]

    pe0_hat = pos_enc_copy[0][0] / torch.linalg.norm(pos_enc_copy[0][0])
    pe1_hat = pos_enc_copy[0][1] / torch.linalg.norm(pos_enc_copy[0][1])

    def _modify(layer_idx, mlp_output):
        if layer_idx == 0:
            alpha = torch.dot(mlp_output[0][0], pe0_hat)
            mlp_output[0][0] = mlp_output[0][0] - alpha * pe0_hat + alpha * pe1_hat
            mlp_output[0][1] = mlp_output[0][1] + alpha * pe0_hat - alpha * pe1_hat
        return mlp_output

    return run_intervention_loop(model, layer_input, ppes, modify_mlp_fn=_modify)


def intervention_f_nullify_bos_token(model, token_embeddings, pos_enc):
    """(f) Nullify BOS Token: zero out the BOS token embedding before processing."""
    pos_enc_copy = pos_enc.clone()
    te_copy = token_embeddings.clone()
    te_copy[0][0] = torch.zeros_like(te_copy[0][0])
    layer_input = te_copy + pos_enc_copy
    ppes = pos_enc_copy[0] + model.transformer.h[0].mlp(pos_enc_copy)[0]
    return run_intervention_loop(model, layer_input, ppes)


def intervention_g_no_mlp(model, token_embeddings, pos_enc):
    """(g) No MLP: skip the MLP block in every layer."""
    pos_enc_copy = pos_enc.clone()
    layer_input = token_embeddings.clone() + pos_enc_copy
    ppes = pos_enc_copy[0] + model.transformer.h[0].mlp(pos_enc_copy)[0]
    return run_intervention_loop(model, layer_input, ppes, skip_mlp=True)


def intervention_h_no_pe(model, token_embeddings, pos_enc):
    """(h) No PE: input is token embeddings only; no positional signal anywhere."""
    layer_input = token_embeddings.clone()
    return run_intervention_loop(model, layer_input, ppes=None)


def intervention_i_zero_top_wk(model, token_embeddings, pos_enc):
    """(i) Zero Top-3 Wk: in every layer, zero out the 3 columns of Wk whose
    indices correspond to the top-3 absolute-value dimensions of EPE[0] (ppes[0]).

    The intuition is that these large-activation coordinates of the effective
    positional embedding dominate the key projection; removing them tests how
    much the BOS-attention sink depends on those specific dimensions.

    The same column indices are zeroed in Wk for every layer.
    """
    pos_enc_copy = pos_enc.clone()
    layer_input = token_embeddings.clone() + pos_enc_copy
    ppes = pos_enc_copy[0] + model.transformer.h[0].mlp(pos_enc_copy)[0]

    # Find top-3 dimensions of |EPE[0]| and fix them for every layer.
    _, topk_indices = torch.topk(ppes[0].abs(), k=3)
    fixed_wk_zero_indices = topk_indices.tolist()

    return run_intervention_loop(
        model, layer_input, ppes,
        attn_kwargs={"fixed_wk_zero_indices": fixed_wk_zero_indices},
    )


def intervention_j_zero_random_wk(model, token_embeddings, pos_enc):
    """(j) Zero Random Wk: in every layer, zero out 3 randomly chosen columns of Wk.

    Acts as a control for intervention (i): if (i) shows a large effect, this
    confirms that the result is specific to the top-EPE dimensions rather than
    any random ablation of Wk columns.
    """
    random.seed(DEFAULT_SEED)
    pos_enc_copy = pos_enc.clone()
    layer_input = token_embeddings.clone() + pos_enc_copy
    ppes = pos_enc_copy[0] + model.transformer.h[0].mlp(pos_enc_copy)[0]

    return run_intervention_loop(
        model, layer_input, ppes,
        attn_kwargs={"random_wk_zero_rows": True},
    )


# Ordered registry — the iteration order defines the (a)…(j) grid layout.
INTERVENTIONS = [
    ("int_a", "(a)", "Baseline",              intervention_a_baseline),
    ("int_b", "(b)", "No Query Bias",         intervention_b_nullify_query_bias),
    ("int_c", "(c)", "Remove First PE",       intervention_c_remove_first_pe),
    ("int_d", "(d)", "Swap EPE",              intervention_d_swap_epe_normalized),
    ("int_e", "(e)", "Swap PE",               intervention_e_swap_pe),
    ("int_f", "(f)", "Nullify BOS Token",     intervention_f_nullify_bos_token),
    ("int_g", "(g)", "No MLP",                intervention_g_no_mlp),
    ("int_h", "(h)", "No PE",                 intervention_h_no_pe),
    ("int_i", "(i)", "Zero Top-3 Wk",        intervention_i_zero_top_wk),
    ("int_j", "(j)", "Zero Random Wk",       intervention_j_zero_random_wk),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_interventions(model, token_embeddings, pos_enc):
    """Run every registered intervention and return {key: [layer_weights]}."""
    results = {}
    for key, _label, desc, fn in INTERVENTIONS:
        print(f"  Running {key} ({desc})...")
        with torch.no_grad():
            results[key] = fn(model, token_embeddings, pos_enc)
    return results


LAYER_RANGE_START = 3   # 0-indexed inclusive  (paper layer 4)
LAYER_RANGE_END = 11    # 0-indexed exclusive  (paper layer 11)

def compute_bos_attention_metric(attn_weights_per_layer, num_layers, layer_scope="mid"):
    """Average attention to BOS from 2nd-half tokens over selected layers, across all heads.

    Parameters
    ----------
    attn_weights_per_layer : list[Tensor]
        One ``[num_heads, seq_len, seq_len]`` tensor per layer.
    num_layers : int
        Total number of layers in the model.
    layer_scope : str
        ``"mid"`` — layers 4-11 (1-indexed), i.e. 0-indexed [3, 11).
        ``"all"`` — every layer ``0 .. num_layers-1``.

    Returns
    -------
    float
        Scalar BOS-attention metric (averaged over heads, tokens, and layers).
    """
    if layer_scope == "all":
        layer_start = 0
        layer_end = num_layers
    elif layer_scope == "mid":
        layer_start = LAYER_RANGE_START
        layer_end = min(LAYER_RANGE_END, num_layers)
    else:
        raise ValueError(f"layer_scope must be 'all' or 'mid', got {layer_scope!r}")

    seq_len = attn_weights_per_layer[0].shape[-1]
    second_half_start = seq_len // 2

    values = []
    for layer_idx in range(layer_start, layer_end):
        attn = attn_weights_per_layer[layer_idx]  # [num_heads, seq, seq]
        bos_attn = attn[:, second_half_start:, 0].mean().item()
        values.append(bos_attn)

    return float(np.mean(values))


def _write_readable_bos_summaries(output_path, scores_by_key, suffix):
    """Write human-readable CSV and TXT.

    CSV: intervention name + mean score (2 decimal places).
    TXT: same, plus a 'relative' section showing each intervention's score
         as a percentage of the baseline (int_a) score.
    """
    rows_csv = []
    means = {}
    for key, _label, desc, _fn in INTERVENTIONS:
        mean = float(np.array(scores_by_key[key]).mean())
        means[key] = mean
        rows_csv.append({"intervention": desc, "score": f"{mean:.2f}"})

    baseline = means.get("int_a", 1.0) or 1.0   # avoid div-by-zero

    lines_txt = []
    for key, _label, desc, _fn in INTERVENTIONS:
        lines_txt.append(f"{desc}\t{means[key]:.2f}")

    lines_txt.append("")
    lines_txt.append("relative (% of baseline)")
    for key, _label, desc, _fn in INTERVENTIONS:
        pct = 100.0 * means[key] / baseline
        lines_txt.append(f"{desc}\t{pct:.1f}%")

    stem = f"bos_attention_summary_{suffix}"
    pd.DataFrame(rows_csv).to_csv(output_path / f"{stem}.csv", index=False)
    (output_path / f"{stem}.txt").write_text("\n".join(lines_txt) + "\n", encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════════════
# Mode 1 — Sentence analysis (per-head + head-averaged 2×4 grid figures)
# ═══════════════════════════════════════════════════════════════════════════════

def _save_attention_grid(all_results, layer_idx, matrices_fn, title, save_path):
    """Save a 2×4 grid of attention heatmaps for all 8 interventions.

    Parameters
    ----------
    all_results : dict  {key: list[Tensor[num_heads, seq, seq]]}
    layer_idx   : int
    matrices_fn : callable(key) -> 2-D array  — extracts the matrix to plot.
    title       : str   — figure suptitle.
    save_path   : Path
    """
    matrices = {key: matrices_fn(key) for key, *_ in INTERVENTIONS}
    vmin = min(m.min() for m in matrices.values())
    vmax = max(m.max() for m in matrices.values())

    fig, axes = plt.subplots(2, 5, figsize=(30, 11))

    for idx, (key, label, desc, _fn) in enumerate(INTERVENTIONS):
        row, col = divmod(idx, 5)
        ax = axes[row][col]
        m = matrices[key]
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        cax = ax.matshow(m, cmap="viridis", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=18)
        ax.set_xlabel(f"{label} {desc}", fontsize=26, labelpad=14)
        ax.tick_params(axis="both", which="both", labelsize=16)
        ax.xaxis.set_ticks_position("bottom")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sentence_analysis(model, tokenizer, sentence, output_dir):
    """Generate per-(layer, head) and per-layer-averaged 2×4 heatmap grids.

    For each layer produces:
      * ``layer_{l:02d}_head_{h:02d}.png`` — one figure per head (2×4 grid).
      * ``layer_{l:02d}_avg.png``          — heads averaged (2×4 grid).
    """
    output_path = Path(output_dir) / "sentence_analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    pos_enc, token_embeddings = get_initial_embeddings(model, inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print(f"Sentence: {sentence!r}")
    print(f"Tokens ({len(tokens)}): {tokens}")

    all_results = run_all_interventions(model, token_embeddings, pos_enc)

    num_layers = len(model.transformer.h)
    num_heads = model.config.n_head
    total = num_layers * (num_heads + 1)   # +1 for the head-avg figure per layer
    saved = 0

    for layer_idx in range(num_layers):
        # all_results[key][layer_idx] has shape [num_heads, seq, seq]

        # Per-head figures
        for head_idx in range(num_heads):
            _save_attention_grid(
                all_results, layer_idx,
                matrices_fn=lambda key, h=head_idx: all_results[key][layer_idx][h],
                title=f"Layer {layer_idx + 1}  ·  Head {head_idx + 1}",
                save_path=output_path / f"layer_{layer_idx + 1:02d}_head_{head_idx + 1:02d}.png",
            )
            saved += 1
            print(f"  [{saved}/{total}] Saved layer_{layer_idx + 1:02d}_head_{head_idx + 1:02d}.png")

        # Head-averaged figure
        _save_attention_grid(
            all_results, layer_idx,
            matrices_fn=lambda key: all_results[key][layer_idx].mean(dim=0),
            title=f"Layer {layer_idx + 1}  ·  Avg over heads",
            save_path=output_path / f"layer_{layer_idx + 1:02d}_avg.png",
        )
        saved += 1
        print(f"  [{saved}/{total}] Saved layer_{layer_idx + 1:02d}_avg.png")

    print(f"\nAll {total} figures saved to {output_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Mode 2 — Dataset analysis (three benchmark datasets, per-dataset + pooled)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_sentences(model, tokenizer, sentences, ds_label, num_layers):
    """Run all interventions on *sentences*, return {key: (mid_scores, all_scores)}."""
    scores_mid = {key: [] for key, *_ in INTERVENTIONS}
    scores_all = {key: [] for key, *_ in INTERVENTIONS}
    n = len(sentences)
    for i, sentence in enumerate(sentences):
        print(f"  [{ds_label} {i + 1}/{n}] {sentence[:80]}...")
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        pos_enc, token_embeddings = get_initial_embeddings(model, inputs)
        results = run_all_interventions(model, token_embeddings, pos_enc)
        for key, *_ in INTERVENTIONS:
            scores_mid[key].append(
                compute_bos_attention_metric(results[key], num_layers, "mid")
            )
            scores_all[key].append(
                compute_bos_attention_metric(results[key], num_layers, "all")
            )
    return scores_mid, scores_all


def _stats_rows(scores, ds_name):
    """Turn per-sentence scores into summary rows for the detailed CSV."""
    rows = []
    for key, _label, desc, _fn in INTERVENTIONS:
        vals = np.array(scores[key])
        mean = float(vals.mean())
        stderr = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append({
            "dataset": ds_name,
            "intervention": key,
            "description": desc,
            "n_examples": len(vals),
            "mean_bos_attention": f"{mean:.6f}",
            "stderr": f"{stderr:.6f}",
            "display": f"{mean:.4f} ± {stderr:.4f}",
        })
    return rows


def dataset_analysis(model, tokenizer, output_dir,
                     sample_size=DEFAULT_SAMPLE_SIZE,
                     cut_length=DEFAULT_CUT_LENGTH,
                     seed=DEFAULT_SEED):
    """Compute BOS-attention metric on three standard benchmarks.

    Datasets: SST-2 (natural language), GSM8K (math), HumanEval (code).
    Each dataset is sampled to *sample_size* examples that have at least
    *cut_length* tokens, then every example is truncated to exactly
    *cut_length* tokens before processing.

    Outputs under ``<output_dir>/dataset_analysis/``:
      sample_manifest.csv                 — exact examples evaluated
      bos_attention_stats_by_dataset.csv  — per-dataset detailed stats
      bos_attention_stats_overall.csv     — pooled across all 3 datasets
      bos_attention_summary_{scope}.{csv,txt}  — pooled readable summaries (mid / all)
      <ds_name>/bos_attention_summary_{scope}.{csv,txt}  — per-dataset readable
    """
    output_path = Path(output_dir) / "dataset_analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    num_layers = len(model.transformer.h)

    sampled, manifest_rows = sample_benchmark_datasets(
        tokenizer, sample_size=sample_size,
        cut_length=cut_length, seed=seed,
    )
    pd.DataFrame(manifest_rows).to_csv(output_path / "sample_manifest.csv", index=False)
    print(f"Manifest saved ({len(manifest_rows)} examples).\n")

    all_mid = {}    # ds_name -> {key: [scores]}
    all_scope = {}  # ds_name -> {key: [scores]}

    for ds_name, sentences in sampled.items():
        print(f"\n{'═'*60}")
        print(f"  Dataset: {ds_name}  ({len(sentences)} examples)")
        print(f"{'═'*60}")
        s_mid, s_all = _run_sentences(model, tokenizer, sentences, ds_name, num_layers)
        all_mid[ds_name] = s_mid
        all_scope[ds_name] = s_all

        ds_dir = output_path / ds_name
        ds_dir.mkdir(exist_ok=True)
        _write_readable_bos_summaries(ds_dir, s_all, "all_layers")
        _write_readable_bos_summaries(ds_dir, s_mid, "mid_layers")

    # Detailed per-dataset CSV
    rows_by_ds = []
    for ds_name in sampled:
        rows_by_ds.extend(_stats_rows(all_mid[ds_name], ds_name))
    pd.DataFrame(rows_by_ds).to_csv(
        output_path / "bos_attention_stats_by_dataset.csv", index=False
    )

    # Pool across datasets
    pooled_mid   = {key: [] for key, *_ in INTERVENTIONS}
    pooled_scope = {key: [] for key, *_ in INTERVENTIONS}
    for ds_name in sampled:
        for key, *_ in INTERVENTIONS:
            pooled_mid[key].extend(all_mid[ds_name][key])
            pooled_scope[key].extend(all_scope[ds_name][key])

    rows_overall = _stats_rows(pooled_mid, "all_datasets")
    df_overall = pd.DataFrame(rows_overall)
    df_overall.to_csv(output_path / "bos_attention_stats_overall.csv", index=False)

    _write_readable_bos_summaries(output_path, pooled_scope, "all_layers")
    _write_readable_bos_summaries(output_path, pooled_mid,   "mid_layers")

    print(f"\n{'═'*60}")
    print("Pooled results across all datasets:")
    print(df_overall[["intervention", "description", "display"]].to_string(index=False))
    print(f"\nAll outputs saved to {output_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Attention intervention experiments on GPT-2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["sentence", "dataset"],
        default="dataset",
        help="'sentence' produces per-(layer,head) heatmap grids; "
             "'dataset' computes BOS-attention statistics on 3 benchmarks.",
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default=DEFAULT_SENTENCE,
        help="Input sentence for sentence mode (ignored in dataset mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Root directory for all outputs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Examples to sample from each benchmark dataset (dataset mode).",
    )
    parser.add_argument(
        "--cut-length",
        type=int,
        default=DEFAULT_CUT_LENGTH,
        help="Token count threshold and truncation target. Examples shorter "
             "than this are discarded; longer ones are truncated to this length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for dataset sampling.",
    )
    args = parser.parse_args()

    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    print("Model loaded.\n")

    if args.mode == "sentence":
        sentence_analysis(model, tokenizer, args.sentence, args.output_dir)
    else:
        dataset_analysis(
            model, tokenizer, args.output_dir,
            sample_size=args.sample_size,
            cut_length=args.cut_length,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
