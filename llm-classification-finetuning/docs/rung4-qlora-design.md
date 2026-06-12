# Rung 4: QLoRA-Finetuned LLM — Design & Learning Notes

*The modern recipe: how a 9B-parameter model trains on a single Kaggle GPU.*

## 1. Why jump from linear models to a 9B LLM?

Rungs 1–2 read *style* (structure, lexical statistics). But preference judging often
hinges on **semantics**: which answer is actually correct, which one understood the
question, which hedge is appropriate. Only a model that *reads* both responses with
real language understanding can score those. The 2024 winning solutions were all
finetuned 7–9B LLMs; nothing smaller was competitive past ~0.95 log loss.

The puzzle this rung teaches: a 9B model in fp16 needs ~18GB for weights alone —
training it naively needs 4× that (gradients + optimizer states). Kaggle gives you a
16GB T4 (or 2). The whole rung is about closing that gap with two ideas.

## 2. Idea one: LoRA — train 0.5% of the parameters

Finetuning doesn't need to move every weight. **LoRA** (Low-Rank Adaptation) freezes
the pretrained weight matrix `W` and learns a low-rank update:

```
W' = W + (α/r) · B·A      A: r×d, B: d×r, r « d  (r = 8–64 typical)
```

Intuition: task adaptation lives in a low-dimensional subspace; you don't need a
full-rank correction to steer a pretrained model. Consequences:

- **Trainable params collapse** from 9B to tens of millions (the A/B pairs on
  attention + MLP projections). Gradient + optimizer memory shrinks proportionally.
- **The base model is untouched** — adapters are ~100MB files; swap per task.
- `r` (rank) and `α` (scaling) are the knobs: more rank = more capacity, more memory,
  more overfitting risk. r=16–32 is the sweet spot for classification heads.

## 3. Idea two: quantization — store the frozen part in 4 bits

LoRA killed the gradient memory, but the frozen weights still need ~18GB in fp16.
**QLoRA** stores them in **4-bit NF4** (NormalFloat — a quantization grid matched to
the bell-curve distribution of neural weights), cutting weights to ~5GB. During the
forward pass, blocks dequantize to bf16 just-in-time for the matmul; gradients flow
through the dequantized values into the (full-precision) LoRA adapters.

Memory ledger for Gemma-2-9b on a 16GB T4:

| Component | fp16 full FT | QLoRA |
|---|---|---|
| weights | 18 GB | ~5 GB (NF4 + double quant) |
| gradients | 18 GB | ~0.1 GB (LoRA only) |
| optimizer (Adam) | 36 GB | ~0.2 GB |
| activations | several GB | ~2–4 GB (checkpointing + short seqs) |
| **total** | **~75 GB** ❌ | **~8–10 GB** ✅ |

Two supporting tricks you'll see in every recipe: **gradient checkpointing** (recompute
activations in backward instead of storing — trade ~30% speed for huge memory) and
**paged optimizers** (spill optimizer states to CPU RAM on spikes).

## 4. Task framing: classification head, not generation

We don't ask the model to *write* "A is better". We feed one packed sequence —

```
<prompt> [SEP] <response_a> [SEP] <response_b>
```

— take the last hidden state, and attach a 3-way classification head trained with
cross-entropy (= log loss, our exact metric). Generation-style answers waste tokens
and calibrate terribly; a head gives clean probabilities.

Critical engineering details (each worth real log-loss):

1. **Truncation strategy.** Sequences must fit ~1.5–2k tokens. Naive head-truncation
   deletes whole responses. Budget per part (e.g., prompt 256, each response 768) and
   truncate *each part from its middle* (keep head + tail — answers open and close
   with their most characteristic content).
2. **Position-swap consistency.** Train on both (A,B) and (B,A) orderings (label
   swapped accordingly) — model learns position-invariance instead of inheriting the
   dataset's position bias. At inference, average the two orderings (TTA): the bias
   cancels by construction (see BRIEFING.md).
3. **Calibration.** Check reliability after training; a single temperature scalar
   fitted on a holdout often buys 0.005–0.01 log loss.

## 5. Kaggle execution plan

Constraint: comp kernels run **offline** — model weights must come from a Kaggle
Dataset/Model, not the internet. Two-kernel pattern:

- **Train kernel** (GPU, internet ON, *not* attached to comp): pull Gemma-2-9b (or
  Llama-3.1-8B) from Kaggle Models hub, QLoRA-finetune on train.csv (1 epoch, bf16,
  packed seqs), save adapter + tokenizer + temperature as a Kaggle Dataset output.
- **Inference kernel** (GPU, internet OFF, attached to comp): load base from Kaggle
  Models + adapter from our Dataset, batch-predict with position-swap TTA, write
  submission.csv.

Expected: **~0.92–0.96** single model (winners' ensembles reached ~0.87–0.88).
Train time: several hours on T4×2 — budget one Kaggle GPU session, subset to ~30–40k
rows if needed.

## 6. Interview vocabulary from this rung

PEFT, LoRA rank/alpha, NF4/double quantization, paged optimizer, gradient
checkpointing, sequence packing, truncation strategy, position-swap TTA, temperature
scaling, classification head vs generative scoring, offline-inference constraints.
