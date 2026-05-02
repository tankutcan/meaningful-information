# Meaningful Information in Retokenized Text — Lab Notebook

## Core Question

A retokenization E' of a text s is a different sequence of tokens that decodes to the same string as the canonical tokenization E0. Does a language model treat E' as carrying the same semantic content as E0? Specifically: if the model sees E' first, can it predict E0 better than without any context?

## Information-Theoretic Framework

### Setup

Let $s$ be a text string. Let $E_0 = (e_1, e_2, \ldots, e_T)$ be the canonical BPE tokenization and $E' = (e'_1, e'_2, \ldots, e'_{T'})$ be a retokenized version, where $\text{decode}(E_0) = \text{decode}(E') = s$.

We consider a language model $M$ and define:

- $H_M(E_0)$: the model's cross-entropy on the canonical sequence, i.e., the average $-\log P_M(e_t | e_{<t})$ per token
- $H_M(E_0 | E')$: the model's cross-entropy on the canonical sequence _given the retokenized prefix_, i.e., the average $-\log P_M(e_t | E', \text{sep}, e_{<t})$ per token

The **information gain** from the retokenized prefix is:

$$\Delta H = H_M(E_0) - H_M(E_0 | E')$$

This measures how many nats per token the model "saves" by having seen the retokenized version first.

### Interpretation

$\Delta H$ quantifies the **mutual information** between $E'$ and $E_0$ as estimated by the model. In information-theoretic terms:

$$I_M(E_0; E') \approx T \cdot \Delta H$$

where $T = |E_0|$. This is the total information (in nats) that the retokenized text provides about the canonical text, as measured through the lens of the model's predictions.

### Key quantities

1. **$H_M(E_0)$** — the unconditional entropy rate. This reflects the model's uncertainty about the text without any context. It should approximate the entropy rate of the language for well-trained models (~1.5–2.5 nats/token for English Wikipedia).

2. **$H_M(E_0 | E')$** — the conditional entropy rate. If the model perfectly understands that $E'$ encodes the same text as $E_0$, this should approach zero (the model knows exactly what's coming). If the model treats $E'$ as noise, this should equal $H_M(E_0)$.

3. **$\Delta H / H_M(E_0)$** — the fractional information gain. This is the fraction of the model's uncertainty that is resolved by the retokenized prefix. A value near 1 means the model extracts nearly all the content from the retokenization; a value near 0 means the retokenization is opaque.

### What these quantities tell us about tokenization invariance

The central claim of our paper is that the canonical BPE tokenization is an arbitrary convention — the same text can be segmented many ways, and the model's internal representation should (ideally) be invariant to the choice of segmentation. The information gain $\Delta H$ tests this directly:

- **If $\Delta H \approx H_M(E_0)$:** The model can fully decode the retokenized text. It has learned that different segmentations are equivalent, and its internal representation is (approximately) tokenization-invariant.

- **If $\Delta H \approx 0$:** The model cannot extract information from the retokenization. Each segmentation is processed as an independent, opaque sequence.

- **If $0 < \Delta H < H_M(E_0)$:** The model extracts partial information. This could indicate that it recognizes some sub-word patterns across segmentations but not all, or that the prompt framing is insufficient to trigger full cross-segmentation understanding.

### Connection to KL divergence experiments

Our earlier KL divergence experiments measured $D_{KL}(P(next|E_0) \| P(next|E'))$ — the divergence in next-token predictions at a single position after the entire context. This measured sensitivity in a different way: how much the model's predictions change when the input segmentation changes.

The information gain experiment is complementary:
- KL divergence asks: "does the model give the _same_ predictions from different segmentations?"
- Information gain asks: "does the model understand that different segmentations carry the _same content_?"

These are related but distinct. A model could have high KL (different predictions from different segmentations) but also high $\Delta H$ (it can still extract the content from a retokenization when explicitly prompted to). The KL experiment was confounded by entropy scaling; the information gain experiment avoids this by measuring reduction in uncertainty rather than absolute divergence.

### Dependence on retokenization probability p

The retokenization probability $p$ controls how different $E'$ is from $E_0$. At $p = 0$, $E' = E_0$ and $\Delta H = H_M(E_0)$ trivially (the model sees the exact same text twice). At $p = 1.0$, $E'$ is maximally different from $E_0$ while still decoding to the same string.

We expect:
- $\Delta H$ should decrease with $p$ (more scrambled segmentation → harder to extract content)
- But even at $p = 1.0$, if the model has learned robust sub-word representations, $\Delta H$ should remain substantial

The ratio $\Delta H(p) / \Delta H(0)$ would measure how much information is lost as the segmentation diverges further from canonical.

### Theoretical bounds

The mutual information $I(E_0; E')$ is bounded by $H(E_0)$: the retokenized text cannot provide more information about $E_0$ than $E_0$ contains. Since $E_0$ and $E'$ are deterministic functions of the same string $s$, the _true_ mutual information (not as estimated by the model) satisfies:

$$I(E_0; E') = H(E_0) = H(E')$$

because knowing $E'$ determines $s$ determines $E_0$, and vice versa. So the "ideal" model should achieve $\Delta H = H_M(E_0)$, i.e., the conditional entropy rate should be zero. Any nonzero $H_M(E_0 | E')$ represents the model's failure to fully exploit the deterministic relationship between $E'$ and $E_0$.

This gives us a clean metric: the **tokenization invariance gap** is $H_M(E_0 | E') / H_M(E_0)$, the fraction of uncertainty that the model _fails_ to resolve despite having all the information available.

---

## Phase 1: Single-example proof of concept (2026-04-21)

**Setup:** Single wikitext passage (100 tokens), OLMo-2-7B base, p=1.0.

**Result:** The model dramatically reduces its uncertainty when given the retokenized prefix.
- $H_M(E_0) \approx 1.98$ nats/token
- $H_M(E_0 | E') \approx 0.20$ nats/token
- $\Delta H \approx 1.78$ nats/token
- Fractional gain: $\Delta H / H_M(E_0) \approx 90\%$

The model resolves ~90% of its uncertainty from the retokenized prefix, even at p=1.0 (maximum segmentation distance). This confirms that OLMo-2-7B has learned substantial tokenization invariance — it can decode the content of a maximally different segmentation.

**Files:** `example.json`, `cumulative_logprob_example.png`, `tokenization_visual.png`

---

## Phase 2: Batch statistics (2026-04-21)

**Setup:** 50 wikitext passages × 20 retokenizations × 3 p-values (0.1, 0.5, 1.0). OLMo-2-7B base.

**Script:** `code/cumulative_logprob_batch.py`
**SLURM:** `code/slurm_cumlogprob_batch.sh` — job 487278
**Output:** `/scratch/tcan/tokenization-project/results/meaningful_info/batch_p1p0.json`

**Status:** Completed 2026-04-22.

**Results:**

The retokenized prefix resolves 86–89% of the model's uncertainty across all 50 passages:

| Condition | Mean slope (nats/tok) | Median | Fraction resolved |
|-----------|----------------------|--------|-------------------|
| No prefix | 2.194 | 2.230 | — |
| p=0.1 | 0.237 | 0.192 | 89% |
| p=0.5 | 0.279 | 0.221 | 87% |
| p=1.0 | 0.307 | 0.245 | 86% |

Key observations from the four-panel plot (`batch_slopes.png`):

- **A) Per-passage slope_a is tightly distributed across retokenizations.** For a given passage, different retokenizations yield similar entropy rates. The variance comes from passage difficulty, not retokenization randomness.
- **B) Two completely separated populations.** slope_b (no prefix) and slope_a (with prefix) have essentially no overlap, regardless of p.
- **C) No correlation between slope_a and slope_b.** Hard passages (high unconditional entropy) are resolved just as well as easy ones. The model's ability to decode retokenized text doesn't depend on the difficulty of the underlying text.
- **D) The gap between p values is small.** Moving from p=0.1 to p=1.0 only increases slope_a from 0.237 to 0.307 — the model loses only ~3% more information even when the segmentation is maximally different.

**Interpretation:** OLMo-2-7B has learned near-complete tokenization invariance in this prompting setup. The residual 0.2–0.3 nats/tok likely reflects the model's imperfect ability to maintain a bijective character-level mapping across segmentation boundaries, not a failure to understand the content.

---

## Phase 3: Controls, random init, and training checkpoint sweep (2026-04-22)

**Script:** `code/cumulative_logprob_conditions.py`
**SLURM:** `code/slurm_logprob_all.sh` — job 487309
**Plot:** `all_conditions.png`

### 3a) Control conditions (base model, p=1.0)

Three conditions compared to the retokenized prefix:

| Condition | Mean slope (nats/tok) | Median | vs no-prefix |
|-----------|----------------------|--------|--------------|
| retok (same text) | 0.176 | 0.140 | 92% reduction |
| random text (different passage) | 2.156 | 2.170 | ~0% reduction |
| shuffled (scrambled token order) | 2.141 | 2.121 | ~2% reduction |
| no prefix | 2.194 | 2.230 | — |

**Key finding:** The information gain is entirely from the _sequential structure_ of the retokenized text. A random passage provides zero benefit. Shuffled tokens (correct vocabulary items, wrong order) provide essentially zero benefit. The model must reconstruct the character-level sequence from the retokenized token order.

### 3b) Random initialization

| Condition | slope_a | slope_b | Gain |
|-----------|---------|---------|------|
| Random init, with retok prefix | 12.157 | 12.234 | 0.6% |
| Base model, with retok prefix | 0.176 | 2.194 | 92% |

A randomly initialized model derives **zero information** from the retokenized prefix. Both slopes are ~12 nats/tok (near log(vocab_size) — essentially random predictions). The ability to decode retokenized text is entirely _learned_.

### 3c) Training checkpoint sweep

| Step | Tokens | slope_a | slope_b | ΔH/H(E0) |
|------|--------|---------|---------|-----------|
| 0 (random) | 0 | 12.157 | 12.234 | 0.6% |
| 1,000 | 5B | 4.655 | 5.317 | 12.5% |
| 5,000 | 21B | 2.802 | 3.277 | 14.5% |
| 10,000 | 42B | 2.292 | 3.040 | 24.6% |
| 25,000 | 105B | 1.059 | 2.830 | 62.6% |
| 50,000 | 210B | 0.670 | 2.755 | 75.7% |
| 101,000 | 424B | 0.347 | 2.626 | 86.8% |
| 200,000 | 839B | 0.354 | 2.667 | 86.7% |
| 400,000 | 1678B | 0.288 | 2.574 | 88.8% |
| 600,000 | 2517B | 0.264 | 2.553 | 89.7% |
| 800,000 | 3356B | 0.226 | 2.499 | 91.0% |
| 928,000 | 3893B | 0.238 | 2.453 | 90.3% |

**Key findings:**

1. **Tokenization invariance is learned, not innate.** The fractional gain rises from 0.6% (random init) to 91% (late training), following an S-shaped curve.

2. **The critical period is steps 10k–100k (42B–424B tokens).** The gain jumps from 25% to 87% in this window — roughly one order of magnitude of training. Before step 10k, the model has only rudimentary invariance; after step 100k, it's nearly saturated.

3. **Invariance saturates around 90%.** From step 101k onward, the gain plateaus at 87–91%. The residual ~10% may represent an inherent limitation: the prompt framing doesn't perfectly communicate the bijection, or the model's attention mechanism can't perfectly align tokens across different segmentations.

4. **The unconditional entropy rate (slope_b) also drops during training** — from 12.2 (random) to 2.5 (trained) — but this is just the model learning language. The _conditional_ entropy rate (slope_a) drops much faster, from 12.2 to 0.24, showing that invariance is learned on top of language understanding.

5. **This resolves the KL puzzle.** The earlier KL divergence experiments showed flat KL across training. That metric was confounded by the scaling of prediction entropy. The information gain metric cleanly separates "how confident is the model" from "how invariant is it to tokenization" and reveals a dramatic learning curve that was hidden in the raw KL.

---

## Phase 3.5: Post-training models (pending cluster)

**Goal:** Compare SFT, DPO, and Instruct models to the base model. Does post-training improve or degrade tokenization invariance?

**Script:** `code/slurm_logprob_posttrain.sh` — job 487421
**Status:** Completed 2026-04-22.

**Results:**

| Model | slope_a | slope_b | ΔH/H(E0) |
|-------|---------|---------|-----------|
| Base | 0.176 | 2.194 | 92.0% |
| SFT | 0.146 | 2.421 | 94.0% |
| DPO | 0.178 | 2.577 | 93.1% |
| Instruct | 0.185 | 2.599 | 92.9% |

Post-training slightly improves invariance — SFT reaches 94%, the highest. The unconditional entropy rate increases after RLHF (the model becomes less confident on generic Wikipedia), but the conditional rate stays flat, so fractional gain ticks up. The effect is modest: post-training doesn't substantially change the tokenization invariance learned during pretraining.

---

## Phase 4: Reverse direction — predict E' from E0 (pending cluster)

**Goal:** Measure $H_M(E' | E_0)$ — how well the model predicts retokenized text given the canonical version as context. This is the other direction of the bijection from Phase 2.

**Setup:** "The following texts, separated by ----, are identical. {E0} ---- {E'}" — now E0 is the prefix and E' is the prediction target. Also compute baseline: predict E' with no prefix.

**Design decisions:**
- Report slopes in **both** nats/retok_token and nats/canon_token for comparability
- The baseline (predicting E' cold) is itself interesting: the model was trained on canonical tokenizations, so retokenized sequences are out-of-distribution. The baseline slope_b for E' could be much higher than for E0.

**Asymmetry to watch for:**
- Forward: composing fragments into known tokens (easy — the model has seen these tokens)
- Reverse: decomposing known tokens into sub-word fragments (harder — requires generating tokens the model rarely produces in training)
- E' has ~2× more tokens than E0, so cumulative -logprob spans a longer sequence

**Script:** `code/cumulative_logprob_reverse.py`
**SLURM:** `code/slurm_logprob_reverse.sh` — job 487393
**Status:** Completed 2026-04-22.

**Results:**

| Direction | slope w/ prefix | slope no prefix | Gain |
|-----------|----------------|-----------------|------|
| Forward (predict E0 from E') | 0.18 nats/canon_tok | 2.19 nats/canon_tok | **92%** |
| Reverse (predict E' from E0) | 5.57 nats/retok_tok | 7.22 nats/retok_tok | **23%** |
| Reverse (per canon tok) | 12.29 nats/canon_tok | 15.93 nats/canon_tok | **23%** |

**Key findings:**

1. **Strong asymmetry.** The forward direction resolves 92% of uncertainty; the reverse only 23%. The model can compose sub-word fragments into canonical tokens very well, but decomposing canonical tokens into the specific retokenized fragments is much harder.

2. **Retokenized sequences are out-of-distribution.** The baseline for predicting E' cold is 7.2 nats/retok_tok — far higher than the 2.2 nats/canon_tok for predicting E0 cold. The model has never seen retokenized sequences during training, so they are fundamentally harder to predict.

3. **Information-theoretically consistent.** $H(E' | E_0) > H(E_0 | E')$ because E' carries entropy beyond the text content: the randomness of the segmentation choice itself. Even a perfect model that understood E0 fully would need to guess which specific segmentation was chosen. The 23% gain comes from the model learning which decompositions are structurally valid, not from predicting the exact one.

**Plot:** `forward_vs_reverse.png`

---

## Phase 5: Canonical prefix — isolating tokenization uncertainty (2026-04-22)

**Goal:** By comparing H_M(E0|E0) with H_M(E0|E'), isolate the tokenization uncertainty component. If the model perfectly extracts the text from E' but is unsure whether the second occurrence will be canonical, the difference H_M(E0|E') - H_M(E0|E0) measures the pure tokenization entropy.

**Script:** `code/cumulative_logprob_canon_prefix.py` — job 488235
**Plot:** `canon_vs_retok_prefix.png`

**Results:**

| Condition | Mean slope (nats/tok) | Gain |
|-----------|-----------------------|------|
| No prefix | 2.194 | — |
| Canonical prefix (E0 → E0) | 0.061 | 97.2% |
| Retokenized prefix (E' → E0) | 0.174 | 92.1% |

**Entropy decomposition of H_M(E0):**

| Component | nats/tok | Share | Interpretation |
|-----------|---------|-------|----------------|
| Content resolved | 2.020 | 92.1% | Text extracted from E' |
| Tokenization uncertainty | 0.113 | 5.2% | Model hedging on segmentation after seeing E' |
| Residual | 0.061 | 2.8% | Imperfect repetition even with E0 prefix |

**Key insight:** The residual H_M(E0|E') is dominated by the model's knowledge of the text content (92%), with tokenization uncertainty contributing only 5.2%. Having seen a retokenized prefix, the model shifts its posterior slightly toward non-canonical segmentations, adding 0.113 nats/tok of entropy. But this is tiny compared to the 2.21 nats/tok that would result from uniform uncertainty over valid segmentations — the model has a very strong prior for canonical tokenization even after seeing E'.

The 0.061 nats/tok residual with canonical prefix is a floor set by the model's imperfect ability to copy sequences, not by tokenization uncertainty.

---

## Future experiments to consider

1. **Segmentation distance dependence:** Plot $\Delta H$ vs $d_{seg}$ and vs $|E'|/|E_0|$. Does the information gain degrade smoothly with segmentation distance, or is there a threshold?

2. **Position-resolved analysis:** Does the model gain more information about early vs late tokens in $E_0$? The retokenized prefix is processed sequentially, so the model might extract more from tokens that correspond to the beginning of $E'$.

3. **Different models/architectures:** Compare across model families (GPT-2, Pythia, LLaMA, OLMo) to see whether tokenization invariance is a universal learned property or architecture-dependent.

4. **Scaling with model size:** Does larger model size → better tokenization invariance? Compare OLMo-2-1B, 7B, 13B.

5. **Without the instruction prompt:** Just concatenate E' and E0 without the "these texts are identical" framing. How much does explicit prompting matter vs the model's intrinsic invariance?

6. **Longer sequences:** Does the information gain hold for passages of 500+ tokens, or does the model's ability to maintain the bijection degrade with length?

---

## File index

| File | Description |
|------|-------------|
| `experiment.md` | This lab notebook |
| `example.json` | Phase 1 single-example data |
| `cumulative_logprob.py` | Phase 1 local script (moved to code/) |
| `cumulative_logprob_example.png` | Phase 1 cumulative -logprob plot |
| `tokenization_visual.png` | Canonical vs retokenized segmentation visualization |
| `plot_cumulative_logprob.py` | Phase 1 plotting script |
| `plot_tokenization_visual.py` | Segmentation visualization script |
| `batch_p1p0.json` | Phase 2 batch data (50 passages × 20 retoks × 3 p-values) |
| `plot_batch_slopes.py` | Phase 2 plotting script |
| `batch_slopes.png` | Phase 2 four-panel results |
| `controls_base.json` | Phase 3 controls data (retok, random_text, shuffled) |
| `random_init.json` | Phase 3 random-init model data |
| `ckpt_step*.json` | Phase 3 training checkpoint data (11 checkpoints) |
| `plot_all_conditions.py` | Phase 3 plotting script |
| `all_conditions.png` | Phase 3 four-panel results (controls + training curve) |
| `posttrain_sft.json` | Phase 3.5 SFT model data |
| `posttrain_dpo.json` | Phase 3.5 DPO model data |
| `posttrain_instruct.json` | Phase 3.5 Instruct model data |
| `reverse_base.json` | Phase 4 reverse direction data |
| `plot_forward_vs_reverse.py` | Phase 4 plotting script |
| `forward_vs_reverse.png` | Phase 4 forward vs reverse comparison |
| `plots.ipynb` | Interactive Jupyter notebook with all plots |
