# Meaningful Information in Retokenized Text

Data and notebooks for experiments measuring how much **content information** a language model extracts from retokenized text — text encoded with a non-canonical BPE segmentation that decodes to the same string.

## Core question

BPE tokenization is not unique: the same text can be segmented many different ways, all decoding to the same string. If a model sees a retokenized version E' of a passage, can it predict the canonical tokenization E0 of that same passage? If so, the model has learned that different segmentations carry the same content — it has achieved **tokenization invariance**.

## Method

### Setup

Let $s$ be a text string. Let $E_0 = (e_1, e_2, \ldots, e_T)$ be its canonical BPE tokenization and $E' = (e'_1, e'_2, \ldots, e'_{T'})$ be a retokenization, where $\text{decode}(E_0) = \text{decode}(E') = s$.

We present the model $M$ with a prompt of the form:

```
The following texts, separated by ----, are identical. {E'} ---- {E0}
```

and measure the per-token log-probability of each token in E0 given everything that precedes it.

### Cumulative information content

For each token $e_t$ in the target sequence, the model assigns a conditional probability $P_M(e_t \mid e_{<t}, \text{prefix})$. The **token-level information content** (surprisal) is:

$$h_t = -\log P_M(e_t \mid e_{<t}, \text{prefix})$$

The **cumulative information content** at position $n$ is:

$$C(n) = \sum_{t=1}^{n} h_t = -\sum_{t=1}^{n} \log P_M(e_t \mid e_{<t}, \text{prefix})$$

This is the quantity plotted in the cumulative $-\log P$ figures. It grows approximately linearly with $n$, and its slope is the entropy rate.

### Entropy rate

The **entropy rate** (cross-entropy per token) is the slope of the cumulative information content:

$$H_M = \frac{C(T)}{T} = \frac{1}{T} \sum_{t=1}^{T} h_t = -\frac{1}{T} \sum_{t=1}^{T} \log P_M(e_t \mid e_{<t}, \text{prefix})$$

We compute this under different prefix conditions:
- $H_M(E_0)$: no prefix (predict E0 cold)
- $H_M(E_0 \mid E')$: retokenized prefix (predict E0 after seeing E')
- $H_M(E_0 \mid E_0)$: canonical prefix (predict E0 after seeing E0)

### Information gain

The **information gain** from the retokenized prefix is:

$$\Delta H = H_M(E_0) - H_M(E_0 \mid E')$$

This measures how many nats per token the model "saves" by having seen the retokenized version first. The **fractional gain**:

$$\frac{\Delta H}{H_M(E_0)} = 1 - \frac{H_M(E_0 \mid E')}{H_M(E_0)}$$

measures what fraction of the model's uncertainty is resolved by the retokenized prefix. A value near 1 means the model extracts nearly all content from the retokenization; a value near 0 means the retokenization is opaque.

### Entropy decomposition

Comparing three prefix conditions decomposes $H_M(E_0)$ into three components:

$$H_M(E_0) = \underbrace{H_M(E_0) - H_M(E_0 \mid E')}_{\text{content resolved}} + \underbrace{H_M(E_0 \mid E') - H_M(E_0 \mid E_0)}_{\text{tokenization uncertainty}} + \underbrace{H_M(E_0 \mid E_0)}_{\text{residual}}$$

- **Content resolved:** the text content that the model successfully extracts from the retokenized prefix.
- **Tokenization uncertainty:** the additional entropy from not knowing which segmentation will follow. After seeing E', the model knows the text but is uncertain whether the next occurrence will use canonical boundaries.
- **Residual:** the model's imperfect ability to copy even a canonical sequence. This is a floor set by the attention mechanism, not by tokenization.

### Theoretical bounds

Since $E_0$ and $E'$ are deterministic functions of the same string $s$, the true mutual information satisfies $I(E_0; E') = H(E_0)$. A perfect model would achieve $H_M(E_0 \mid E') = 0$. Any nonzero conditional entropy represents the model's failure to fully exploit the deterministic relationship between $E'$ and $E_0$.

## Key results

All experiments use **OLMo-2-7B** (AllenAI) with retokenization probability p=1.0, 50 passages, 20 retokenizations each.

### Wikitext (natural language)

| Metric | Value |
|--------|-------|
| Forward information gain (E' → E0) | **92.1%** |
| Reverse information gain (E0 → E') | 22.8% |
| Tokenization uncertainty | 5.2% of H(E0) |
| Residual (copy overhead) | 2.8% of H(E0) |

The model resolves 92% of its uncertainty about the canonical text after seeing a maximally different retokenization. This ability is **entirely learned**: a randomly initialized model gains 0.6%. It emerges during pretraining following an S-curve, with the critical period at steps 10k–100k (~42B–424B tokens).

### Python code (codeparrot-clean)

| Metric | Value |
|--------|-------|
| Forward information gain (E' → E0) | **76.7%** |
| Reverse information gain (E0 → E') | 21.6% |
| Tokenization uncertainty | 18.6% of H(E0) |
| Residual (copy overhead) | 4.4% of H(E0) |

**Tokenization uncertainty is 3.6x higher for code** than for natural language (18.6% vs 5.2%). After seeing retokenized code, the model extracts the content but is far less certain which canonical segmentation was used. This likely reflects more ambiguous token boundaries in code (operators, identifiers, indentation patterns).

### Entropy decomposition

The model's entropy on canonical text H(E0) decomposes into three components:

| Component | Wikitext | Code | Interpretation |
|-----------|----------|------|----------------|
| Content resolved | 92.1% | 77.0% | Text content extracted from retokenized prefix |
| Tokenization uncertainty | 5.2% | 18.6% | Model unsure which segmentation will follow |
| Residual | 2.8% | 4.4% | Imperfect copying even with canonical prefix |

The residual is identical in absolute terms (0.061 nats/tok for both domains), suggesting a fixed cost of the attention-based copy mechanism.

### Forward/reverse asymmetry

The model resolves ~92% of uncertainty in the forward direction (retokenized → canonical) but only ~23% in reverse (canonical → retokenized). This asymmetry is domain-independent and reflects a structural property: the model has strong priors about canonical tokenization but weak priors about which specific retokenization was chosen.

### Training dynamics

Tokenization invariance is learned, not innate. Tracking fractional gain across OLMo-2-7B training checkpoints:

| Training stage | Tokens seen | Gain |
|---------------|-------------|------|
| Random init | 0 | 0.6% |
| Step 1,000 | 5B | 12.5% |
| Step 10,000 | 42B | 24.6% |
| Step 25,000 | 105B | 62.6% |
| Step 101,000 | 424B | 86.8% |
| Step 928,000 | 3,893B | 90.3% |
| Post-training (SFT) | — | 94.0% |

The critical learning period is steps 10k–100k. Post-training (SFT/DPO/Instruct) provides a modest additional boost.

## Repository structure

```
meaningful-information/
  plots_all.ipynb        # Combined notebook for all plots (14 sections)
  wikitext/              # Wikitext experiment data
    experiment.md        # Detailed lab notebook
    example.json         # Single-example per-token logprobs
    example_3conditions.json  # Three-condition trajectories (same passage)
    batch_p1p0.json      # Batch data (50 passages x 20 retoks x 3 p-values)
    controls_base.json   # Control conditions (retok, random_text, shuffled)
    random_init.json     # Random initialization baseline
    ckpt_step*.json      # Training checkpoint data (11 checkpoints)
    posttrain_*.json     # Post-training models (SFT, DPO, Instruct)
    reverse_base.json    # Reverse direction (predict E' from E0)
    canon_prefix.json    # Canonical prefix (entropy decomposition)
    retok_retok.json     # Retok-retok (full matrix)
  code/                  # Python code experiment data
    experiment.md        # Lab notebook
    controls_base.json   # Control conditions
    reverse_base.json    # Reverse direction
    canon_prefix.json    # Canonical prefix (entropy decomposition)
    retok_retok.json     # Retok-retok (full matrix)
    examples.json        # Per-token logprobs for cumulative plots
```

## Generating plots

Open `plots_all.ipynb` in Jupyter. Run the first cell (imports), then any section:

- **Part I (sections 1–8):** Wikitext plots — single example, tokenization visualization, batch slopes, training sweep, forward/reverse, entropy decomposition, full matrix, three-condition trajectories
- **Part II (sections 9–12):** Code plots — controls, forward/reverse, decomposition, full matrix
- **Part III (sections 13–14):** Cross-domain comparison — side-by-side decomposition, cumulative logprob trajectories

## Model and data

- **Model:** [OLMo-2-7B](https://huggingface.co/allenai/OLMo-2-1124-7B) (AllenAI, November 2024 release)
- **Natural language data:** [Wikitext-103](https://huggingface.co/datasets/wikitext) (test split)
- **Code data:** [codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) (Python, deduplicated)
- **Retokenization:** BPE boundary perturbation with probability p, using all valid segmentations of each merge

## Citation

This work is part of a paper on segmentation sampling. Citation details forthcoming.
