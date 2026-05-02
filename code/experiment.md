# Meaningful Information in Retokenized Code

Repeating the meaningful information experiments from `meaningful-information/` (wikitext) on **Python code** from The Stack (deduplicated).

## Motivation

Natural language and code have very different statistical properties:
- Code is more repetitive (boilerplate, imports, indentation)
- Code has rigid syntactic structure (brackets, keywords, whitespace patterns)
- Code tokens may have less ambiguity in segmentation (e.g., single-character operators)

The question: does the entropy decomposition look different for code?

## Dataset

**codeparrot-clean** — `codeparrot/codeparrot-clean`, 5.3M deduplicated Python files from GitHub.
- Publicly accessible (no authentication needed)
- Streaming mode with 5% sub-sampling for diversity
- Same passage parameters as wikitext: 50 passages, 30–150 tokens each

## Experiments

All experiments use OLMo-2-7B (base), p=1.0, 50 passages × 20 retokenizations.

### Phase 1: Controls (base model)

**Script:** `code/cumulative_logprob_code_conditions.py`
**Output:** `controls_base.json`

Conditions:
- `retok` — retokenized prefix of the same code
- `random_text` — a different code file as prefix
- `shuffled` — retokenized tokens in shuffled order
- `no_prefix` — canonical code alone (baseline)

### Phase 2: Reverse direction

**Script:** `code/cumulative_logprob_code_reverse.py`
**Output:** `reverse_base.json`

Predict retokenized code given canonical prefix. Measures the forward/reverse asymmetry for code.

### Phase 3: Canonical prefix (entropy decomposition)

**Script:** `code/cumulative_logprob_code_canon_prefix.py`
**Output:** `canon_prefix.json`

Three conditions:
- E0 prefix → E0 target (canonical copy)
- E' prefix → E0 target (retokenized prefix)
- No prefix → E0 target (baseline)

Decomposes entropy into: content resolved + tokenization uncertainty + residual.

### Phase 4: Retok-retok (full matrix)

**Script:** `code/cumulative_logprob_code_retok_retok.py`
**Output:** `retok_retok.json`

Three conditions:
- E'1 → E'2 (different retokenizations)
- E'1 → E'1 (same retokenization repeated)
- E'2 cold (no prefix)

Completes the full prefix × target entropy matrix.

## Running

Combined SLURM script runs all 4 phases sequentially (~3 hours total):

```bash
cd /path/to/tokenization-project
sbatch code/slurm_code_experiments.sh
```

Results go to `/scratch/tcan/tokenization-project/results/meaningful_info_code/`.

After copying results to this directory, generate plots with `plots.ipynb`.

## File index

| File | Description |
|------|-------------|
| `experiment.md` | This lab notebook |
| `plots.ipynb` | Jupyter notebook for all plots |
| `controls_base.json` | Phase 1: controls (retok, random_text, shuffled) |
| `reverse_base.json` | Phase 2: reverse direction |
| `canon_prefix.json` | Phase 3: entropy decomposition |
| `retok_retok.json` | Phase 4: retok-retok full matrix |
| `controls.png` | Controls violin plot |
| `forward_vs_reverse.png` | Forward vs reverse comparison (3-panel) |
| `canon_vs_retok_prefix.png` | Entropy decomposition (violin + stacked bar) |
| `full_matrix.png` | Complete prefix × target matrix |
| `code_vs_wikitext.png` | Side-by-side decomposition comparison |

## Results

Cluster job 489510 completed 2026-04-25 on L4 GPU (l4-4 partition). All 4 phases ran in ~12 minutes.

### Phase 1: Controls

| Condition | Mean entropy rate (nats/tok) |
|-----------|------------------------------|
| retok (same code) | 0.326 |
| random text (different file) | 1.264 |
| shuffled (scrambled order) | 1.291 |
| no prefix (baseline) | 1.398 |

**Forward information gain: 76.7%** — the model extracts substantial content from retokenized code, but less than from retokenized prose (92.1% for wikitext).

The controls behave as expected: random text and shuffled tokens provide minimal benefit over no prefix, confirming the information gain is specific to seeing the same content.

### Phase 2: Forward vs reverse

| Direction | With prefix | No prefix | Info gain |
|-----------|-------------|-----------|-----------|
| Forward (predict E0 from E') | 0.326 nats/canon_tok | 1.398 nats/canon_tok | 76.7% |
| Reverse (predict E' from E0) | 4.916 nats/retok_tok | 6.270 nats/retok_tok | 21.6% |

The forward/reverse asymmetry is nearly identical to wikitext (21.6% vs 22.8% reverse gain). This is a domain-independent structural property: the model has strong priors about canonical tokenization but weak priors about which alternative segmentation was used.

### Phase 3: Entropy decomposition

| Component | Code | Code (%) | Wikitext | Wikitext (%) |
|-----------|------|----------|----------|--------------|
| Total H(E0) | 1.398 | 100% | 2.210 | 100% |
| Content resolved | 1.076 | 77.0% | 2.035 | 92.1% |
| Tokenization uncertainty | 0.260 | **18.6%** | 0.114 | 5.2% |
| Residual | 0.061 | 4.4% | 0.061 | 2.8% |

**Tokenization uncertainty is 3.6× higher for code** (18.6% vs 5.2%). This is the headline finding: after seeing retokenized code, the model is far less certain which canonical segmentation was used. This likely reflects:
- More ambiguous token boundaries in code (identifiers like `my_var`, mixed-case `getData`, numeric literals)
- Frequent single-character tokens (operators, brackets, semicolons) that create more valid segmentation alternatives
- Less contextual constraint on boundary placement compared to natural language word boundaries

The residual (canonical copy imperfection) is identical in absolute terms (0.061 nats/tok for both), suggesting this reflects a fixed overhead of the copy mechanism rather than a domain-dependent quantity.

### Phase 4: Full prefix × target matrix

| | → E0 (canonical) | → E'same (same retok) | → E'diff (diff retok) |
|---|---|---|---|
| **E0 prefix** | 0.06 | — | 4.92 |
| **E' prefix** | 0.32 | 0.93 | 4.33 |
| **no prefix** | 1.40 | — | 6.27 |

Notable differences from wikitext:
- E'→E'same is 0.93 nats/tok (vs 1.26 for wikitext) — the model copies retokenized code slightly better than retokenized prose
- E0→E0 is 0.06 (essentially the same as wikitext at 0.06) — canonical copying is equally easy
- E'→E0 gap from E0→E0 is much larger for code (0.26 vs 0.11) — consistent with higher tokenization uncertainty

### Summary of key findings

1. **Code has 3.6× higher tokenization uncertainty** than natural language (18.6% vs 5.2%). The model is much less certain about segmentation boundaries in code.
2. **Forward information gain is lower for code** (76.7% vs 92.1%), entirely driven by the higher tokenization uncertainty — the content extraction capability is similar.
3. **The forward/reverse asymmetry is domain-independent** (~22% reverse gain for both code and prose).
4. **Code has lower baseline entropy** (1.40 vs 2.21 nats/tok), confirming code is more predictable than prose.
5. **The residual (copy overhead) is identical** at 0.061 nats/tok for both domains, suggesting a fixed cost of the attention-based copy mechanism.
