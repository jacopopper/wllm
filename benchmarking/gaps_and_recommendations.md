# Gaps & General Recommendations for wllm

**Synthesized from analysis of top white-box UQ / hallucination detection methods.**

This document lists **method-agnostic** improvements. No paper or method names appear in API proposals.

## Probability / Logprob Surface

Current state:
- When `extract.logprobs` requested: per-token `logprob` (of the chosen token) + `top_logprobs` list + optional approximate entropy (renormalized over top-k).
- `raw_logits: true` and exact full-distribution entropy are explicitly rejected (capability = unsupported).
- Top-k is bounded by server `--max-top-k` (default 50).

Gaps observed across methods:
- Many simple but powerful scores are built directly from the *sequence of chosen token probabilities* (min, product for seq prob, length-normalized geometric mean, margins).
- Better uncertainty estimates (negentropy, calibrated margins) benefit from either higher top-k or access to more of the distribution.
- Exact (or higher-fidelity) entropy is frequently desired or used as a feature.

Recommended general enhancements:
- Always surface the chosen token's logprob/probability reliably and conveniently (even for `tokens: true` without full logprobs spec, or as a top-level convenience).
- Raise or make configurable the practical top-k limit for research workloads (or support "all returned by backend" with clear warnings).
- Provide (or document) helpers for common aggregates: chosen token probs array, sequence log-prob, length-normalized score, renormalized top-k entropy.
- Consider safe, opt-in ways to obtain better distribution statistics without exposing the full vocabulary (future vLLM surface or approximations).
- Keep capability reporting accurate; never fake full logits.

## Hidden States Capture

Current state:
- Flexible selectors for layers ("middle", "middle_third", ints, lists, negative) and positions ("prompt", "generated", "last", "last_generated", ints...).
- Optional pooling (`mean`/`max`/`last`/null).
- Two modes: replay (isolated vLLM pooling runner) and online (eager generation runner).
- Captured at transformer block output.
- Limits + byte caps; artifact-backed for larger requests.
- Requires `tensor_parallel_size=1` + typically low `gpu_memory_utilization` for replay.

Gaps:
- Probing-style methods often want specific positions (last token of prompt / last generated token) at one or more layers with no pooling (raw vectors).
- Spectral methods (covariance over tokens or over samples) need matrices of per-token (or per-sample last-token) hidden states, often at mid-to-late layers.
- "Prompt-only" or "statement-only" extraction (no generation) is valuable for pre-answer truthfulness probes.
- More internal sites (post-attention, post-FFN, residual contributions) are used in some advanced analyses.
- Capturing many layers or long sequences is expensive in replay (VRAM) and in payload size.
- Online vs replay numerical differences must be understood by users doing precise work.

Recommended general enhancements:
- First-class support / conveniences for "last prompt token", "last generated token" (and prompt-only extraction paths).
- Easy extraction of unpooled per-position hidden states for selected layers/positions (artifact-friendly by default for research).
- Support (or clearly document) requesting hidden states for the prompt alone.
- Consider additional capture sites if they can be added cleanly behind the compat layer.
- Improve ergonomics for "research matrix" extraction: helpers that return (num_selected_positions, hidden_dim) arrays ready for covariance, probing, etc.
- Better guidance and preflight around replay costs and when online is appropriate.
- Richer per-tensor metadata (capture site, exact semantics).

## Attention Weights

Current state:
- Experimental, behind `--enable-attention-weights`.
- Separate Transformers replay model (`output_attentions=True`, `use_cache=False`) after vLLM generation.
- Selectors for layers, heads ("all" or list), query positions, key positions (including `previous_token`).
- Artifact-recommended because quadratic in sequence length.
- Not from the active vLLM fused path.

Gaps:
- Attention-pattern methods rely on precise head- and token-relationship patterns (e.g., previous-token attention in specific heads).
- Replay cost and potential numerical mismatch with vLLM internals make it heavy for routine use.
- Selecting "interesting" heads currently requires either prior knowledge or full capture + post-filtering.
- Full attention for long sequences is prohibitive; smarter selection or summarization is needed.

Recommended general enhancements:
- Harden, document, and make the attention path less "experimental" (clear fidelity/cost notes vs serving path).
- Preserve and promote `previous_token` and fine-grained (layer/head/query/key) selection.
- Provide (or document how to request) slices useful for recurrent/previous-token analyses.
- Consider head-level metadata or lightweight ways to surface patterns without full (L, H, Q, K) tensors.
- If/when vLLM surfaces better attention access, evaluate integrating while keeping the replay path as fallback.
- Strong warnings + limits for quadratic blowup; encourage artifact + selective extraction.

## Multi-Sample / Ensemble Collection

Current state:
- Extraction endpoints enforce `n=1`.
- Users can call the API multiple times (different seeds) and manage traces themselves.
- No built-in grouping or alignment of multiple traces for one conceptual prompt.

Gaps:
- Several strong methods (EigenScore variants, Semantic Entropy, consistency) are built on K independent generations + aggregation or comparison of their internals.
- Collecting K traces manually is workable but repetitive; aligning hidden states or logprobs across samples requires care (same prompt tokenization, ordering, etc.).
- Sampling parameter consistency across the K calls matters.

Recommended general enhancements:
- Documented patterns + small utilities for "K-sample research collection" (same prompt, varied sampling).
- Optional support for small n>1 in extract paths (or a dedicated batch-extract that returns a list of traces) if it can be done without complicating the single-generation hot path.
- Helpers to stack/align per-sample hidden or logprob features (e.g., K x D matrix from last-token hiddens).
- Clear seed/sampling metadata in traces for reproducibility.

## Research Ergonomics & Loading

Current state:
- Excellent one-liner loaders: `load_trace_bundle`, `load_artifact`.
- `TraceEnvelope` + `TensorRecord` with good metadata (shapes, dtypes, layers/positions/heads, capture_mode, artifact refs).
- `scripts/dataset_workflow.py` as end-to-end example.
- Resource limits + `/extraction-schema` for clients to adapt.

Gaps:
- Researchers doing UQ repeatedly write boilerplate to pull "chosen logprobs as array", "hidden matrix for generated tokens at layer X", "attention to previous for selected heads".
- Large extraction workflows need careful limit tuning and artifact management.
- No standard "research feature bundle" shape that many methods consume.

Recommended general enhancements:
- A small set of **generic** (non-method) helpers, e.g. in `research/` or a new lightweight `features` area:
  - `chosen_logprobs(trace) -> np.ndarray`
  - `hidden_matrix(trace, layer_selector, positions="generated") -> np.ndarray`
  - `attention_prev_token(trace, layer, heads) -> ...`
  - Common aggregates and normalizations.
- These must remain generic and documented as "building blocks".
- Improve artifact ergonomics for research scale (clearer large-extraction workflow, compression choices, examples).
- Better examples showing "from server to numpy features for UQ-style analysis" in one screenful.
- Consider optional "light" inline returns for small common cases alongside artifact option.

## Limits, Fidelity, Ops & Preflight

Current:
- Server-side caps on top-k, #layers, #heads, #positions, inline bytes, total captured bytes, artifact bytes.
- `wllm doctor` for environment sanity (vLLM version, torch/transformers presence, local model paths).
- Clear capability matrix in schema responses.
- Replay hidden requires headroom (GPU mem util <= ~0.5, TP=1).

Gaps for research:
- Research workloads often intentionally push limits (many layers, many heads for attention analysis, long contexts).
- Distinguishing "replay numericals" vs "online" vs "Transformers attention replay" is critical for reproducible science.
- No easy pre-check for "will this hidden/attention request fit?"

Recommendations:
- Make large extraction (`--enable-large-extraction`) + artifact path the documented happy path for serious extraction.
- Expand `doctor` or add a "preflight extract" mode that validates selectors/limits against a loaded model's topology without running full generation.
- Richer capture metadata (numerical path, exact vLLM/Transformers versions if relevant, dtype notes).
- Clear docs on numerical interchangeability (or lack thereof) between modes.
- Guidance for VRAM planning when using replay paths.

## Non-Goals / Out of Scope for "General"

- Implementing any specific UQ scorer inside wllm core or as first-class request fields.
- Full raw logits / exact vocab entropy (unless vLLM surface changes; keep explicit).
- Online attention capture from the vLLM serving path (documented as out of scope).
- Removing resource limits (they protect the server).
- Streaming extraction (current design choice).

## Prioritized General Next Steps (Recommended for wllm)

These are **not** tied to any single paper. They are derived from repeated friction across the studied methods.

### Tier 1 (High impact, small surface, broad applicability)
1. **Chosen token probability as first-class, lightweight data**
   - Ensure the probability (or logprob) of the actually generated/chosen token is always easily available when tokens or light extraction is requested.
   - Add or document a simple accessor / artifact array for the sequence of chosen logprobs/probs.
   - Generic research helper: something like `chosen_logprobs(trace)` or equivalent.
   - **Implemented**: `trace.tokens.chosen_logprobs` is now populated for extract paths (even with only `"tokens": true`). Helper `research.features.chosen_logprobs(trace)`.

2. **Prompt / statement hidden-state access**
   - Make it straightforward to extract hidden states for the prompt (or a provided statement) without generation, or with max_tokens=0 semantics.
   - Common conveniences for "last token of prompt" and "last generated token" at chosen layers.
   - This directly helps probing-style work (SAPLMA-like) and pre-answer uncertainty.
   - **Implemented**: `max_tokens: 0` + selectors on `"prompt"` / `"last"` works for pre-generation hidden states. `last_token_hidden(...)` helper added.

3. **Generic research feature helpers (non-method-specific)**
   - Small module (e.g. under `research/` or a documented `features` area) providing numpy-friendly views:
     - chosen probs
     - hidden matrix for selected layers + positions
     - previous-token attention slices when available
   - These are building blocks, not scorers.

### Tier 2 (Important for coverage of attention + spectral + multi-sample methods)
4. **Attention practicality**
   - Reduce "experimental" status with clear docs on cost, fidelity (replay vs. vLLM path), and recommended usage for pattern analysis (previous_token etc.).
   - Helpers or selection ergonomics for common relational patterns.
   - Keep strong limits + artifact encouragement.

5. **Multi-sample / K-trace ergonomics**
   - Documented patterns + small utilities for collecting K traces for one prompt (varying seed/sampling).
   - Helpers to produce stacked feature matrices across samples (for EigenScore-over-samples, semantic entropy hybrids, etc.).
   - Consider small, opt-in support for n>1 in extraction contexts if it can be clean.

### Tier 3 (Nice-to-haves / longer term)
- More capture sites inside the transformer block (if cleanly available via compat layer).
- Higher or research-tunable top-k + better distribution stats options.
- Enhanced preflight / doctor for research extraction planning (will-this-fit checks).
- Richer per-tensor / per-trace metadata for reproducibility (capture path, versions).

All changes must:
- Remain generic.
- Update capabilities / schema / docs.
- Keep private vLLM access isolated.
- Not introduce paper-specific concepts into public surfaces.

See the per-method notes for the concrete "if wllm gave X" statements that fed into this list.

## Prioritization Heuristic Used

1. High impact + low surface change (e.g., always expose chosen logprob cleanly + generic helpers).
2. Unlocks multiple method families (prob + hidden + attention + multi-sample).
3. Preserves existing invariants (generic API, error semantics, vLLM compat isolation, no paper names).
4. Improves researcher UX and reproducibility without hurting normal serving performance.

See `benchmarking/README.md` for how the study was performed and links to per-method evidence.
