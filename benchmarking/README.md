# wllm Benchmarking for White-Box UQ & Hallucination Detection

This folder contains an evidence-based study of real white-box (and closely related) methods used by researchers for uncertainty quantification (UQ) and hallucination detection in LLMs.

**Goal**: Identify what data and capabilities would make implementing these methods (and future similar research) *much easier* on top of wllm, then derive **general, method-agnostic** improvements.

## Philosophy (aligned with wllm)

- wllm provides **generic, high-quality traces and artifacts** (`TraceEnvelope` + NPZ/PT tensors).
- Researchers own their analysis: paper-specific logic lives in user code or `research/` adapters (never in core schemas/server).
- See `src/research/` (stubs deliberately say "unsupported; use generic trace tensors") and tests that prevent paper names from leaking into public API.

## Top 5 Methods Studied

1. **Token-probability white-box scorers** (UQLM `WhiteBoxUQ` and foundational papers: Manakul et al. 2023, Malinin & Gales 2021, etc.)
   - Min token probability, Length-Normalized Token/Sequence Probability (LNTP), sequence probability, (top-k) negentropy, probability margin.
   - Single-generation, extremely low overhead.

2. **EigenScore (INSIDE)** (Chen et al., ICLR 2024)
   - Spectral analysis (covariance eigenvalues / log-det) of hidden states (often last-token or per-token at selected layers) across one or K sampled responses.
   - Measures semantic consistency in representation space.

3. **RAUQ** ("Uncertainty-Aware Attention Heads", Vazhentsev et al. 2025)
   - Leverages distinctive attention patterns (esp. drop in attention to previous token) in a small subset of heads during hallucinated continuations.
   - Combines selected attention weights with token probabilities; recurrent propagation of uncertainty.

4. **SAPLMA-style probing** (Azaria & Mitchell 2023 + many follow-ups)
   - Train linear (or small) probes on hidden activations (specific layers, typically last token of prompt/statement or generated tokens) to predict truthfulness or hallucination risk.
   - "The internal state knows when it's lying."

5. **Semantic Entropy** (Kuhn et al. 2023, Farquhar et al. updates; also in UQLM)
   - Multi-sample generations + semantic clustering (via NLI or embeddings), then entropy over semantic clusters.
   - Foundational UQ method; highlights needs for consistent multi-trace collection.

(See individual `methods/*/README.md` or notes for detailed data contracts, wllm mappings, and friction analysis.)

## How Data Contracts Were Analyzed

- Primary sources: UQLM library (https://github.com/cvs-health/uqlm), INSIDE/EigenScore repos and paper, RAUQ papers, SAPLMA paper + follow-up repros, Semantic Entropy implementations.
- For each: extracted the exact arrays/tensors consumed by the core scoring or feature functions.
- Mapped to current wllm `extract` options, `TraceData` (logprobs/hidden_states/attentions), `TensorRecord`, and artifact loading.
- Documented workarounds, performance costs, missing fields, and selector ergonomics.

## Key Outputs

- `gaps_and_recommendations.md` — synthesized, general next steps (no method names).
- Per-method notes with:
  - Short method summary + citation.
  - Required inputs ("Data Contract").
  - Current wllm request + load example.
  - Friction points.
  - "If wllm provided X, this would be trivial."

## Running wllm for These Workflows

```bash
wllm serve Qwen/Qwen3-0.6B --local-files-only --gpu-memory-utilization 0.35 --tensor-parallel-size 1
# or with attention: --enable-attention-weights

# Example extraction for rich data (tokens + logprobs + middle hidden + artifacts)
curl -X POST http://localhost:8000/v1/traces \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Explain calibration briefly.",
    "max_tokens": 32,
    "extract": {
      "tokens": true,
      "logprobs": {"top_k": 10, "include_prompt": true, "allow_approximate_entropy": true},
      "hidden_states": [{"layers": "middle_third", "positions": "generated"}],
      "artifacts": {"format": "npz", "include": ["logprobs", "hidden_states"]}
    }
  }'
```

Then load with the one-liners:

```python
from artifacts import load_artifact
from tracing.serialization import load_trace_bundle

trace = load_trace_bundle("./wllm-artifacts", response["trace_manifest"])
tensors = load_artifact("./wllm-artifacts", response["artifacts"][0])
```

See `scripts/dataset_workflow.py` for a full prompt-file → trace → load → adapter example.

## Recommendations Direction (preview)

Improvements should be **general**:
- Richer and more reliable probability information (chosen token prob always first-class; better support for distribution stats).
- More flexible and cheaper hidden-state capture (per-token, more layers/granularity via artifacts, prompt-only / pre-generation access, additional capture sites).
- Practical attention (harden the experimental path, improve fidelity/cost story, head-level selection ergonomics).
- Multi-sample collection and alignment helpers (easy K traces for EigenScore-style or semantic methods).
- Research ergonomics (generic feature helpers that turn traces+artifacts into numpy matrices/arrays commonly needed; better metadata).
- Documentation and preflight targeted at UQ-style workloads.

See `gaps_and_recommendations.md` for the prioritized list.

## Contributing / Extending

- Add new method notes under `methods/NEW_METHOD/` following the template.
- Update gaps doc only with cross-cutting insights.
- Any core changes must be justified by broad utility (not "this one method needs X").

Generated large data and temp clones go in ignored subpaths.

## References & Sources (high level)

- UQLM: https://github.com/cvs-health/uqlm (and associated papers)
- INSIDE / EigenScore: Chen et al. ICLR 2024 (and eigenscore repos)
- RAUQ: Vazhentsev et al. (Uncertainty-Aware Attention Heads)
- SAPLMA: Azaria & Mitchell, EMNLP Findings 2023
- Semantic Entropy: Kuhn et al. 2023, Farquhar et al., and implementations in UQLM / other repos

Full citations and links in the per-method notes.
