# EigenScore (INSIDE)

**Paper**: Chen et al., "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection", ICLR 2024.

## Short Summary

Uses the internal hidden states of one (or more) generated responses. For a response, collect hidden state vectors (often last-token at a chosen layer, or across tokens), form a matrix, compute covariance, and derive a score from the eigenvalues (e.g., log-determinant of a regularized covariance). Lower "volume"/higher consistency in representation space indicates lower hallucination risk. Variants operate over K sampled responses.

## Data Contract

Primary:
- Hidden state vectors for the response (shape examples seen: (K, D) where K = num samples or num tokens, D = hidden dim at the chosen layer).
- Often specifically the **last generated token** hidden state at the **final layer** (or selected mid/late layers).
- For per-token versions: matrix over generated positions at one or more layers.

Computation sketch (from paper analyses):
- Center the matrix X = H - mean
- EigenScore ~ log det( (1/D) X X^T + alpha I )

Requires consistent hidden states across the compared items (same layer, same position semantics).

## Current wllm Mapping

```json
{
  "extract": {
    "hidden_states": [
      {"layers": "middle", "positions": "last_generated", "pool": null}
    ],
    "artifacts": {"include": ["hidden_states"]}
  }
}
```

- Returns `TensorRecord`(s) with `layers`, `positions`.
- Loaded artifact gives the actual tensor(s).
- Can request multiple layers or "generated" positions and post-process (e.g., take last position per layer).

For multi-sample EigenScore: run K separate traces (vary seed/temp), collect last-hidden vectors, stack.

## Friction & Gaps

- "last_generated" + specific layer selection works, but users must know the exact selector and do the stacking themselves.
- No built-in "collect K responses and give me the (K, D) hidden matrix" helper.
- Replay hidden has VRAM cost (second model instance); large models may not support it.
- Capture is at "transformer_block_output"; some analyses want other points in the block.
- For single-response per-token covariance, need many positions without pooling — supported but may hit position/byte limits.
- Alignment across samples (exact same prompt tokenization, position semantics) is user responsibility.

## "If wllm gave X..."

- Easy way to request "last token hidden at layer(s)" with clear semantics.
- Generic helper to produce a stacked matrix over multiple traces/samples for spectral methods.
- Prompt-only hidden extraction (some variants look at pre-generation states).
- Better multi-layer / cheap capture for "which layer is best" ablations.
- Clear metadata so you know you have exactly the vectors needed for the covariance.

## References

- Main paper + eigenscore code repos (Alibaba/D2I-ai etc.).
- Follow-up papers that use or compare EigenScore on internal states.
