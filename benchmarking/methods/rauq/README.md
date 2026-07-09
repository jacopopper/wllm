# RAUQ (Recurrent Attention-based Uncertainty Quantification)

**Paper(s)**: Vazhentsev et al., "Uncertainty-Aware Attention Heads: Efficient Unsupervised Uncertainty Quantification for LLMs" (2025).

## Short Summary

Observes that, in certain "uncertainty-aware" attention heads, the attention weight from a newly generated token back to its immediate predecessor drops when the model is about to hallucinate. RAUQ identifies or uses such heads, uses the attention patterns (especially previous-token) as uncertainty signals, and propagates uncertainty across the sequence (recurrently) often in combination with token probabilities.

Very efficient when the right heads can be selected.

## Data Contract

- Attention weight tensors with fine-grained access:
  - Selected layers and heads.
  - Query positions (often generated tokens).
  - Key positions, crucially "the previous token" for each query.
- Typically (or slices of) attention scores of shape involving (layers, heads, query_pos, key_pos).
- Combined with token probabilities in many variants.

Key pattern: for each generation step q, look at attention[q, q-1] in specific heads.

## Current wllm Mapping

Attention extraction must be enabled at server start:
`wllm serve ... --enable-attention-weights`

Request example:
```json
{
  "extract": {
    "attentions": [{
      "layers": "middle",
      "heads": [0, 1, 5],
      "query_positions": "generated",
      "key_positions": "previous_token"
    }],
    "artifacts": {"format": "npz", "include": ["attentions"]}
  }
}
```

wllm has explicit support for `"previous_token"` key selector — a great match for the core RAUQ pattern.

However:
- Currently implemented via separate Transformers replay model (after generation).
- Experimental flag + extra deps (torch + transformers).
- Quadratic memory with sequence length → artifact strongly recommended.
- Selection of "which heads are uncertainty-aware" is usually done by analysis on data or prior knowledge; wllm lets you request specific heads.

## Friction & Gaps

- Attention path is explicitly experimental and replay-only (Transformers forward pass, not from vLLM internals). Cost and potential distribution shift vs. the serving model's attention.
- Capturing enough heads/layers for "find the uncertainty heads" or full analysis is expensive.
- No built-in discovery of uncertainty-aware heads from the trace.
- Combining attention slices with logprobs for the recurrent propagation is left entirely to the researcher (easy in principle, repetitive).
- VRAM and time overhead makes it less attractive for routine large-scale dataset collection compared to pure token-prob methods.

## "If wllm gave X..."

- Reliable, well-documented attention extraction with clear fidelity notes (replay vs native).
- Efficient selection / summarization for previous-token attention patterns across heads.
- Optional lightweight "head pattern" features (e.g., previous-token attention per head per position) without full quadratic tensors.
- Generic helpers to extract previous-token attention matrix slices aligned with token positions/logprobs.
- Lower-cost path or caching for attention when only specific relational patterns are needed.

## References

- Vazhentsev et al. 2025 paper and related follow-ups.
- Mentions in surveys and other hallucination detection papers that use attention dynamics.
