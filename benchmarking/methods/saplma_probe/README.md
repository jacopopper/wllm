# SAPLMA-style Probing on Hidden States

**Paper**: Azaria & Mitchell, "The Internal State of an LLM Knows When It's Lying", EMNLP Findings 2023 (and many probing follow-ups: linear probes, CCS, etc.).

## Short Summary

The hidden activations of an LLM, when processing a statement (input or generated), contain information about whether that statement is true or false / likely to be a hallucination. Train a simple classifier (linear or small MLP) on hidden vectors (often at specific layers, frequently the last token representation) to predict truthfulness. Can be done on prompt states (pre-answer) or on generated content.

## Data Contract

- Hidden state vector(s) at one or more layers for a specific position:
  - Commonly the representation of the **last token** of the prompt/statement.
  - Or last generated token, or per-token for sequence-level labels.
- Layer(s) chosen by validation (often mid-to-late layers perform well).
- Labels (true/false or hallucinated/not) for supervised training of the probe.
- Multiple examples to fit the probe.

After training, inference is just forward the probe on new hidden vectors.

## Current wllm Mapping

Use hidden state extraction targeted at the desired position/layer:

```json
{
  "extract": {
    "hidden_states": [
      {"layers": [10, 12, 14], "positions": "last", "pool": null},
      {"layers": [10, 12, 14], "positions": "last_generated", "pool": null}
    ]
  }
}
```

For prompt-only probing: you can send the statement as the prompt (or use chat messages) and request hidden states without (or with minimal) generation.

Loaded artifacts or inline small tensors give the vectors. You then collect many such vectors + labels and train `sklearn.linear_model.LogisticRegression` or a tiny torch probe.

## Friction & Gaps

- "last" vs "last_generated" vs specific token positions requires understanding spans and selectors.
- No dedicated "extract hidden states for this statement/prompt only (no generation)" optimized path (you can approximate by max_tokens=0 or small, but it's not first-class).
- For large-scale probing datasets, repeatedly requesting specific single vectors can be verbose; limits may require artifact path.
- Users must manage layer selection, train/val splits, and probe training themselves (expected).
- Capture site is block output; some probing literature experiments with other residual points.

## "If wllm gave X..."

- Explicit "prompt hidden states" or "statement representation" extraction mode that returns clean vectors for the end of the input without forcing generation.
- Convenience for "last token hidden at chosen layers" that returns ready-to-probe arrays with clear layer/position labels.
- Generic helpers in the research layer for "get probe feature matrix from a list of trace responses".
- Good support for mid-layer selection and many examples (efficient artifact roundtrips).

## References

- Azaria & Mitchell 2023.
- Follow-ups using probes on hidden states for hallucination / truthfulness / factuality.
- Related: InternalInspector, ICR Probe, and papers comparing probing vs spectral vs consistency methods.
