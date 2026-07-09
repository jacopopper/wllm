# Low-Level vLLM Patch Sketches for Inference Transparency

These are informal sketches of changes that would make wllm (and similar tools)
much more powerful with lower overhead and higher fidelity. They are **not**
implemented here; they would be contributions to the vLLM project.

## 1. Native Attention Weights Exposure

**Goal**: Allow attention weights to be returned from the main generation
path (no separate Transformers replay, better fidelity to the actual
inference kernels, lower memory).

**Sketch (vLLM side)**:

In `vllm/sampling_params.py`:
```python
@dataclass
class SamplingParams:
    ...
    return_attention_weights: bool = False   # new
    # or per-layer/head selection for efficiency
```

In the attention layers / model executor, when `return_attention_weights`,
capture and return attn weights (for the selected heads) alongside the
output.

Expose on `CompletionOutput`:
```python
attention_weights: Optional[list[torch.Tensor]] = None  # or structured
```

Then in `RequestOutput` / generation path, propagate them.

**wllm side impact**:
- Remove or deprecate the `--enable-attention-weights` + replay path for
  most use cases.
- `extract.attentions` can be served with almost zero extra cost when the
  flag is on.
- RAUQ etc. become practical at scale.

## 2. Raw Logits on Output (or via cheap side-channel)

**Goal**: Full (or configurable) logits for generated tokens without
approximation or post-hoc replay.

**Sketch**:
- `SamplingParams(return_logits: bool = False, top_k_logits: Optional[int] = None)`
- `CompletionOutput.logits: Optional[torch.Tensor]` (or list per step)
- Or a separate `get_raw_logits()` on the output object.

This would make `extract.logprobs.raw_logits` trivial and always exact
(entropy, margins, etc.).

## 3. Richer / Cheaper Hidden State Capture

- Add `output_hidden_states` support that can target specific sites
  (post-attn, post-mlp, pre/post residual) without a full second runner.
- Layer-wise or streaming capture to reduce VRAM for replay on >7B models.
- Make online hidden capture compatible with prefix caching where safe.

## 4. Multi-Sample with Shared Internals

- Better support for `n>1` that returns per-candidate hidden/attention
  without duplicating the entire forward machinery.
- Or a "sample_and_inspect" mode that re-uses prefixes.

These changes would let wllm drop many of its current workarounds and
limitations while keeping the serving path fast for non-research traffic.
