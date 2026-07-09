# Token-Probability White-Box Scorers

**Representative implementations**: UQLM `WhiteBoxUQ` (single-generation scorers), foundational papers including Manakul et al. (SelfCheckGPT-related token metrics), Malinin & Gales (length-normalized probabilities), various negentropy / margin variants.

## Short Summary

These are the simplest, lowest-overhead white-box UQ methods. They operate purely on the per-token probabilities (or logprobs) of the generated sequence (sometimes prompt too). No hidden states or attention required.

Common scores:
- Minimum token probability (of the chosen tokens)
- Length-normalized sequence / token probability (geometric mean)
- Sequence probability (product)
- Mean / Min top-k negentropy (or entropy over top-k)
- Probability margin

Higher score = higher confidence / lower hallucination risk in many evaluations.

## Data Contract (what the scorers actually consume)

- Sequence of **chosen token log-probabilities** (or probabilities) for the generated tokens. Often also for prompt if included.
- For top-k variants: the top-k alternative logprobs at each step (to compute renormalized entropy/negentropy).
- Optional: ability to compute or access the probability of the selected token even when only top-k candidates are returned (by matching token id).

Typical shapes:
- `chosen_logprobs`: (generated_len,) or (prompt_len + generated_len,)
- `top_logprobs` per step: list of {token, logprob} or array.

Many implementations convert logprob -> prob = exp(logprob), then aggregate (min, prod, geo-mean, etc.).

## Current wllm Mapping (as of study)

Request:
```json
{
  "extract": {
    "tokens": true,
    "logprobs": { "top_k": 10, "include_prompt": true, "allow_approximate_entropy": true }
  }
}
```

In the returned trace:
- `trace.logprobs.generated[i].logprob` — logprob of the chosen token for generated position i.
- `trace.logprobs.generated[i].top_logprobs` — top alternatives.
- Similar for prompt if requested.
- Approx entropy available when allowed.

Loading from artifacts (when `"artifacts": {"include": ["logprobs"]}`):
- `generated_logprobs`, `generated_logprob_token_ids` (and prompt variants) are stored as arrays in the NPZ/PT.

One can reconstruct chosen probs easily from the inline trace or artifact.

## Friction & Gaps Identified

- Chosen token logprob is present **only when** `logprobs` extraction is requested. There is no lightweight "just give me the chosen probs" without the top-k machinery (though top_k=1 works).
- Top-k is capped by server config (default 50). Some advanced variants want higher or different renormalization.
- Exact full-distribution entropy / better distribution stats are unsupported (as documented). Approx over top-k is provided but must be explicitly allowed.
- For pure sequence probability products, users must do the exp + product themselves (easy, but boilerplate).
- When using artifacts, the arrays are there but naming (`generated_logprobs`) requires reading docs or source to map back to "chosen vs alternatives".

## "If wllm gave X, implementation would be trivial"

- A reliable, always-available array or accessor for the sequence of **chosen token logprobs/probs** (for generated, optionally prompt).
- Generic helper: `chosen_logprobs(trace_or_loaded)` or equivalent in the research layer.
- Documented + convenient way to request "just the probabilities I need for UQ baselines" with low overhead.
- Higher practical top-k for research (or clear path to request more).
- Built-in or example helpers for min, length-normalized geo-mean, sequence logprob, top-k negentropy.

## References

- UQLM white-box scorers and notebooks (cvs-health/uqlm)
- Manakul et al. 2023 (SelfCheckGPT)
- Malinin & Gales 2021 (and related)
- Various follow-ups using negentropy / margin on token probs.

**Note**: Keep any implementation of these scores in user/researcher code or benchmarking repros. Do not add method-specific fields to wllm schemas.
