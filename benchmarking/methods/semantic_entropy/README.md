# Semantic Entropy

**Key papers**: Kuhn et al. 2023 (Semantic Uncertainty), Farquhar et al. 2024 updates, also implemented in UQLM and other toolkits.

## Short Summary

Instead of token-level entropy, measure uncertainty over *semantic* clusters of multiple sampled answers. Generate several responses to the same prompt, cluster them by meaning (using NLI entailment or embedding similarity), then compute entropy over the cluster distribution. High semantic entropy = high uncertainty / higher chance of hallucination on the factual content.

Some variants incorporate token probabilities inside the semantic framework ("token-probability-based semantic entropy").

## Data Contract

Core needs:
- Multiple (K) full generated responses for the same prompt (with their text or token sequences).
- A way to cluster them semantically (external NLI model or embeddings — not provided by the LLM internals necessarily).
- For hybrid white-box versions: the token probabilities/logprobs associated with each sample (to combine with semantic clustering).

Optional: hidden states or other features per sample for richer variants.

## Current wllm Mapping

wllm extraction currently enforces `n=1`. To get K samples:
- Call `/v1/traces` (or completions) K times with different `seed` or sampling params.
- Collect the generated text + (if requested) logprobs / hidden states for each.
- Then do clustering + entropy outside (using an NLI model via another library, sentence-transformers, etc.).

You can request tokens + logprobs + hidden states on each sample for richer analysis.

## Friction & Gaps

- No built-in multi-sample extraction or "generate K responses + aligned traces" endpoint.
- Managing K independent HTTP calls, ensuring consistent tokenization/prompt rendering, and aligning features (e.g., stacking last hiddens across the K responses) is entirely manual.
- For methods that want both the generations *and* their internal features, you pay the extraction cost K times with no special batching help.
- Semantic clustering itself is out of scope for wllm (correctly — it's a research choice of NLI/embedder), but the "easy to collect K consistent white-box traces" part is in scope.

## "If wllm gave X..."

- Documented, reproducible way to request K samples for one conceptual prompt and receive a grouped set of traces (or a simple helper that does the loop and returns aligned structures).
- Helpers to produce "K samples × features" matrices (e.g., K last-hidden vectors, or per-sample sequence logprobs) ready for semantic entropy or EigenScore-over-samples.
- Clear sampling metadata so the K generations are comparable.
- (Nice-to-have) optional small-n support in extract for the common research pattern of "a few samples for UQ".

## References

- Kuhn et al. "Semantic Uncertainty..." (2023)
- Farquhar et al. and UQLM implementations
- Many papers comparing semantic entropy to white-box token and internal-state methods.
