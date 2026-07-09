# Methods Index

This directory contains analysis for representative white-box UQ / hallucination detection approaches.

| Directory                    | Category                  | Core Data Needs                     | Primary wllm Surface Today      | Key Friction                  |
|------------------------------|---------------------------|-------------------------------------|---------------------------------|-------------------------------|
| token_probability_whitebox  | Token probs (single-gen) | Chosen token logprobs/probs (+ top-k for entropy variants) | logprobs extraction + artifacts | Chosen prob tied to logprobs request; top-k limits; exact entropy unsupported |
| eigenscore_inside           | Hidden states (spectral) | Hidden vectors (last token or per-token) at layer(s), optionally over K samples | hidden_states selectors + artifacts | No first-class multi-sample stacking; replay cost; prompt-only less obvious |
| rauq                        | Attention patterns       | Attention weights for query=generated, key=previous_token in selected heads | attentions (experimental replay) + previous_token selector | Experimental + replay cost/ fidelity; no head discovery |
| saplma_probe                | Probing on activations   | Hidden vectors at chosen layers/positions (esp. last of prompt or statement) | hidden_states (last / last_generated) | No dedicated prompt-only hidden extraction; selector verbosity for common probing patterns |
| semantic_entropy            | Multi-sample + semantics | K generations (+ optional internals) + external clustering | Multiple separate traces | n=1 only; no built-in K-sample collection or alignment helpers |

See individual READMEs for detailed data contracts, example requests, and "if wllm gave X" suggestions.

All recommendations are generalized in `../gaps_and_recommendations.md`.
