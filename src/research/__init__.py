from research.base import ResearchAdapter, ResearchResult
from research.features import (
    build_activation_map,
    chosen_logprobs,
    entropy_from_raw_logits,
    get_hidden_trajectories,
    get_prefill_activation_map,
    hidden_states_matrix,
    last_token_hidden,
    stack_features_across_samples,
)

__all__ = [
    "ResearchAdapter",
    "ResearchResult",
    "build_activation_map",
    "chosen_logprobs",
    "entropy_from_raw_logits",
    "get_hidden_trajectories",
    "get_prefill_activation_map",
    "hidden_states_matrix",
    "last_token_hidden",
    "stack_features_across_samples",
]
