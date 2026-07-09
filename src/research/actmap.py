from __future__ import annotations

from typing import Any

import numpy as np

from research.base import ResearchResult
from research.features import build_activation_map, get_hidden_trajectories
from schemas.traces import TraceEnvelope


class ActMapAdapter:
    """Example adapter that builds a flexible activation-map representation
    from generic hidden-state artifacts.

    This is *not* a verbatim reimplementation of any specific paper method.
    It demonstrates how to turn prompt (prefill) + generated (decoding) hidden
    trajectories into a pooled map using configurable channels.

    Particularly useful for prefill-only scenarios such as malicious prompt
    detection (request with max_tokens=0 + positions focused on "prompt").

    Usage (researcher code — your experimentation logic stays here):
        from research.features import get_prefill_activation_map, get_hidden_trajectories, build_activation_map
        # prefill only
        prefill_map = get_prefill_activation_map(trace, artifact_tensors, ...)
        # or full control
        traj = get_hidden_trajectories(trace, artifact_tensors)
        actmap = build_activation_map(traj.get("prompt") or traj.get("generated"), ...)
    """

    name = "actmap"

    def run(self, trace: TraceEnvelope, **options: object) -> ResearchResult:
        artifact_tensors = options.get("artifact_tensors", {})
        try:
            traj = get_hidden_trajectories(trace, artifact_tensors=artifact_tensors)
            # Prefer generated (decoding) trajectory for ActMap-style; fall back to prompt (prefill)
            key = "generated" if "generated" in traj and traj["generated"].size > 0 else "prompt"
            if key not in traj or traj[key].size == 0:
                return ResearchResult(
                    name=self.name,
                    status="error",
                    warnings=["No hidden trajectory found for prompt or generated tokens."],
                )
            actmap = build_activation_map(
                traj[key],
                num_layer_bins=options.get("num_layer_bins", 32),
                num_dim_bins=options.get("num_dim_bins", 128),
                channel_specs=options.get("channel_specs"),
            )
            return ResearchResult(
                name=self.name,
                status="ok",
                values={
                    "map_shape": list(actmap.shape),
                    "phase": key,
                    "num_layers_in_traj": int(traj[key].shape[0]),
                    "num_tokens_in_traj": int(traj[key].shape[1]),
                },
            )
        except Exception as e:
            return ResearchResult(
                name=self.name,
                status="error",
                warnings=[f"Failed to build activation map: {e}"],
            )
