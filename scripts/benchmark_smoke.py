from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifacts.store import ArtifactStore
from extractors.planning import ResourceLimits
from runtime.vllm_runtime import VLLMRuntime, VLLMRuntimeConfig
from schemas.extraction import ExtractRequest
from schemas.openai import ChatCompletionRequest, ChatMessage


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare trace-free and bounded extraction generation.")
    parser.add_argument("model")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    runtime = VLLMRuntime(VLLMRuntimeConfig(model=args.model, local_files_only=args.local_files_only))
    chat = ChatCompletionRequest(
        model=args.model,
        messages=[ChatMessage(role="user", content="Say hello in one short sentence.")],
        max_tokens=16,
    )
    started = time.perf_counter()
    runtime.generate_chat(chat)
    normal_ms = (time.perf_counter() - started) * 1000.0

    extract = ExtractRequest(
        model=args.model,
        messages=[ChatMessage(role="user", content="Say hello in one short sentence.")],
        max_tokens=16,
        extract={"tokens": True, "logprobs": {"top_k": 5}},
    )
    trace = runtime.generate_extract(extract, limits=ResourceLimits(), artifact_store=ArtifactStore(__import__("pathlib").Path("./wllm-artifacts")), persist=False)
    extract_ms = trace.metadata.timing_ms.total
    print(f"normal_ms={normal_ms:.2f}")
    print(f"extract_ms={extract_ms:.2f}")
    print(f"measured_delta_ms={extract_ms - normal_ms:.2f}")
    print(f"capture_ms={trace.metadata.timing_ms.capture:.2f}")
    print(f"postprocess_ms={trace.metadata.timing_ms.postprocess:.2f}")
    print(f"serialization_ms={trace.metadata.timing_ms.serialization:.2f}")
    print(f"extraction_overhead_ms={trace.metadata.timing_ms.extraction_overhead:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
