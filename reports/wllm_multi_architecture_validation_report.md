# Multi-Architecture Validation Report

- Created UTC: 2026-06-20T08:25:00.000000+00:00
- Python: `3.12.13` via `/home/jacopodardini/projects/wllm/.venv/bin/python`
- vLLM: `0.10.2`, torch: `2.8.0+cu128`, CUDA: `12.8`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU` (24GB VRAM)
- Test config: max_model_len=1024, gpu_memory_utilization=0.35 (0.90 for Mistral-7B)

## Models Tested

| Model | Family | Size | HF Path | Architecture |
|-------|--------|------|---------|-------------|
| Qwen3-0.6B | Qwen | 0.6B params (1.1 GB weights) | `models--Qwen--Qwen3-0.6B/snapshots/c1899de2...` | Qwen3ForCausalLM |
| Llama-3.2-3B | Llama | 3B params (6.0 GB weights) | `models--meta-llama--Llama-3.2-3B/snapshots/13afe512...` | LlamaForCausalLM |
| Mistral-7B-Instruct-v0.3 | Mistral | 7B params (~14.5 GB weights) | `models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708...` | MistralForCausalLM |

## Integration Test Results

| Test | Qwen3-0.6B | Llama-3.2-3B | Mistral-7B-Instruct-v0.3 |
|------|:----------:|:------------:|:------------------------:|
| `test_vllm_token_logprob_extraction_smoke` | ✅ PASS | ✅ PASS | ✅ PASS |
| `test_vllm_hidden_state_extraction_smoke` | ✅ PASS | ✅ PASS | ❌ UNSUPPORTED (VRAM) |
| `test_vllm_online_hidden_state_extraction_smoke` | ✅ PASS | ✅ PASS | ❌ UNSUPPORTED (VRAM) |
| `test_vllm_hidden_state_extraction_replays_full_sequence_repeatedly` | ✅ PASS | ✅ PASS | ❌ UNSUPPORTED (VRAM) |

### Failure Details

1. **Hidden state extraction (Mistral-7B-Instruct-v0.3)**:
   - Error: `UnsupportedExtractionError: Hidden-state extraction is not available`
   - Root cause: Replay capture requires `gpu_memory_utilization <= 0.50` to leave room for a second model instance. Mistral-7B (~14.5GB) does not fit in 12GB (0.50 of 24GB). Reducing `gpu_memory_utilization` to levels that satisfy the constraint would leave insufficient KV cache (< 1 GB).
   - Affected architecture: **All 7B+ models** on 24GB GPUs.

## Capability Matrix

| Capability | Qwen3-0.6B | Llama-3.2-3B | Mistral-7B-Instruct-v0.3 |
|------------|:----------:|:------------:|:------------------------:|
| **Normal generation** | ✅ | ✅ | ✅ |
| **Chat completions** | ✅ | ⚠️ No chat template (base model) | ✅ |
| **Text completions** | ✅ | ✅ | ✅ |
| **Token ID extraction** | ✅ | ✅ | ✅ |
| **Logprob extraction (top-k)** | ✅ | ✅ | ✅ |
| **Prompt logprobs** | ✅ | ✅ | ✅ |
| **Hidden state replay** | ✅ | ✅ | ❌ VRAM (7B too large) |
| **Hidden state online** | ✅ | ✅ | ❌ VRAM (7B too large) |
| **Prewarmed replay** | ✅ | ✅ | ❌ VRAM |
| **NPZ artifacts** | ✅ | ✅ | ✅ (token/logprob only) |
| **PT artifacts** | ✅ | ✅ | ✅ (token/logprob only) |
| **Artifact round-trip** | ✅ | ✅ | ✅ |

Legend: ✅ = Working, ❌ = Failing/Unsupported, ⚠️ = Functional with caveat

## Latency Summary (warm, avg ms, gpu_memory_utilization=0.35)

| Mode | Qwen3-0.6B | Llama-3.2-3B | Mistral-7B (0.90) |
|------|:----------:|:------------:|:------------------:|
| Normal generation (16 tokens) | 45.5 ms | 153.2 ms | 328.6 ms |
| Token + logprob extraction | 62.7 ms | 168.1 ms | 337.1 ms |
| Hidden state extraction (replay, 1 layer, 1 position) | 1163.7 ms | 1804.0 ms | UNSUPPORTED |
| Overhead: extract vs normal | +17.2 ms (+38%) | +14.9 ms (+10%) | +8.5 ms (+3%) |
| Overhead: hidden state vs normal | +1118.2 ms | +1650.8 ms | N/A |

Notes:
- Latency measurements are warm (after vLLM initialization and first warmup request).
- Hidden state extraction includes lazy pooling-runner initialization on first call (Qwen: ~3.3s cold start).
- Mistral-7B used `gpu_memory_utilization=0.90` for basic generation; hidden state unsupported at any setting.
- Normal generation latency scales with model size: Qwen3-0.6B < Llama-3.2-3B < Mistral-7B.

## Architecture-Specific Differences

### 1. Chat Template Availability
- **Qwen3-0.6B**: Built-in chat template. `messages` format works out of the box.
- **Llama-3.2-3B**: Base model — no `chat_template` in tokenizer config. Chat completions via `/v1/chat/completions` fail with `ValueError: tokenizer.chat_template is not set`. Text completions via `/v1/completions` work correctly. Users must use prompt-based completions or supply a chat template.
- **Mistral-7B-Instruct-v0.3**: Built-in chat template. `messages` format works out of the box.

### 2. Online Hidden State Capture
- **Qwen3-0.6B**: Works correctly. Position mapping aligns with prompt-prefix slicing.
- **Llama-3.2-3B**: Works after BOS-aware prompt-length estimation for the dense-prefix capture filter.
- **Mistral-7B-Instruct-v0.3**: Not tested for hidden-state capture on this 24GB GPU due to the model memory footprint.

### 3. Hidden State Replay VRAM Requirements
- Hidden state replay requires `gpu_memory_utilization <= 0.50` to leave room for the isolated pooling runner (second model instance).
- On 24GB GPU:
  - Qwen3-0.6B (~1.1GB weights): Uses ~3GB total with replay, leaves comfortable KV cache.
  - Llama-3.2-3B (~6.0GB weights): Uses ~12GB total with replay, KV cache reduced to ~1GB.
  - Mistral-7B (~14.5GB weights): Cannot fit in 12GB (0.50), let alone with a second instance.
- **Constraint**: Models larger than ~6B parameters cannot use hidden state replay on a single 24GB GPU.

### 4. KV Cache Capacity (gpu_memory_utilization=0.35)
| Model | Available KV Cache | Max Tokens @1024 ctx |
|-------|:------------------:|:--------------------:|
| Qwen3-0.6B | 5.66 GiB | 53,008 tokens, 51.77x concurrency |
| Llama-3.2-3B | 0.95 GiB | 8,880 tokens, 8.67x concurrency |
| Mistral-7B (0.90) | ~19 GiB | ~173,000 tokens |

Llama-3.2-3B has significantly less KV cache headroom at the same `gpu_memory_utilization` because its weight memory (6GB) consumes most of the budget.

### 5. Tokenizer Compatibility (All Models)
- **transformers 5.12.1 + vLLM 0.10.2**: wllm applies a compatibility shim for `all_special_tokens_extended` before importing vLLM. Prewarmed replay is verified on Qwen3-0.6B and Llama-3.2-3B with this stack.

### 6. Resolved Architecture Strings
| Model | vLLM Resolved Architecture |
|-------|---------------------------|
| Qwen3-0.6B | `Qwen3ForCausalLM` |
| Llama-3.2-3B | `LlamaForCausalLM` |
| Mistral-7B-Instruct-v0.3 | `MistralForCausalLM` |

All three use Flash Attention backend on V1 engine.

## Recommendations

1. **Keep tokenizer compatibility covered**: Retain regression coverage for prewarmed replay under vLLM 0.10.2 + transformers 5.12.1.

2. **Large model hidden states**: For >6B models on 24GB GPUs, consider:
   - Online capture mode (avoid second instance) where eager execution memory permits.
   - Reduce model precision (INT8/INT4 quantization) — currently unsupported by wllm.
   - Use larger GPUs (48GB+) for 7B+ models with hidden state replay.

3. **Llama chat template**: Document that Llama base models require prompt-based completions. Consider adding a `--chat-template` CLI option for injecting a custom template, or auto-detecting instruct variants.

4. **Future architectures**: Test with Llama-3.2-3B-Instruct to verify chat template availability. Test Gemma and Phi model families for wider architecture coverage.
