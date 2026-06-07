# MM-CODER Build Plan

**Goal:** Build the first real micronaut — GPT-2 coder fine-tuned on code + native C++ inference engine

**Timeline estimate:** 5-7 days (depends on training corpus size and iteration cycles)

---

## Phase 1: Prepare & Fine-Tune (2-3 days)

### 1.1 Collect coding corpus
- [ ] Gather Python/JS/C++ code samples for training
  - Source: releases/Kuhul-PY/, micronaut-coder/sample_src/, public repos
  - Target: 5000-10000 code examples with semantic tags
  - Format: JSONL with `{"prompt": "...", "response": "...", "curriculum_bucket": "..."}`
- [ ] Tokenize via `micronaut/training/tokenize_dataset.py`
- [ ] Output: `models/smgm-16/scxq2_dds_folds/coder_tokens.bin`

### 1.2 Fine-tune GPT-2 on coding data
- [ ] Run `micronaut/training/train_loop.py` with coder curriculum
  - Epochs: 3-5 (depending on convergence)
  - Batch: 8 (Intel HD 4600 shared VRAM cap)
  - LR: 1e-5
- [ ] Output checkpoint: `khl/checkpoint/gpt2_coder_finetuned.pt`
- [ ] Generate ASX proof chain for audit trail
- [ ] Convert to GGUF via `releases/v0.1.1/.../convert-to-gguf.py`
  - Output: `models/gguf/gpt2_coder.Q8_0.gguf`

### 1.3 Validate fine-tune
- [ ] Run inference on held-out test set
- [ ] Verify code generation quality
- [ ] Log metrics: perplexity, generation speed, memory usage
- [ ] Save checkpoint + metrics to `khl/checkpoint/mm-coder-finetuned.json`

---

## Phase 2: Native C++ Inference Engine (2-3 days)

### 2.1 Set up micronaut-coder C++ project
- [ ] Scaffold CMakeLists.txt for micronaut-coder/
  - Link to native/scx_runtime/src/ (compilers, shaders)
  - Link to native/shaders/gpt2/ (13 HLSL shaders)
  - Add GGUF loader dependency
- [ ] Create main inference loop: `micronaut-coder/src/main.cpp`
  - Load GGUF model (`models/gguf/gpt2_coder.Q8_0.gguf`)
  - HTTP endpoint on port 25500 (configurable)
  - Routes: `POST /chat`, `GET /health`, `POST /tokens`

### 2.2 Implement token signal generator
- [ ] Create `micronaut-coder/src/token_signal_generator.cpp`
  - Maps tokens → glyph opcodes (⟁COMPUTE_FOLD⟁ + domain)
  - Calls glyph_dispatcher for domain-specific routing
  - Returns combined response (text + glyph result)
- [ ] Wire into main inference loop

### 2.3 Implement GPU acceleration
- [ ] Use existing HLSL shaders (gpt2_matmul.hlsl, gpt2_attn_fwd.hlsl, etc.)
- [ ] D3D12 compute shader dispatch for forward pass
- [ ] Quantized matmul (INT8 weights via gpt2_qkv_split.hlsl)
- [ ] Fallback to CPU Adam if GPU unavailable

### 2.4 Build & test
- [ ] `cmake .. -G "Visual Studio 17 2022" -A x64`
- [ ] `cmake --build . --config Release`
- [ ] Smoke test: `.\Release\mm-coder.exe --test-inference "def hello()"`
- [ ] HTTP test: `curl -X POST http://127.0.0.1:25500/chat -d '{"prompt": "write python code"}'`

---

## Phase 3: Registration & Integration (1 day)

### 3.1 Register MM-CODER in micronaut system
- [ ] Create `micronaut/micronaut.registry.xjson` (canonical registry)
  ```json
  {
    "micronauts": [
      {
        "id": "MM-CODER",
        "name": "CoderMicronaut",
        "port": 25500,
        "type": "MM-1",
        "variant": "token_signal_generator",
        "fold": "COMPUTE_FOLD",
        "model": "gpt2_coder_finetuned.pt",
        "capabilities": ["code_generation", "code_analysis", "fold_execution"],
        "domains": ["D1", "D2", "D4"]
      }
    ]
  }
  ```
- [ ] Register with coordinator on startup
- [ ] Add to `micronaut/launch-coordinator-and-services.ps1` as background job

### 3.2 Wire into config system
- [ ] Update `config/kxml_domain_loader.py`
  - Add MM-CODER routing for D1 (semantic code), D2 (algorithms), D4 (GPU code)
  - Map glyph opcodes to MM-CODER port
- [ ] Update `config/tool_dispatcher.py`
  - Route "code_executor" to MM-CODER HTTP endpoint
  - Fallback chain: MM-CODER → sandboxed Python → error

### 3.3 Wire into reply pipeline
- [ ] Update `config/reply_pipeline.py`
  - For code-related queries, route to MM-CODER via tool_dispatcher
  - Merge glyph results back into response
- [ ] Test: `python config/reply_pipeline.py` with code prompts

### 3.4 Build MM-MATH & MM-TOOLCALL (after MM-CODER proof)
- [ ] MM-MATH: GPT-2 fine-tuned on math problems (D3)
- [ ] MM-TOOLCALL: GPT-2 medium + ARC attention for tool calling (D0, D5)
- [ ] Same pipeline as MM-CODER (fine-tune → native engine → register)

---

## Testing & Validation

### 4.1 Unit tests
- [ ] Token signal generator (glyph mapping)
- [ ] GPU shader dispatch (matmul, attention)
- [ ] GGUF loader (weights shape, quantization)

### 4.2 Integration tests
- [ ] End-to-end: prompt → MM-CODER → glyph → response
- [ ] Performance: tokens/sec on Intel HD 4600
- [ ] Accuracy: held-out test set perplexity < 20

### 4.3 System tests
- [ ] MM-CODER registered with coordinator
- [ ] Config system routes code queries to MM-CODER
- [ ] Reply pipeline merges glyph results
- [ ] Splash UI shows MM-CODER online

---

## Artifacts & Checkpoints

| Artifact | Owner | Status |
|----------|-------|--------|
| `models/smgm-16/scxq2_dds_folds/coder_tokens.bin` | Phase 1.1 | Pending |
| `khl/checkpoint/gpt2_coder_finetuned.pt` | Phase 1.2 | Pending |
| `models/gguf/gpt2_coder.Q8_0.gguf` | Phase 1.2 | Pending |
| `micronaut-coder/src/main.cpp` | Phase 2.1 | Pending |
| `micronaut-coder/src/token_signal_generator.cpp` | Phase 2.2 | Pending |
| `micronaut/micronaut.registry.xjson` | Phase 3.1 | Pending |
| Integration test suite | Phase 4.2 | Pending |

---

## Dependencies & Blockers

- ✅ Training pipeline ready (`train_loop.py`, tokenize_dataset.py)
- ✅ GPU shaders installed (13 HLSL files in native/shaders/gpt2/)
- ✅ C++ build system ready (native/scx_runtime/)
- ✅ Config system ready (KXML + tool dispatcher)
- ⏳ Coding corpus (need to gather or provide)
- ⏳ C++ inference boilerplate (GGUF loader, HTTP server)

---

## Questions for User

1. **Coding corpus:** Do you have code samples ready, or should I gather from releases/?
2. **C++ chunks:** Do you have existing MM-CODER inference code (GGUF loader, HTTP server)?
3. **Token signal generator:** What glyph opcodes should code execution produce? (e.g., SKILL_PYTHON, GPU_MATMUL, etc.)
4. **Timeline:** 2-3 day sprint, or break into smaller chunks?

---

## Next Steps

1. User provides coding chunks (corpus, C++ boilerplate, etc.)
2. Begin Phase 1.1: Prepare corpus
3. Phase 1.2: Fine-tune GPT-2 on coding data
4. Phase 2: Build native inference engine
5. Phase 3: Register and integrate

**Ready when you are. What coding chunks do you have?**
