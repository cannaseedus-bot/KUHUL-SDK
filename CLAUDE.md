# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## References

@kuhul-es/README.md
@kuhul-es/kuhul_pi_grammar.ebnf

## Project Overview

K'UHUL is an **app/program builder system** with its own opcode-sugar syntax. Any K'UHUL runtime component — class, micronaut, model, agent, bot, skill, tool — can author and consume this syntax. The native layer is C++ with DirectWrite for glyph rendering. Python style follows muPY (MicroPython) conventions. The ECMAScript layer uses the `kuhul-es` compiler (`npm i kuhul-es`), which is TypeScript-compatible.

## Build

```powershell
# Full native build (Windows, requires VS 2022 BuildTools)
.\build_all.ps1

# Node services only
cd micronaut && npm install

# C++ kernel only
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

## Lint

```powershell
# Node (from micronaut/ — requires: npm install)
npm run lint

# Python
ruff check .

# C++ (per-file)
clang-format --dry-run --Werror <file.cpp>
```

## Tests

```powershell
npm test           # smoke + native + cpp + python (run from micronaut/)
npm run smoke      # runtime health only
npm run test:cpp   # cd native/semantic_kernel_cpp/build && Release\unit_tests.exe
npm run test:python  # python/tests/connector_tests.py
```

## Starting the System

```bat
.\KuhulCLI.bat     # Orchestrates full startup: Splash → Coordinator → services → GGUF → UI
```

Splash server requires a one-time admin step: `.\add-kuhul-host.ps1` (adds `kuhul.local` to hosts).

## Micronauts vs Tools — Critical Distinction

**Real micronauts** (`micronaut/micronaut.registry.xjson`) — three, currently being built:
- `MM-MATH` — MathMicronaut, GPT-2 fine-tuned on math, at `E:\models\GPT2\math-GPT`
- `MM-CODER` — CoderMicronaut, native C++ engine (`micronaut-coder/`), GPT-2 coder fine-tune
- `MM-TOOLCALL` — ToolcallMicronaut, GPT-2 medium fine-tuned on toolcall + geodesic ARC attention

All three are `MM-1` variants (token_signal_generator, `⟁COMPUTE_FOLD⟁`) from the S7-LLM canonical system. All three can chat.

**Tools** — everything in `commands/*.bots.py`, `commands/micronaut_registry.json`, and `micronaut/services/` are tools (service agents), not micronauts. They will be wired as `[[micronaut.tools]]` entries once the micronauts are complete. Do not call them micronauts.

**Fold registry**: `micronaut/folds.toml` — canonical 15-fold system from S7-LLM. Authoritative. Mutation forbidden.

## Port Map

| Service | Port |
|---------|------|
| Coordinator (CO-1) | 25100 |
| GGUF inference | 5000 |
| DDS tensor server | 5001 |
| Splash / UI | 25200 |
| Micronauts 1-8 | 25101–25108 |
| Data Harvester | 25109 |
| Learning Engine | 25110 |

All ports are fixed. Services must register with Coordinator via `POST /register` before routing.

## Required Environment Variables

```
GGUF_ENDPOINT        = http://localhost:5000/v1/chat
ORCHESTRATOR_API_KEY = kuhul-v0.7-verified-access
GGUF_MODEL           = models/gguf/gpt2.Q8_0.gguf
REGISTRY             = micronaut/unified-registry.json
PYTHONPATH           = /micronaut/python
```

## Critical Gotchas

- **ESM only**: All Node files use `.mjs` extension — no CommonJS `require()`.
- **BSON serialization**: Micronaut payloads use BSON (not JSON) for inter-service messages.
- **Deterministic routing**: Same `(intent, params)` must always resolve to the same action — hash-verified. Do not introduce non-determinism in routing logic.
- **SQLite WAL mode**: `sessions.db` runs in WAL mode for concurrency — do not disable or change journal mode.
- **GGUF/DDS as PS jobs**: The GGUF and DDS servers run as PowerShell background jobs to survive health check timeouts; don't convert them to inline processes.
- **ISO 8601 UTC**: All timestamps across all languages must be ISO 8601 UTC format.

## Code Style

- **K'UHUL syntax**: Opcode sugar — universal across all runtime components. Don't mistake it for standard Python, JS, or C++; it has its own semantics.
- **`kuhul-es` is both the ECMAScript compiler AND the physics engine.** Entropy, fields, curvature, and collapse are enforced physics invariants — not naming conventions.
- **Physics invariants (non-negotiable):** `entropy = 0.21` (fixed), `collapse_only`, `field_perception` (read-only), `compression_law`, `unreachable_states`.
- **Six canonical glyphs:** `Sek` (output), `Pop` (stack pop), `Wo` (write/source), `Ch'en` (read/sink), `Yax` (conditional projection), `Xul` (terminate/collapse).
- **Binding types:** `π` = immutable/frozen · `τ` = temporal/mutable with full history tracking.
- **No branching or mutation in the π (enforcement) domain** — only perception, collapse, and output.
- **Compilation pipeline:** `.kuhules` source → TS AST → KUHULProgram IR → ASX (JSON) → KPI (binary) → K-UX projection (CSS/DOM/SVG/Canvas/Terminal).
- **Python**: muPY (MicroPython) style — lean, minimal runtime assumptions. Not standard CPython/Pydantic patterns.
- **ECMAScript / TypeScript**: Compiled via `kuhul-es` (`npm i kuhul-es`, repo: https://github.com/cannaseedus-bot/KUHUL-ES.git). TypeScript-compatible. Symbols and objects are primary constructs.
- **C++**: DirectWrite for native glyph rendering; custom allocators for geometric tensors; AVX2 SIMD for iGPU paths.
- **PowerShell**: Use `HKCU:\Software\...` registry path format (not `HKEY_CURRENT_USER\`).
