"""
kxml_domain_loader.py — KXML Domain Specifications (Python Port)

This is the canonical source of truth for K'UHUL domains.
Mirrors kuhul-es/compiler/src/kxml_parser.ts in Python.

Domains D0-D5 define:
- Classification keywords
- Tool bindings (meta-intent-map)
- DDS adapter shard references
- Training parameters

Glyph Opcode System:
- Loads glyph_opcode_system.xjson for PUA codepoint → opcode → micronaut routing
- Maps tools to glyphs: get_glyph_for_tool(tool_name) → PUA codepoint
- Maps glyphs to targets: get_micronaut_target(glyph) → micronaut port/alias
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import json
from pathlib import Path


@dataclass
class ToolBinding:
    """Tool available in a domain."""
    name: str
    description: str
    call_signature: str  # e.g. "code_executor(code, analysis_type)"
    output_format: str = "text"  # text | code | json
    confidence: float = 0.8  # 0-1 estimate of tool usefulness


@dataclass
class AdapterRef:
    """Reference to a DDS R32F adapter shard."""
    shard_index: int  # 98-103 for D0-D5
    rank: int = 8  # LoRA rank
    format_variant: str = "R32F"
    float_type: str = "float32"


@dataclass
class TrainingParams:
    """Training configuration per domain."""
    mutation_rate: float  # 0.002-0.010
    mutation_limit: float  # 4x mutation rate
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None


@dataclass
class KXMLDomain:
    """Canonical K'UHUL domain specification."""
    id: int  # 0-5
    name: str
    description: str
    keywords: List[str]
    tools: List[ToolBinding]
    adapter: AdapterRef
    training_params: Optional[TrainingParams] = None


# ============================================================================
# CANONICAL DOMAIN SPECIFICATIONS (D0-D5)
# ============================================================================

CANONICAL_DOMAINS: List[KXMLDomain] = [
    # D0: K'UHUL Source Semantic
    KXMLDomain(
        id=0,
        name="kuhul_source_semantic",
        description="K'UHUL language, glyph operations, fold geometry, semantic syntax",
        keywords=[
            "kuhul", "glyph", "xcfe", "micronaut", "brain", "opcode",
            "khl", "xcfg", "fold", "sek", "xul", "wo", "pop"
        ],
        tools=[
            ToolBinding(
                name="fold_executor",
                description="Execute fold geometry operations",
                call_signature="fold_executor(operation, field_curvature)",
                confidence=0.9
            ),
            ToolBinding(
                name="glyph_dispatcher",
                description="Dispatch K'UHUL glyphs (Sek, Pop, Wo, Ch'en, Yax, Xul)",
                call_signature="glyph_dispatcher(glyph_name, args)",
                confidence=0.95
            ),
            ToolBinding(
                name="semantic_parser",
                description="Parse K'UHUL semantics and π/τ bindings",
                call_signature="semantic_parser(source_code)",
                confidence=0.85
            ),
        ],
        adapter=AdapterRef(shard_index=98, rank=8, format_variant="R32F"),
        training_params=TrainingParams(mutation_rate=0.003, mutation_limit=0.012)
    ),

    # D1: Semantic Code Math
    KXMLDomain(
        id=1,
        name="semantic_code_math",
        description="Code semantics, embeddings, transformers, attention mechanisms, language models",
        keywords=[
            "semantic", "token", "embedding", "transformer", "attention",
            "language", "model", "inference", "context", "prompt"
        ],
        tools=[
            ToolBinding(
                name="code_executor",
                description="Execute semantic code analysis",
                call_signature="code_executor(code, analysis_type)",
                confidence=0.8
            ),
            ToolBinding(
                name="semantic_analyzer",
                description="Analyze code semantics",
                call_signature="semantic_analyzer(source_code)",
                confidence=0.85
            ),
            ToolBinding(
                name="embedding_model",
                description="Generate embeddings from text",
                call_signature="embedding_model(text)",
                confidence=0.8
            ),
        ],
        adapter=AdapterRef(shard_index=99, rank=8, format_variant="R32F"),
        training_params=TrainingParams(mutation_rate=0.002, mutation_limit=0.008)
    ),

    # D2: Algorithmic Code
    KXMLDomain(
        id=2,
        name="algorithmic_code",
        description="Algorithms, data structures, computational complexity, sorting, searching",
        keywords=[
            "algorithm", "sort", "search", "tree", "graph", "hash",
            "complexity", "dynamic_programming", "recursion", "iteration"
        ],
        tools=[
            ToolBinding(
                name="algorithm_runner",
                description="Run algorithmic computations",
                call_signature="algorithm_runner(algorithm_name, inputs)",
                confidence=0.8
            ),
            ToolBinding(
                name="complexity_analyzer",
                description="Analyze time/space complexity",
                call_signature="complexity_analyzer(algorithm_code)",
                confidence=0.75
            ),
            ToolBinding(
                name="data_structure_tool",
                description="Implement and verify data structures",
                call_signature="data_structure_tool(ds_type, operations)",
                confidence=0.8
            ),
        ],
        adapter=AdapterRef(shard_index=100, rank=8, format_variant="R32F"),
        training_params=TrainingParams(mutation_rate=0.002, mutation_limit=0.008)
    ),

    # D3: Mathematical Analysis
    KXMLDomain(
        id=3,
        name="mathematical_analysis",
        description="Mathematics, calculus, linear algebra, statistics, differential equations",
        keywords=[
            "math", "equation", "integral", "derivative", "matrix",
            "vector", "probability", "statistics", "algebra", "geometry"
        ],
        tools=[
            ToolBinding(
                name="math_solver",
                description="Solve mathematical equations",
                call_signature="math_solver(equation, variable)",
                confidence=0.85
            ),
            ToolBinding(
                name="symbolic_engine",
                description="Symbolic mathematics (differentiation, integration)",
                call_signature="symbolic_engine(operation, expression)",
                confidence=0.8
            ),
            ToolBinding(
                name="numerical_tool",
                description="Numerical computation and simulation",
                call_signature="numerical_tool(algorithm, parameters)",
                confidence=0.8
            ),
        ],
        adapter=AdapterRef(shard_index=101, rank=8, format_variant="R32F"),
        training_params=TrainingParams(mutation_rate=0.003, mutation_limit=0.012)
    ),

    # D4: GPU/Shader Code
    KXMLDomain(
        id=4,
        name="gpu_shader_code",
        description="HLSL, GLSL, WGSL shaders, GPU compute, parallel algorithms, graphics programming",
        keywords=[
            "shader", "gpu", "hlsl", "glsl", "wgsl", "compute", "parallel",
            "texture", "buffer", "kernel", "renderpass", "directx", "vulkan"
        ],
        tools=[
            ToolBinding(
                name="shader_compiler",
                description="Compile and validate shaders",
                call_signature="shader_compiler(shader_code, target_platform)",
                confidence=0.9
            ),
            ToolBinding(
                name="gpu_executor",
                description="Execute GPU compute kernels",
                call_signature="gpu_executor(kernel_code, work_size)",
                confidence=0.85
            ),
            ToolBinding(
                name="graphics_optimizer",
                description="Optimize graphics pipelines",
                call_signature="graphics_optimizer(pipeline_spec)",
                confidence=0.7
            ),
        ],
        adapter=AdapterRef(shard_index=102, rank=8, format_variant="R32F"),
        training_params=TrainingParams(mutation_rate=0.010, mutation_limit=0.040)  # GPU-heavy exploration
    ),

    # D5: Cryptographic Math
    KXMLDomain(
        id=5,
        name="cryptographic_math",
        description="Cryptography, hashing, elliptic curves, number theory, quantum-safe algorithms",
        keywords=[
            "crypto", "hash", "encrypt", "decrypt", "prime", "rsa", "ecdsa",
            "quantum", "lattice", "merkle", "signature", "key_exchange"
        ],
        tools=[
            ToolBinding(
                name="cryptography_library",
                description="Cryptographic operations (encrypt, sign, hash)",
                call_signature="cryptography_library(operation, key_params)",
                confidence=0.9
            ),
            ToolBinding(
                name="number_theory_tool",
                description="Number theory algorithms",
                call_signature="number_theory_tool(algorithm, inputs)",
                confidence=0.85
            ),
            ToolBinding(
                name="quantum_simulator",
                description="Quantum-resistant algorithm evaluation",
                call_signature="quantum_simulator(algorithm_spec)",
                confidence=0.7
            ),
        ],
        adapter=AdapterRef(shard_index=103, rank=8, format_variant="R32F"),
        training_params=TrainingParams(mutation_rate=0.003, mutation_limit=0.012)
    ),
]


# ============================================================================
# Public API
# ============================================================================

def get_domain(domain_id: int) -> Optional[KXMLDomain]:
    """Get domain by ID (0-5)."""
    if 0 <= domain_id < len(CANONICAL_DOMAINS):
        return CANONICAL_DOMAINS[domain_id]
    return None


def get_domain_by_name(name: str) -> Optional[KXMLDomain]:
    """Get domain by name."""
    for domain in CANONICAL_DOMAINS:
        if domain.name == name:
            return domain
    return None


def get_all_domains() -> List[KXMLDomain]:
    """Get all canonical domains."""
    return CANONICAL_DOMAINS.copy()


def classify_domain(text: str) -> tuple[int, float]:
    """
    Classify text to a domain based on keyword matching.
    Returns (domain_id, confidence).
    """
    text_lower = text.lower()
    scores = {}

    for domain in CANONICAL_DOMAINS:
        matches = sum(1 for kw in domain.keywords if kw.lower() in text_lower)
        if matches > 0:
            confidence = matches / len(domain.keywords)
            scores[domain.id] = confidence

    if not scores:
        return 0, 0.0  # Default to D0

    best_domain = max(scores, key=scores.get)
    return best_domain, scores[best_domain]


# ============================================================================
# Glyph Opcode System (PUA codepoint routing)
# ============================================================================

# Lazy-loaded glyph opcode map
_glyph_opcode_map: Optional[Dict] = None
_glyph_to_tool: Optional[Dict[str, str]] = None  # glyph → tool_name
_tool_to_glyph: Optional[Dict[str, str]] = None  # tool_name → glyph


def _load_glyph_opcode_system() -> Optional[Dict]:
    """Load glyph_opcode_system.xjson from releases."""
    global _glyph_opcode_map, _glyph_to_tool, _tool_to_glyph

    if _glyph_opcode_map is not None:
        return _glyph_opcode_map

    try:
        xjson_path = Path(__file__).parent.parent / "releases" / "mx2lm-runtime" / "glyph_opcode_system.xjson"
        if not xjson_path.exists():
            return None

        with open(xjson_path) as f:
            _glyph_opcode_map = json.load(f)

        # Build reverse lookup tables
        _glyph_to_tool = {}
        _tool_to_glyph = {}

        for category, glyphs in _glyph_opcode_map.get("glyph_opcode_map", {}).items():
            for pua_code, spec in glyphs.items():
                opcode = spec.get("opcode", "")
                # Extract tool name from opcode (e.g., SKILL_PYTHON → "python", AGENT_THINK → "think")
                if "_" in opcode:
                    tool_name = opcode.split("_", 1)[1].lower()
                    _glyph_to_tool[pua_code] = tool_name
                    _tool_to_glyph[tool_name] = pua_code

        # Add default mappings for KXML domain tools (not in glyph opcode map)
        # These map to sensible glyphs or use fallback routes
        _domain_tool_defaults = {
            "fold_executor": "U+E040",  # Generic fold opcode (may not exist, but placeholder)
            "glyph_dispatcher": "U+E001",  # AGENT_ACT
            "semantic_parser": "U+E000",  # AGENT_THINK
            "code_executor": "U+E010",  # SKILL_PYTHON
            "semantic_analyzer": "U+E010",  # SKILL_PYTHON
            "embedding_model": "U+E010",  # SKILL_PYTHON
            "algorithm_runner": "U+E010",  # SKILL_PYTHON
            "complexity_analyzer": "U+E010",  # SKILL_PYTHON
            "data_structure_tool": "U+E010",  # SKILL_PYTHON
            "math_solver": "U+E010",  # SKILL_PYTHON
            "symbolic_engine": "U+E010",  # SKILL_PYTHON
            "numerical_tool": "U+E010",  # SKILL_PYTHON
            "shader_compiler": "U+E020",  # GPU operation
            "gpu_executor": "U+E020",  # GPU operation
            "graphics_optimizer": "U+E020",  # GPU operation
            "cryptography_library": "U+E010",  # SKILL_PYTHON
            "number_theory_tool": "U+E010",  # SKILL_PYTHON
            "quantum_simulator": "U+E010",  # SKILL_PYTHON
        }

        for tool_name, glyph in _domain_tool_defaults.items():
            if tool_name not in _tool_to_glyph:
                _tool_to_glyph[tool_name] = glyph

        return _glyph_opcode_map
    except Exception as e:
        print(f"Warning: Failed to load glyph_opcode_system.xjson: {e}")
        return None


def get_glyph_for_tool(tool_name: str) -> Optional[str]:
    """
    Get PUA codepoint for a tool name.
    E.g., get_glyph_for_tool("python") → "U+E010"
    """
    _load_glyph_opcode_system()
    if _tool_to_glyph is None:
        return None
    return _tool_to_glyph.get(tool_name.lower())


def get_micronaut_target(glyph: str) -> Optional[str]:
    """
    Get micronaut target (port/alias) for a glyph codepoint.
    E.g., get_micronaut_target("U+E010") → "PYIDE-1" or port 3189
    """
    xjson = _load_glyph_opcode_system()
    if xjson is None:
        return None

    # Search for glyph in all categories
    for category, glyphs in xjson.get("glyph_opcode_map", {}).items():
        spec = glyphs.get(glyph, {})
        if spec:
            # Return micronaut route if specified, else port from meta
            if "micronaut_route" in spec:
                return spec["micronaut_route"]
            # Fallback: use meta port
            return str(xjson.get("meta", {}).get("port", ""))

    return None


def get_all_glyph_opcodes() -> Dict:
    """Get entire glyph opcode map."""
    return _load_glyph_opcode_system() or {}


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("KXML Domain Specifications (D0-D5)")
    print("=" * 60)

    for domain in CANONICAL_DOMAINS:
        print(f"\nD{domain.id}: {domain.name}")
        print(f"  Description: {domain.description}")
        print(f"  Keywords: {', '.join(domain.keywords[:5])}...")
        print(f"  Tools: {', '.join(t.name for t in domain.tools)}")
        print(f"  Adapter: DDS_SHARD_{domain.adapter.shard_index} (R32F, rank {domain.adapter.rank})")

    print("\n" + "=" * 60)
    print("Testing domain classification...")
    test_cases = [
        "kuhul glyph fold sek xul",
        "embedding transformer semantic code",
        "sort algorithm tree structure",
        "integral derivative matrix algebra",
        "shader gpu hlsl compute",
        "rsa crypto encrypt hash",
    ]

    for text in test_cases:
        domain_id, conf = classify_domain(text)
        domain = get_domain(domain_id)
        print(f"'{text}' -> D{domain_id} ({domain.name}, {conf:.2%})")

    print("\n" + "=" * 60)
    print("Testing glyph opcode routing...")
    test_tools = ["python", "tensor", "think", "seal"]
    for tool in test_tools:
        glyph = get_glyph_for_tool(tool)
        if glyph:
            target = get_micronaut_target(glyph)
            print(f"'{tool}' -> {glyph} -> {target}")
        else:
            print(f"'{tool}' -> (not found)")
