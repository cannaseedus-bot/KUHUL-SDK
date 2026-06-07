from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GLYPHS = {
    "◯": "base_field",
    "π": "pi_geometry",
    "⟁": "fold_compute",
    "µ": "scale",
    "⊗": "tensor_product",
    "⚡": "energy",
    "✓": "validation",
    "→": "transform",
    "↔": "composition",
    "⇄": "feedback",
    "⥂": "parallel_execution",
    "⥀": "reversible",
    "⟷": "bidirectional_flow",
}

FIELD_GLYPHS = {
    "wind": "→ ⟁ µ",
    "attraction_well": "◯ ⊗ π",
    "navigation_force": "⥂ ⟷ ⊗",
    "scroll_inertia": "⇄ π ⚡",
}

class KuhulDirectXNative:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.glyphs = DEFAULT_GLYPHS
        self.field_specs = self._load_field_specs()
        # Fix: native_glyph_engine.exe is in releases/v0.2.0/.../hive-runtime/Kuhul-cpp/
        releases_path = self.root / "releases" / "v0.2.0-kuhul-directx-native" / "hive-runtime" / "Kuhul-cpp"
        self.native_engine = releases_path / "native_glyph_engine.exe"
        self.glyph_shader = self.root / "directx" / "shaders" / "compiled" / "glyph_compute.cso"

    def _load_field_specs(self):
        specs = {}
        for path in sorted((self.root / "field_system" / "specs").glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                specs[path.stem] = json.load(handle)
        return specs

    def validate_glyph_program(self, glyph_sequence: str) -> list[str]:
        glyphs = glyph_sequence.split()
        unknown = [glyph for glyph in glyphs if glyph not in self.glyphs]
        if unknown:
            raise ValueError(f"unknown KUHUL glyph(s): {unknown}")
        return glyphs

    def field_to_glyph_program(self, field_type: str) -> str:
        return FIELD_GLYPHS.get(field_type, "⟁")

    def inspect_native_engine(self) -> str:
        proc = subprocess.run([str(self.native_engine), "inspect"], capture_output=True, text=True, check=True)
        return proc.stdout

    def run_native_demo(self) -> str:
        proc = subprocess.run([str(self.native_engine), "run-demo"], capture_output=True, text=True, check=True)
        return proc.stdout

    def describe(self) -> dict:
        return {
            "field_spec_count": len(self.field_specs),
            "glyph_count": len(self.glyphs),
            "native_engine": str(self.native_engine),
            "glyph_shader": str(self.glyph_shader),
            "glyph_shader_exists": self.glyph_shader.exists(),
            "integration_slots": {
                "glyph_to_gpu_compiler": str(self.root / "directx" / "glyph_to_gpu_compiler.exe"),
                "directwrite_font": str(self.root / "directx" / "kuhul_glyph_set.ttf"),
            },
        }

if __name__ == "__main__":
    runtime = KuhulDirectXNative()
    print(json.dumps(runtime.describe(), indent=2))
    print(runtime.run_native_demo())
