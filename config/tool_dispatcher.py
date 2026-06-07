"""
tool_dispatcher.py — Unified Tool Dispatcher

Routes tool calls to working release code:
- code_executor → releases/Kuhul-PY/code_interpreter.py (sandboxed Python)
- model_invoker → releases/Kuhul-PY/model_manager.py (HuggingFace LLM)
- fold_executor → HTTP POST to glyph WS (port 8082)
- glyph_dispatcher → HTTP POST to glyph WS (port 8082)

Dispatcher entry point: dispatch_tool(tool_name, params) -> dict
"""

import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Add releases to path for imports
RELEASES_PATH = Path(__file__).parent.parent / "releases"
SYS_PATH_ORIG = sys.path.copy()


def _import_from_release(module_name: str, release_dir: str):
    """Dynamically import from releases directory."""
    rel_path = RELEASES_PATH / release_dir
    if str(rel_path) not in sys.path:
        sys.path.insert(0, str(rel_path))
    return __import__(module_name)


# ============================================================================
# Tool Handlers (each returns dict with result)
# ============================================================================

def execute_code(code: str, language: str = "python", timeout: int = 5) -> Dict[str, Any]:
    """Execute code in sandboxed environment."""
    try:
        code_interp = _import_from_release("code_interpreter", "Kuhul-PY")
        result = code_interp.run_python_code(code, timeout)
        return {
            "status": "success",
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "exit_code": result.get("exit_code", 0)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "exit_code": 1
        }


def invoke_model(prompt: str, model_id: Optional[str] = None) -> Dict[str, Any]:
    """Invoke HuggingFace model via ModelManager."""
    try:
        model_mgr = _import_from_release("model_manager", "Kuhul-PY")
        singleton = model_mgr.ModelManagerSingleton()
        response = singleton.generate_response(prompt, model_id or "default")
        return {
            "status": "success",
            "response": response,
            "model": model_id or "default"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def execute_fold(operation: str, field_curvature: Optional[float] = None) -> Dict[str, Any]:
    """Execute fold geometry operation via glyph WS."""
    try:
        import urllib.request
        from kxml_domain_loader import get_glyph_for_tool, get_micronaut_target

        glyph = get_glyph_for_tool("fold_executor")
        if not glyph:
            return {"status": "error", "error": "fold_executor tool not found in glyph map"}

        target = get_micronaut_target(glyph)
        if not target:
            return {"status": "error", "error": f"No target for glyph {glyph}"}

        # POST to glyph WS server
        payload = {
            "glyph": glyph,
            "operation": operation,
            "field_curvature": field_curvature or 0.0
        }

        req = urllib.request.Request(
            "http://127.0.0.1:8082/glyph",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read())
            return {
                "status": "success",
                "result": result,
                "target": target
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def dispatch_glyph(glyph_code: str, args: Optional[Dict] = None) -> Dict[str, Any]:
    """Dispatch glyph opcode via WebSocket."""
    try:
        import urllib.request
        from kxml_domain_loader import get_micronaut_target

        target = get_micronaut_target(glyph_code)
        if not target:
            return {"status": "error", "error": f"No target for glyph {glyph_code}"}

        payload = {
            "glyph": glyph_code,
            "args": args or {}
        }

        req = urllib.request.Request(
            "http://127.0.0.1:8082/glyph",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read())
            return {
                "status": "success",
                "result": result,
                "target": target
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# Main Dispatcher
# ============================================================================

TOOL_HANDLERS = {
    "code_executor": lambda params: execute_code(**params),
    "model_invoker": lambda params: invoke_model(**params),
    "fold_executor": lambda params: execute_fold(**params),
    "glyph_dispatcher": lambda params: dispatch_glyph(**params),
    # Add more tools as needed
}


def dispatch_tool(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch tool call to appropriate handler.

    Args:
        tool_name: Name of tool (e.g., "code_executor")
        params: Tool-specific parameters

    Returns:
        Dict with status, result, and any errors
    """
    if tool_name not in TOOL_HANDLERS:
        return {
            "status": "error",
            "error": f"Unknown tool: {tool_name}",
            "available_tools": list(TOOL_HANDLERS.keys())
        }

    try:
        return TOOL_HANDLERS[tool_name](params)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Tool dispatch failed: {e}",
            "tool": tool_name
        }


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("[TEST] Tool Dispatcher")
    print("=" * 60)

    # Test 1: Code executor
    print("\nTest 1: code_executor")
    result = dispatch_tool("code_executor", {"code": "print(2 + 2)"})
    print(f"  Status: {result.get('status')}")
    print(f"  Output: {result.get('stdout', '').strip()}")

    # Test 2: Fold executor (will fail if glyph WS not running, but that's ok)
    print("\nTest 2: fold_executor")
    result = dispatch_tool("fold_executor", {"operation": "compress", "field_curvature": 0.5})
    print(f"  Status: {result.get('status')}")
    if result.get("status") == "error":
        print(f"  Error: {result.get('error')} (expected if WS server not running)")

    # Test 3: Unknown tool
    print("\nTest 3: Unknown tool")
    result = dispatch_tool("unknown_tool", {})
    print(f"  Status: {result.get('status')}")
    print(f"  Error: {result.get('error')}")

    print("\n" + "=" * 60)
    print("[OK] Tool dispatcher tests complete")
