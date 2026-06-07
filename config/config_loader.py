"""
config_loader.py — Unified Configuration Loader (KXML + Phase 1 JSON)

Merges KXML domain specifications (canonical source of truth) with
Phase 1 JSON configs (personas, policies, emotions, flairs, reply pipeline).

KXML domains D0-D5 are mapped to personas, and tool bindings from KXML
become the canonical tool list. This unified system bridges the TypeScript
kuhul-es compiler with the Python runtime layer.

Data flow:
  KXML (kxml_domain_loader.py) → Personas (mapped from domains)
  meta-intent-map.json → Tool routing (via reply_pipeline)
  Policies + Emotions + Flairs → Personality and constraints
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

# Lazy imports to avoid circular dependency issues
# These are imported inside methods that need them
KXMLDomain = None  # Will be set on first use


# ============================================================================
# Typed Data Classes
# ============================================================================

@dataclass
class Persona:
    name: str
    flair_id: int
    policy_profile: str
    capabilities: List[str]
    default_emotion: str


@dataclass
class Policy:
    name: str
    description: str
    blocked_topics: List[str]
    allowed_actions: List[str]
    block_response: str
    environment: Optional[str] = None


@dataclass
class Emotion:
    name: str
    response_modifiers: Dict[str, float]


@dataclass
class Flair:
    id: int
    prefix: str
    suffix: str
    tone: str
    temperature_override: float
    use_formatting: str
    punctuation_style: str


@dataclass
class Tool:
    name: str
    approval_required: bool
    languages: List[str]
    targets: Optional[List[str]] = None


@dataclass
class Command:
    name: str
    flair_id: int


@dataclass
class ReplyPipeline:
    steps: List[str]
    hardcoded_triggers: Dict[str, Dict[str, Any]]
    response_format: Dict[str, Any]


# ============================================================================
# Config Loader
# ============================================================================

class ConfigLoader:
    """Canonical configuration loader for K'UHUL runtime components."""

    def __init__(self, config_dir: str = None):
        """
        Initialize config loader.

        Args:
            config_dir: Path to config directory. Defaults to directory of this file.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent

        self.config_dir = Path(config_dir)
        self._personas: Dict[str, Persona] = {}
        self._policies: Dict[str, Policy] = {}
        self._emotions: Dict[str, Emotion] = {}
        self._flairs: Dict[int, Flair] = {}
        self._tools: Dict[str, Tool] = {}
        self._commands: List[Command] = []
        self._reply_pipeline: Optional[ReplyPipeline] = None

        self._load_all()

    def _load_all(self) -> None:
        """Load all config files and validate."""
        # Load flairs first (needed for persona flair mapping)
        self._load_flairs()
        # Load emotions and policies next
        self._load_emotions()
        self._load_policies()
        # Load personas and tools (which depend on flairs)
        self._load_personas()
        self._load_tools()
        # Load commands and reply pipeline
        self._load_commands()
        self._load_reply_pipeline()
        # Validate everything
        self._validate()

    def _flair_name_to_id(self, flair_name: str) -> int:
        """Convert flair name to ID based on order in flairs (cached)."""
        if not hasattr(self, '_flair_name_cache'):
            self._flair_name_cache = {}
            data = self._load_file("flairs.json")
            flair_id = 0
            for name in data.get("flairs", {}).keys():
                self._flair_name_cache[name] = flair_id
                flair_id += 1

        return self._flair_name_cache.get(flair_name, 0)

    def _load_file(self, filename: str) -> Dict:
        """Load and parse a JSON config file."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            return json.load(f)

    def _load_emotions(self) -> None:
        """Load emotions.json"""
        data = self._load_file("emotions.json")

        for emotion_name in data.get("emotions", []):
            modifiers = data.get("response_modifiers", {}).get(emotion_name, {})
            self._emotions[emotion_name] = Emotion(
                name=emotion_name,
                response_modifiers=modifiers
            )

    def _load_policies(self) -> None:
        """Load policy.json"""
        data = self._load_file("policy.json")

        for policy_name, policy_spec in data.get("policies", {}).items():
            self._policies[policy_name] = Policy(
                name=policy_name,
                description=policy_spec.get("description", ""),
                blocked_topics=policy_spec.get("blocked_topics", []),
                allowed_actions=policy_spec.get("allowed_actions", []),
                block_response=policy_spec.get("block_response", ""),
                environment=policy_spec.get("environment")
            )

    def _load_flairs(self) -> None:
        """Load flairs.json"""
        data = self._load_file("flairs.json")

        flair_id = 0
        for flair_name, flair_spec in data.get("flairs", {}).items():
            self._flairs[flair_id] = Flair(
                id=flair_id,
                prefix=flair_spec.get("prefix", ""),
                suffix=flair_spec.get("suffix", ""),
                tone=flair_spec.get("tone", "neutral"),
                temperature_override=flair_spec.get("temperature_override", 0.7),
                use_formatting=flair_spec.get("use_formatting", "default"),
                punctuation_style=flair_spec.get("punctuation_style", "standard")
            )
            flair_id += 1

    def _load_personas(self) -> None:
        """Load personas from KXML domains + personas.json (legacy)"""
        # Lazy import to avoid circular dependencies
        from kxml_domain_loader import get_all_domains

        # Primary: KXML domains are implicit personas
        for domain in get_all_domains():
            persona_id = f"domain_{domain.name}"
            flair_id = self._flair_name_to_id("technical_expert")  # Default flair for domain personas

            self._personas[persona_id] = Persona(
                name=domain.name,
                flair_id=flair_id,
                policy_profile="dev_friendly",  # Domains default to dev-friendly policy
                capabilities=[t.name for t in domain.tools],
                default_emotion="focused"
            )

        # Secondary: Load legacy personas.json (for agent archetypes)
        try:
            data = self._load_file("personas.json")
            for persona_spec in data.get("personas", []):
                persona_id = persona_spec.get("id", "")
                if not persona_id:
                    continue

                # Convert flair_id from string name to index
                flair_id = self._flair_name_to_id(persona_spec.get("flair_id", "technical_expert"))

                # Only add if not already in KXML (KXML is canonical)
                if persona_id not in self._personas:
                    self._personas[persona_id] = Persona(
                        name=persona_spec.get("name", persona_id),
                        flair_id=flair_id,
                        policy_profile=persona_spec.get("policy_profile", ""),
                        capabilities=persona_spec.get("capabilities", []),
                        default_emotion=persona_spec.get("default_emotion", "neutral")
                    )
        except FileNotFoundError:
            pass  # personas.json is optional if KXML is used

    def _load_tools(self) -> None:
        """Load tools from KXML domains (canonical source) + tools.json (legacy)"""
        # Lazy import to avoid circular dependencies
        from kxml_domain_loader import get_all_domains

        # Primary: KXML domains have tool bindings
        for domain in get_all_domains():
            for tool_binding in domain.tools:
                # Create tool from KXML binding
                self._tools[tool_binding.name] = Tool(
                    name=tool_binding.name,
                    approval_required=False,  # KXML doesn't specify this
                    languages=[],  # KXML doesn't specify languages
                    targets=[f"D{domain.id}"]  # Available in this domain
                )

        # Secondary: Load legacy tools.json (in case there are additional tools)
        try:
            data = self._load_file("tools.json")
            tools_data = data.get("tools", [])
            if isinstance(tools_data, dict):
                tools_data = tools_data.items()
            elif isinstance(tools_data, list):
                tools_data = [(spec.get("id", i), spec) for i, spec in enumerate(tools_data)]

            for tool_id, tool_spec in tools_data:
                # Only add if not already in KXML (KXML is canonical)
                if tool_id not in self._tools:
                    tool_name = tool_spec.get("name", tool_spec.get("id", tool_id))
                    self._tools[tool_id] = Tool(
                        name=tool_name,
                        approval_required=tool_spec.get("requires_approval", False),
                        languages=tool_spec.get("languages", []),
                        targets=tool_spec.get("models")
                    )
        except FileNotFoundError:
            pass  # tools.json is optional if KXML is used

    def _load_commands(self) -> None:
        """Load commands.json"""
        data = self._load_file("commands.json")

        for cmd_name, cmd_spec in data.get("commands", {}).items():
            # Commands don't have explicit flair_id, so use default (0)
            self._commands.append(Command(
                name=cmd_name,
                flair_id=0
            ))

    def _load_reply_pipeline(self) -> None:
        """Load reply-structure.json"""
        data = self._load_file("reply-structure.json")

        self._reply_pipeline = ReplyPipeline(
            steps=data.get("response_pipeline", []),
            hardcoded_triggers=data.get("hardcoded_triggers", {}),
            response_format=data.get("response_format", {})
        )

    def _validate(self) -> None:
        """Validate all cross-references."""
        errors = []

        # Validate personas reference valid policies and emotions
        for persona_name, persona in self._personas.items():
            if persona.policy_profile not in self._policies:
                errors.append(f"Persona '{persona_name}' references unknown policy '{persona.policy_profile}'")
            if persona.default_emotion not in self._emotions:
                errors.append(f"Persona '{persona_name}' references unknown emotion '{persona.default_emotion}'")
            if persona.flair_id not in self._flairs:
                errors.append(f"Persona '{persona_name}' references unknown flair ID {persona.flair_id}")

        # Validate flairs exist for all referenced IDs
        for cmd in self._commands:
            if cmd.flair_id not in self._flairs:
                errors.append(f"Command '{cmd.name}' references unknown flair ID {cmd.flair_id}")

        # Validate reply-structure hardcoded triggers reference valid emotions
        if self._reply_pipeline:
            for trigger_name, trigger_spec in self._reply_pipeline.hardcoded_triggers.items():
                emotion = trigger_spec.get("emotion")
                if emotion and emotion not in self._emotions:
                    errors.append(f"Hardcoded trigger '{trigger_name}' references unknown emotion '{emotion}'")

        if errors:
            raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    # ========================================================================
    # Public API
    # ========================================================================

    def get_persona(self, name: str) -> Optional[Persona]:
        """Get persona by name."""
        return self._personas.get(name)

    def get_all_personas(self) -> Dict[str, Persona]:
        """Get all personas."""
        return self._personas.copy()

    def get_policy(self, profile: str) -> Optional[Policy]:
        """Get policy by profile name."""
        return self._policies.get(profile)

    def get_all_policies(self) -> Dict[str, Policy]:
        """Get all policies."""
        return self._policies.copy()

    def get_emotion(self, name: str) -> Optional[Emotion]:
        """Get emotion by name."""
        return self._emotions.get(name)

    def get_all_emotions(self) -> Dict[str, Emotion]:
        """Get all emotions."""
        return self._emotions.copy()

    def get_flair(self, flair_id: int) -> Optional[Flair]:
        """Get flair by ID."""
        return self._flairs.get(flair_id)

    def get_all_flairs(self) -> Dict[int, Flair]:
        """Get all flairs."""
        return self._flairs.copy()

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> Dict[str, Tool]:
        """Get all tools."""
        return self._tools.copy()

    def get_commands(self) -> List[Command]:
        """Get all commands."""
        return self._commands.copy()

    def get_reply_pipeline(self) -> Optional[ReplyPipeline]:
        """Get the reply structure pipeline."""
        return self._reply_pipeline

    def get_domain(self, domain_id: int):
        """Get KXML domain by ID (0-5)."""
        from kxml_domain_loader import get_domain
        return get_domain(domain_id)

    def get_all_domains(self):
        """Get all KXML domains (D0-D5)."""
        from kxml_domain_loader import get_all_domains
        return get_all_domains()

    def classify_text(self, text: str) -> tuple[int, float]:
        """
        Classify text to a domain using KXML keywords.
        Returns (domain_id, confidence).
        """
        from kxml_domain_loader import classify_domain
        return classify_domain(text)

    def summary(self) -> str:
        """Return a summary of loaded config."""
        return (
            f"Config valid: "
            f"{len(self._personas)} personas, "
            f"{len(self._policies)} policies, "
            f"{len(self._emotions)} emotions, "
            f"{len(self._flairs)} flairs, "
            f"{len(self._tools)} tools, "
            f"{len(self._commands)} commands, "
            f"6 KXML domains (D0-D5)"
        )


# ============================================================================
# Singleton Instance (for module-level access)
# ============================================================================

_loader: Optional[ConfigLoader] = None


def initialize(config_dir: str = None) -> ConfigLoader:
    """Initialize the global config loader."""
    global _loader
    _loader = ConfigLoader(config_dir)
    return _loader


def get_loader() -> ConfigLoader:
    """Get the global config loader (or initialize if not done)."""
    global _loader
    if _loader is None:
        _loader = ConfigLoader()
    return _loader


# Convenience exports (use after initialize())
def get_persona(name: str) -> Optional[Persona]:
    return get_loader().get_persona(name)


def get_policy(profile: str) -> Optional[Policy]:
    return get_loader().get_policy(profile)


def get_emotion(name: str) -> Optional[Emotion]:
    return get_loader().get_emotion(name)


def get_flair(flair_id: int) -> Optional[Flair]:
    return get_loader().get_flair(flair_id)


def get_tool(name: str) -> Optional[Tool]:
    return get_loader().get_tool(name)


def get_commands() -> List[Command]:
    return get_loader().get_commands()


def get_reply_pipeline() -> Optional[ReplyPipeline]:
    return get_loader().get_reply_pipeline()


# KXML domain access
def get_domain(domain_id: int):
    """Get KXML domain by ID (canonical)."""
    return get_loader().get_domain(domain_id)


def get_all_domains(self=None):
    """Get all KXML domains (D0-D5)."""
    return get_loader().get_all_domains()


def classify_text(text: str) -> tuple[int, float]:
    """Classify text to a domain using KXML keywords."""
    return get_loader().classify_text(text)


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    import sys
    try:
        loader = initialize()
        print(f"[OK] {loader.summary()}")
        print(f"[OK] All config cross-references valid")
    except Exception as e:
        print(f"[ERROR] Config error: {e}", file=sys.stderr)
        exit(1)
