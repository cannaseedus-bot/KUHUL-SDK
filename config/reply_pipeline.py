"""
reply_pipeline.py — 7-Step Reply Structure Pipeline

Implements the response pipeline defined in reply-structure.json:
1. policy_check — blocks if topic is in policy.blocked_topics
2. emotion_resolve — extracts modifiers from emotion
3. hardcoded_lookup — returns shortcut for pattern matches (bypass_tools=True)
4. tool_selection — picks tool from tools.json based on prompt
5. skill_invoke — calls tool endpoint
6. flair_apply — applies prefix/suffix + tone to response
7. format_output — final string formatting with emotion modifiers

All steps are optional; returns default response if all fail.
"""

import re
from typing import Optional, Dict, Any, Callable
from config_loader import (
    get_loader, Persona, Policy, Emotion, Flair, Tool, ReplyPipeline
)


class ReplyStructurePipeline:
    """7-step reply pipeline for K'UHUL agents."""

    def __init__(self):
        """Initialize with config loader."""
        self.config = get_loader()

    # ========================================================================
    # Step 1: Policy Check
    # ========================================================================

    def policy_check(self, prompt: str, persona: Persona) -> tuple[bool, Optional[str]]:
        """
        Check if prompt violates policy.

        Returns:
            (allowed: bool, block_response: str|None)
        """
        policy = self.config.get_policy(persona.policy_profile)
        if not policy:
            return True, None

        # Check if any blocked topic appears in prompt
        prompt_lower = prompt.lower()
        for topic in policy.blocked_topics:
            if topic.lower() in prompt_lower:
                return False, policy.block_response

        return True, None

    # ========================================================================
    # Step 2: Emotion Resolve
    # ========================================================================

    def emotion_resolve(self, persona: Persona) -> Dict[str, float]:
        """
        Resolve emotion modifiers for persona.

        Returns:
            Dictionary of response modifiers (exclamation_chance, formal_tone, etc.)
        """
        emotion = self.config.get_emotion(persona.default_emotion)
        if not emotion:
            return {}

        return emotion.response_modifiers.copy()

    # ========================================================================
    # Step 3: Hardcoded Lookup
    # ========================================================================

    def hardcoded_lookup(self, prompt: str) -> Optional[str]:
        """
        Check hardcoded triggers in reply-structure.json.

        Returns:
            Response text if pattern matches, None otherwise
        """
        pipeline = self.config.get_reply_pipeline()
        if not pipeline:
            return None

        prompt_lower = prompt.lower()

        for trigger_name, trigger_spec in pipeline.hardcoded_triggers.items():
            patterns = trigger_spec.get("patterns", [])

            # Check if any pattern matches
            for pattern in patterns:
                # Treat pattern as regex (case-insensitive)
                try:
                    if re.search(pattern, prompt_lower):
                        return trigger_spec.get("response", "")
                except re.error:
                    # Fallback: simple substring match
                    if pattern.lower() in prompt_lower:
                        return trigger_spec.get("response", "")

        return None

    # ========================================================================
    # Step 4: Tool Selection
    # ========================================================================

    def tool_selection(self, prompt: str) -> Optional[Tool]:
        """
        Select appropriate tool for prompt.

        Simple strategy: match prompt keywords to tool names.
        More sophisticated matching could use semantic similarity.

        Returns:
            Tool object if match found, None otherwise
        """
        prompt_lower = prompt.lower()
        tools = self.config.get_all_tools()

        for tool_name, tool in tools.items():
            if tool_name.lower() in prompt_lower:
                return tool

        return None

    # ========================================================================
    # Step 5: Skill Invoke
    # ========================================================================

    def skill_invoke(
        self,
        tool: Tool,
        prompt: str,
        tool_executor: Optional[Callable[[Tool, str], str]] = None
    ) -> Optional[str]:
        """
        Call tool endpoint.

        Special case: if tool is "model_invoker", use real ChatSession from Kuhul-PY.
        Otherwise, use provided tool_executor or dispatch via tool_dispatcher.

        Args:
            tool: Tool to invoke
            prompt: User prompt
            tool_executor: Optional callable(tool, prompt) → response string

        Returns:
            Tool response, or None if execution fails
        """
        # Special case: use ChatSession for model invocation
        if tool.name == "model_invoker":
            try:
                import sys
                from pathlib import Path
                releases = Path(__file__).parent.parent / "releases" / "Kuhul-PY"
                if str(releases) not in sys.path:
                    sys.path.insert(0, str(releases))

                from chat import ChatSession
                session = ChatSession()
                response = session.send(prompt)
                return response
            except Exception as e:
                return f"Model invocation error: {e}"

        # Default: use tool_executor or tool_dispatcher
        if tool_executor is not None:
            try:
                return tool_executor(tool, prompt)
            except Exception as e:
                return f"Tool error: {e}"

        # Fallback: use tool_dispatcher
        try:
            from tool_dispatcher import dispatch_tool
            result = dispatch_tool(tool.name, {"prompt": prompt})
            if result.get("status") == "success":
                return result.get("response") or result.get("result", "")
            else:
                return f"Tool error: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Tool dispatch error: {e}"

    # ========================================================================
    # Step 6: Flair Apply
    # ========================================================================

    def flair_apply(self, response: str, flair: Flair) -> str:
        """
        Apply flair prefix/suffix and tone to response.

        Returns:
            Flair-wrapped response
        """
        return f"{flair.prefix}{response}{flair.suffix}"

    # ========================================================================
    # Step 7: Format Output
    # ========================================================================

    def format_output(self, response: str, emotion_modifiers: Dict[str, float]) -> str:
        """
        Apply emotion-based formatting to response.

        Modifiers:
        - exclamation_chance: probability of adding '!'
        - formal_tone: scale formality (higher = more formal)
        - positive_words: scale cheerfulness
        - action_words: emphasize action/urgency
        - precision: scale technical clarity

        Returns:
            Formatted response
        """
        formatted = response

        # Exclamation injection (placeholder — actual would be probabilistic)
        if emotion_modifiers.get("exclamation_chance", 0) > 0.5:
            if not formatted.endswith("!"):
                formatted += "!"

        # Precision: could add technical markers, citations, etc.
        # (This is a placeholder for demonstrating the concept)

        return formatted

    # ========================================================================
    # Main Pipeline Execution
    # ========================================================================

    def execute(
        self,
        prompt: str,
        persona: Persona,
        tool_executor: Optional[Callable[[Tool, str], str]] = None
    ) -> Dict[str, Any]:
        """
        Execute the full 7-step pipeline.

        Returns:
            {
                "response": str,
                "emotion": str,
                "flair_applied": bool,
                "tool_used": str|None,
                "blocked": bool
            }
        """
        # Step 1: Policy Check
        allowed, block_response = self.policy_check(prompt, persona)
        if not allowed:
            return {
                "response": block_response,
                "emotion": persona.default_emotion,
                "flair_applied": False,
                "tool_used": None,
                "blocked": True
            }

        # Step 2: Emotion Resolve
        emotion_modifiers = self.emotion_resolve(persona)

        # Step 3: Hardcoded Lookup
        hardcoded = self.hardcoded_lookup(prompt)
        if hardcoded:
            return {
                "response": hardcoded,
                "emotion": persona.default_emotion,
                "flair_applied": False,
                "tool_used": None,
                "blocked": False
            }

        # Step 4: Tool Selection
        tool = self.tool_selection(prompt)

        # Step 5: Skill Invoke
        response = None
        tool_name = None
        if tool:
            response = self.skill_invoke(tool, prompt, tool_executor)
            tool_name = tool.name

        # Fallback if no tool or tool failed
        if not response:
            response = "I'm not sure how to respond to that. Could you provide more context?"

        # Step 6: Flair Apply
        flair = self.config.get_flair(persona.flair_id)
        flair_applied = False
        if flair:
            response = self.flair_apply(response, flair)
            flair_applied = True

        # Step 7: Format Output
        response = self.format_output(response, emotion_modifiers)

        return {
            "response": response,
            "emotion": persona.default_emotion,
            "flair_applied": flair_applied,
            "tool_used": tool_name,
            "blocked": False
        }


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    # Test the pipeline
    pipeline = ReplyStructurePipeline()
    loader = pipeline.config

    # Get first persona
    personas = loader.get_all_personas()
    if not personas:
        print("✗ No personas found")
        exit(1)

    persona_name = list(personas.keys())[0]
    persona = personas[persona_name]

    print(f"Testing with persona: {persona_name}")

    # Test 1: Hardcoded trigger (greeting)
    result = pipeline.execute("hello", persona)
    print(f"\nTest 1 (greeting): {result['response']}")

    # Test 2: Policy check
    persona_policy = loader.get_policy(persona.policy_profile)
    if persona_policy and persona_policy.blocked_topics:
        blocked_topic = persona_policy.blocked_topics[0]
        result = pipeline.execute(f"tell me about {blocked_topic}", persona)
        print(f"\nTest 2 (blocked topic '{blocked_topic}'): {result['response']}")
        print(f"  Blocked: {result['blocked']}")

    # Test 3: Normal input
    result = pipeline.execute("how do I optimize code?", persona)
    print(f"\nTest 3 (normal input): {result['response']}")
    print(f"  Emotion: {result['emotion']}")
    print(f"  Flair applied: {result['flair_applied']}")
    print(f"  Tool used: {result['tool_used']}")

    print("\n✓ Pipeline tests complete")
