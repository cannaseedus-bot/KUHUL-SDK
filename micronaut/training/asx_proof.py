"""
asx_proof.py — ASX Proof Chain for Training Auditability

Extracted from releases/Kuhul-PY/kuhul_es.py

The ASX proof envelope system creates a deterministic, tamper-evident chain
of training events. Each envelope is cryptographically hashed and linked to
the previous envelope, forming an immutable audit log.

Envelope format:
  {
    "@version": "asx://proof/envelope.v1",
    "@domain": "kuhul-es.training",
    "@event": "training.open | training.tick | training.close",
    "@subject": session_id,
    "@seq": sequence_number,
    "@prev": previous_envelope_hash,
    "@payload": {event-specific data},
    "@hash": sha256(canonical JSON of all above fields)
  }
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional


ASX_PROOF_VERSION = "asx://proof/envelope.v1"
ASX_DOMAIN_TRAINING = "kuhul-es.training"


def _canon(obj: Any) -> str:
    """Deterministic JSON: stable key ordering + stable separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_core(core: dict[str, Any]) -> str:
    """SHA-256 hash of canonical JSON representation."""
    return hashlib.sha256(_canon(core).encode("utf-8")).hexdigest()


def build_envelope(
    *,
    subject: str,
    event: str,  # training.open | training.tick | training.close
    seq: int,
    prev: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a deterministic ASX proof envelope.

    Args:
        subject: Session ID or identifier (e.g., "train_1")
        event: Event type (training.open, training.tick, training.close)
        seq: Sequence number (monotonically increasing from 1)
        prev: SHA-256 hash of previous envelope (empty string for first)
        payload: Event-specific data (dict)

    Returns:
        Complete envelope with deterministic @hash
    """
    core = {
        "@version": ASX_PROOF_VERSION,
        "@domain": ASX_DOMAIN_TRAINING,
        "@event": event,
        "@subject": subject,
        "@seq": seq,
        "@prev": prev,
        "@payload": payload,
    }
    return {**core, "@hash": _hash_core(core)}


def verify_proof_chain(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Verify an ASX proof chain for validity.

    Checks:
    - All required fields present
    - @seq strictly increasing
    - @prev hashes form a valid chain
    - Each envelope's @hash matches its canonical core

    Args:
        events: List of proof envelopes (in order)

    Returns:
        Dict with 'ok' (bool), count, head/tail hashes, or error details
    """
    if not events:
        return {"ok": False, "error": "empty_envelope_list"}

    prev = ""
    last_seq = 0

    for i, env in enumerate(events):
        # Check all required fields
        for k in ("@version", "@domain", "@event", "@subject", "@seq", "@prev", "@payload", "@hash"):
            if k not in env:
                return {"ok": False, "error": f"missing:{k}", "index": i}

        # Check sequence monotonicity
        if env["@seq"] <= last_seq:
            return {"ok": False, "error": "non_monotonic_seq", "index": i}

        # Check prev hash chain
        if env["@prev"] != prev:
            return {
                "ok": False,
                "error": "prev_mismatch",
                "index": i,
                "expected": prev,
                "got": env["@prev"],
            }

        # Verify hash
        core = dict(env)
        core.pop("@hash", None)
        if _hash_core(core) != env["@hash"]:
            return {"ok": False, "error": "hash_mismatch", "index": i}

        prev = env["@hash"]
        last_seq = env["@seq"]

    return {
        "ok": True,
        "count": len(events),
        "head": events[0]["@hash"],
        "tail": prev,
    }


class TrainingSession:
    """Manages ASX proof chain for a single training session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.ledger: dict[str, Any] = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seq": 0,
            "events": [],
        }

    def open(self, config: dict[str, Any]) -> dict[str, Any]:
        """Record training session open."""
        envelope = build_envelope(
            subject=self.session_id,
            event="training.open",
            seq=1,
            prev="",
            payload={"config": config},
        )
        self.ledger["events"].append(envelope)
        self.ledger["seq"] = 1
        return envelope

    def tick(self, epoch: int, shard: int, loss: float, adapter_id: Optional[str] = None) -> dict[str, Any]:
        """Record a training tick (per-shard or per-epoch milestone)."""
        seq = self.ledger["seq"] + 1
        prev = self.ledger["events"][-1]["@hash"] if self.ledger["events"] else ""

        envelope = build_envelope(
            subject=self.session_id,
            event="training.tick",
            seq=seq,
            prev=prev,
            payload={
                "epoch": epoch,
                "shard": shard,
                "loss": loss,
                "adapter_id": adapter_id,
            },
        )
        self.ledger["events"].append(envelope)
        self.ledger["seq"] = seq
        return envelope

    def close(self, final_loss: float, checkpoint_path: Optional[str] = None) -> dict[str, Any]:
        """Record training session close."""
        seq = self.ledger["seq"] + 1
        prev = self.ledger["events"][-1]["@hash"] if self.ledger["events"] else ""

        envelope = build_envelope(
            subject=self.session_id,
            event="training.close",
            seq=seq,
            prev=prev,
            payload={
                "final_loss": final_loss,
                "checkpoint_path": checkpoint_path,
            },
        )
        self.ledger["events"].append(envelope)
        self.ledger["seq"] = seq
        return envelope

    def verify(self) -> dict[str, Any]:
        """Verify entire chain for this session."""
        return verify_proof_chain(self.ledger["events"])


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("[TEST] ASX Proof Chain")
    print("=" * 60)

    # Create a session
    session = TrainingSession("train_demo_001")

    # Open
    env_open = session.open({"epochs": 2, "lr": 1e-5})
    print(f"\n[OPEN] seq={env_open['@seq']}, hash={env_open['@hash'][:16]}...")

    # Ticks
    for epoch in range(2):
        for shard in range(3):
            loss = 5.0 - (epoch * 3 + shard) * 0.1
            env_tick = session.tick(epoch=epoch, shard=shard, loss=loss)
            print(f"[TICK] epoch={epoch} shard={shard} loss={loss:.2f}")

    # Close
    env_close = session.close(final_loss=3.7, checkpoint_path="/khl/checkpoint/demo_001.pt")
    print(f"\n[CLOSE] seq={env_close['@seq']}, hash={env_close['@hash'][:16]}...")

    # Verify
    result = session.verify()
    print(f"\n[VERIFY] {result}")
    print("=" * 60)
    print("[OK] ASX proof chain test complete")
