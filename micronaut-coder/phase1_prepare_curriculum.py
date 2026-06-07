"""
Phase 1: Prepare Coder Curriculum for Fine-Tuning

Loads coder_outputs.train.jsonl, creates curriculum bucketing (easy->hard),
outputs formatted JSONL ready for tokenization.

Curriculum strategy:
  - Easy: 1-2 conversation turns (quick examples)
  - Medium: 3-5 turns (moderate complexity)
  - Hard: 5+ turns (multi-step problem solving)

Output format: JSONL with {prompt, response, curriculum_bucket, domain, ...}
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import sys
from collections import defaultdict

# Paths
CODER_TRAIN = Path(r"E:\data\smgm16_gpu_bridge\coder_outputs.train.jsonl")
CODER_VAL = Path(r"E:\data\smgm16_gpu_bridge\coder_outputs.val.jsonl")
OUTPUT_TRAIN = Path(__file__).parent / "phase1_coder_curriculum.train.jsonl"
OUTPUT_VAL = Path(__file__).parent / "phase1_coder_curriculum.val.jsonl"

def bucketing_strategy(turn_count: int) -> str:
    """Assign curriculum bucket based on conversation turns."""
    if turn_count <= 2:
        return "easy"
    elif turn_count <= 5:
        return "medium"
    else:
        return "hard"

def process_coder_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw coder record into curriculum format.

    Input format: {chat_format, domain, messages: [{role, content}], output, ...}
    Output format: {prompt, response, curriculum_bucket, domain, turn_count, ...}
    """
    messages = record.get("messages", [])

    # Reconstruct prompt from message history (all but last assistant message)
    prompt_parts = []
    turn_count = 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["user", "system"]:
            prompt_parts.append(f"{role.upper()}: {content}")
            if role == "user":
                turn_count += 1

    prompt = "\n".join(prompt_parts) if prompt_parts else ""
    response = record.get("output", "")

    # If no response in output field, try to extract from last message
    if not response:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                response = msg.get("content", "")
                break

    return {
        "prompt": prompt,
        "response": response,
        "curriculum_bucket": bucketing_strategy(turn_count),
        "turn_count": turn_count,
        "domain": record.get("domain", "semantic_code_math"),
        "score": record.get("score", 0.5),
        "id": record.get("id", ""),
        "gpu_ready": record.get("gpu_ready", False),
    }

def process_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    """
    Process JSONL file: curriculum bucketing + formatting.

    Returns: {easy: count, medium: count, hard: count}
    """
    if not input_path.exists():
        print(f"[ERROR] {input_path} not found")
        return {}

    bucket_counts = defaultdict(int)
    processed = 0
    errors = 0

    print(f"\n[PROCESSING] {input_path.name}")
    print(f"  Reading {input_path.stat().st_size / 1024 / 1024:.1f} MB...")

    with open(input_path, encoding="utf-8", errors="replace") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, 1):
            try:
                record = json.loads(line)
                processed_rec = process_coder_record(record)
                bucket = processed_rec["curriculum_bucket"]
                bucket_counts[bucket] += 1

                f_out.write(json.dumps(processed_rec) + "\n")
                processed += 1

                if line_no % 10000 == 0:
                    print(f"  Progress: {line_no:,} records...")

            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 5:  # Log first 5 errors
                    print(f"  [WARN] Line {line_no}: JSON error: {e}")
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  [WARN] Line {line_no}: {e}")

    return dict(bucket_counts), processed, errors

def main():
    """Main execution."""
    print("=" * 70)
    print("PHASE 1: PREPARE CODER CURRICULUM")
    print("=" * 70)

    # Process training set
    train_buckets, train_count, train_errors = process_file(CODER_TRAIN, OUTPUT_TRAIN)

    # Process validation set
    val_buckets, val_count, val_errors = process_file(CODER_VAL, OUTPUT_VAL)

    # Summary
    print("\n" + "=" * 70)
    print("CURRICULUM SUMMARY")
    print("=" * 70)
    print("\nTraining Set:")
    print(f"  Total records: {train_count:,}")
    print(f"  Easy (1-2 turns): {train_buckets.get('easy', 0):,}")
    print(f"  Medium (3-5 turns): {train_buckets.get('medium', 0):,}")
    print(f"  Hard (5+ turns): {train_buckets.get('hard', 0):,}")
    print(f"  Errors: {train_errors}")
    print(f"  Output: {OUTPUT_TRAIN}")

    print("\nValidation Set:")
    print(f"  Total records: {val_count:,}")
    print(f"  Easy (1-2 turns): {val_buckets.get('easy', 0):,}")
    print(f"  Medium (3-5 turns): {val_buckets.get('medium', 0):,}")
    print(f"  Hard (5+ turns): {val_buckets.get('hard', 0):,}")
    print(f"  Errors: {val_errors}")
    print(f"  Output: {OUTPUT_VAL}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print(f"1. Tokenize training set:")
    print(f"   python micronaut/training/tokenize_dataset.py \\")
    print(f"     --input {OUTPUT_TRAIN} \\")
    print(f"     --out models/smgm-16/scxq2_dds_folds/coder_tokens.bin")
    print(f"\n2. Tokenize validation set:")
    print(f"   python micronaut/training/tokenize_dataset.py \\")
    print(f"     --input {OUTPUT_VAL} \\")
    print(f"     --out models/smgm-16/scxq2_dds_folds/coder_tokens.val.bin")
    print(f"\n3. Fine-tune GPT-2:")
    print(f"   python micronaut/training/train_loop.py \\")
    print(f"     --manifest <shards-manifest.json> \\")
    print(f"     --epochs 3 --batch 8 --lr 1e-5")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
