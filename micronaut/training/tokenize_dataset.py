#!/usr/bin/env python3
"""
tokenize_dataset.py — smgm16 refined JSONL -> tokens.bin
=========================================================
Reads  : train.refined.jsonl (smgm16_gpu_bridge format)
           {"prompt": "...", "response": "...", "curriculum_bucket": "hard", ...}
Writes : tokens.bin
           [uint32 n_sequences][uint32 seq_len][int32 tokens...] x n_sequences

Each example is packed as:
    prompt + "\\n" + response + "<|endoftext|>"
Token stream is tiled into fixed-length blocks of seq_len (default 128).
Records are processed in curriculum order: easy_medium -> medium -> hard -> hard_long
so the training loop sees easier examples first.

Usage:
    python tokenize_dataset.py
    python tokenize_dataset.py --input train.refined.jsonl --out tokens.bin --block 128
    python tokenize_dataset.py --input train.refined.jsonl --out tokens.bin --block 128 --val val.refined.jsonl --val-out val_tokens.bin
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import tiktoken

# ── curriculum order (easy first so the model warms up before hard examples) ─
CURRICULUM_ORDER = {
    "easy_short":  0,
    "easy_medium": 1,
    "medium":      2,
    "hard":        3,
    "hard_long":   4,
}

EOT = "<|endoftext|>"


def load_records(path: str) -> list:
    records = []
    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping line {lineno}: {e}", file=sys.stderr)
    return records


def sort_by_curriculum(records: list) -> list:
    return sorted(records,
                  key=lambda r: CURRICULUM_ORDER.get(r.get('curriculum_bucket', ''), 99))


def record_to_text(rec: dict) -> str:
    prompt   = rec.get('prompt',   '').strip()
    response = rec.get('response', '').strip()
    if not prompt or not response:
        return ''
    return prompt + '\n' + response + EOT


def tokenize(path: str, out_path: str, block_size: int,
             enc: tiktoken.Encoding, curriculum: bool) -> dict:

    t0 = time.time()
    print(f"  loading  : {path}")
    records = load_records(path)
    print(f"  records  : {len(records):,}")

    if curriculum:
        records = sort_by_curriculum(records)
        buckets = {}
        for r in records:
            b = r.get('curriculum_bucket', 'unknown')
            buckets[b] = buckets.get(b, 0) + 1
        for b, n in sorted(buckets.items(), key=lambda x: CURRICULUM_ORDER.get(x[0], 99)):
            print(f"    {b:<16}: {n:>7,}")

    # ── tokenise all records into one flat token stream ───────────────────────
    print(f"  encoding with GPT-2 BPE (block_size={block_size}) ...")
    all_ids = []
    skipped = 0
    for i, rec in enumerate(records):
        text = record_to_text(rec)
        if not text:
            skipped += 1
            continue
        ids = enc.encode(text, allowed_special={EOT})
        all_ids.extend(ids)
        if (i + 1) % 10000 == 0:
            pct = (i + 1) / len(records) * 100
            print(f"    {i+1:>7,} / {len(records):,}  ({pct:.0f}%)  "
                  f"tokens so far: {len(all_ids):,}")

    print(f"  total tokens: {len(all_ids):,}  (skipped {skipped} empty records)")

    # ── pack into fixed-length blocks ─────────────────────────────────────────
    token_arr   = np.array(all_ids, dtype=np.int32)
    n_complete  = len(token_arr) // block_size
    token_arr   = token_arr[:n_complete * block_size]     # discard trailing partial
    sequences   = token_arr.reshape(n_complete, block_size)

    leftover = len(all_ids) - n_complete * block_size
    print(f"  sequences   : {n_complete:,}  x  {block_size} tokens "
          f"({leftover} tokens discarded as partial tail)")

    # ── write tokens.bin ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<II', n_complete, block_size))
        f.write(sequences.astype('<i4').tobytes())

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"  wrote      : {out_path}  ({size_mb:.1f} MB)  [{elapsed:.1f}s]")

    return {
        'n_sequences': n_complete,
        'seq_len':     block_size,
        'total_tokens': len(all_ids),
        'size_mb':     round(size_mb, 2),
    }


def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(_here))  # .kuhul-v1/
    base     = os.path.join(root, 'models', 'smgm-16', 'scxq2_dds_folds')  # JSONL corpus location
    out_base = os.path.join(root, 'khl', 'checkpoint')  # Checkpoint output

    p = argparse.ArgumentParser(
        description='Tokenize smgm16 refined JSONL -> tokens.bin for GPT-2 trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
After running, split into shards with:
  python split_tokens.py --input tokens.bin --outdir shards/ --shard-mb 50
        """)
    p.add_argument('--input',    default=f'{base}/train.refined.jsonl',
                   help='Training JSONL (default: smgm16 train.refined.jsonl)')
    p.add_argument('--out',      default=f'{out_base}/tokens.bin',
                   help='Output tokens.bin path')
    p.add_argument('--val',      default=f'{base}/val.refined.jsonl',
                   help='Validation JSONL (optional, pass empty string to skip)')
    p.add_argument('--val-out',  default=f'{out_base}/val_tokens.bin',
                   help='Output validation tokens.bin path')
    p.add_argument('--block',    type=int, default=128,
                   help='Sequence length in tokens (default: 128, matches trainer --block)')
    p.add_argument('--no-curriculum', action='store_true',
                   help='Disable curriculum ordering (process records as-is)')
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        raise SystemExit(1)

    enc = tiktoken.get_encoding('gpt2')
    print(f"Tokenizer  : GPT-2 BPE  vocab={enc.n_vocab}  eot={enc.eot_token}")
    print()

    # ── training set ─────────────────────────────────────────────────────────
    print("=== TRAIN ===")
    train_stats = tokenize(
        args.input, args.out, args.block, enc,
        curriculum=not args.no_curriculum)

    # ── validation set (optional) ─────────────────────────────────────────────
    val_stats = None
    if args.val and os.path.exists(args.val):
        print()
        print("=== VAL ===")
        val_stats = tokenize(
            args.val, args.val_out, args.block, enc,
            curriculum=False)       # val: no reordering, preserve original order

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=== SUMMARY ===")
    print(f"  train: {train_stats['n_sequences']:,} seqs  "
          f"{train_stats['size_mb']} MB  -> {args.out}")
    if val_stats:
        print(f"  val  : {val_stats['n_sequences']:,} seqs  "
              f"{val_stats['size_mb']} MB  -> {args.val_out}")
    print()
    print("Next step:")
    print(f"  python split_tokens.py --input {args.out} "
          f"--outdir {os.path.dirname(args.out)}/shards --shard-mb 50")


if __name__ == '__main__':
    main()
