#!/usr/bin/env python3
"""
train_loop.py — SCXQDDS shard stream -> gpt2_trainer.exe (D3D11 GPU Adam)
==========================================================================
Iterates shards.manifest.json one shard at a time, writes each shard as a
mini tokens.bin, calls gpt2_trainer.exe, and chains checkpoints across shards
and epochs.

Checkpoint naming:
    {out_dir}/ckpt_e{epoch:02d}_s{shard:05d}.safetensors  -- per-shard
    {out_dir}/ckpt_epoch_{epoch:02d}.safetensors           -- end-of-epoch

Usage:
    python train_loop.py
    python train_loop.py --epochs 3 --batch 4 --lr 3e-5
    python train_loop.py --manifest path/to/shards.manifest.json \\
                         --base-model path/to/init.safetensors \\
                         --out-dir path/to/checkpoints
"""

import argparse
import os
import struct
import subprocess
import sys
import time

# ShardReader lives in split_tokens.py (same directory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from split_tokens import ShardReader  # noqa: E402

# ── defaults ──────────────────────────────────────────────────────────────────
# Updated for unified K'UHUL system (micronaut/training location)

ROOT_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # .kuhul-v1/
DEFAULT_MANIFEST  = os.path.join(ROOT_DIR, 'models', 'smgm-16', 'scxq2_dds_folds', 'shards', 'shards.manifest.json')
DEFAULT_OUT_DIR   = os.path.join(ROOT_DIR, 'khl', 'checkpoint')
DEFAULT_TRAINER   = os.path.join(ROOT_DIR, 'releases', 'v0.1.1-igpu-trainer-xjsl', 'build', 'Release', 'gpt2_trainer.exe')
DEFAULT_TMP_DIR   = os.path.join(_SCRIPT_DIR, 'tmp')

# Max sequences per mini-bin: keeps each file under ~44 MB (44000 × 128 × 4 bytes).
# Intel HD 4600 shared VRAM cap means the trainer can't load much more than 50 MB at once.
DEFAULT_MAX_CHUNK_SEQS = 44_000


# ── helpers ───────────────────────────────────────────────────────────────────

def write_mini_tokens(tokens_np, tmp_dir: str, tag: str) -> str:
    """Write numpy token array as a tokens.bin the trainer can read."""
    os.makedirs(tmp_dir, exist_ok=True)
    n_seq, seq_len = tokens_np.shape
    path = os.path.join(tmp_dir, f'mini_{tag}.bin')
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', n_seq, seq_len))
        f.write(tokens_np.astype('<i4').tobytes())
    return path


def _fix_shapes(broken_path: str, ref_path: str) -> None:
    """Patch empty shape arrays in a gpt2_trainer.exe safetensors output."""
    import json
    import struct as _struct
    try:
        with open(ref_path, 'rb') as f:
            rlen = _struct.unpack('<Q', f.read(8))[0]
            ref_shapes = {k: v['shape'] for k, v in json.loads(f.read(rlen)).items()
                          if k != '__metadata__'}
        with open(broken_path, 'rb') as f:
            blen = _struct.unpack('<Q', f.read(8))[0]
            broken = json.loads(f.read(blen))
            blob   = f.read()
        for k, v in broken.items():
            if k == '__metadata__': continue
            v['shape'] = ref_shapes.get(k) or [
                (v['data_offsets'][1] - v['data_offsets'][0]) // 4
            ]
        hdr = json.dumps(broken, separators=(',', ':')).encode()
        hdr += b' ' * ((8 - len(hdr) % 8) % 8)
        with open(broken_path, 'wb') as f:
            f.write(_struct.pack('<Q', len(hdr)) + hdr + blob)
    except Exception as e:
        print(f"  WARNING: shape-fix failed: {e}", flush=True)


def parse_loss(line: str) -> float | None:
    """Try to pull a loss float from a trainer log line."""
    for token in line.split():
        token = token.rstrip(',;')
        try:
            v = float(token)
            if 0.0 < v < 30.0:   # sane loss range
                return v
        except ValueError:
            pass
    return None


def run_trainer(
    trainer_exe: str,
    data_path:   str,
    model_in:    str | None,
    model_out:   str,
    steps:       int,
    batch:       int,
    lr:          float,
    block:       int,
    tag:         str,
    shard_path:  str | None = None,  # NEW: DDS shard path for streaming
) -> float | None:
    """
    Invoke gpt2_trainer.exe for one shard.
    Returns the last loss value seen, or None if trainer printed none.
    
    If shard_path is provided, uses DDS streaming mode (zero-copy GPU upload).
    Otherwise falls back to CPU upload from data_path.
    """
    cmd = [
        trainer_exe,
        '--out',   model_out,
        '--steps', str(steps),
        '--batch', str(batch),
        '--lr',    str(lr),
        '--block', str(block),
    ]
    
    # Always pass --data; add --shard for DDS streaming hint if available
    cmd += ['--data', data_path]
    if shard_path and os.path.exists(shard_path):
        cmd += ['--shard', shard_path]
    
    if model_in and os.path.exists(model_in):
        cmd += ['--model', model_in]

    # cwd must be the build dir so "../../../../shaders/" resolves to native/shaders/
    trainer_dir = os.path.dirname(os.path.abspath(trainer_exe))

    print(f"  [{tag}] running {len(cmd)//2} args: {' '.join(cmd)}", flush=True)

    last_loss = None
    t0 = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=trainer_dir,
    )
    for line in proc.stdout:
        line = line.rstrip('\n')
        print(f"    | {line}", flush=True)
        v = parse_loss(line)
        if v is not None:
            last_loss = v

    proc.wait()
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  [{tag}] ERROR: trainer exited {proc.returncode} after {elapsed:.1f}s",
              flush=True)
    else:
        loss_str = f'{last_loss:.4f}' if last_loss else 'n/a'
        print(f"  [{tag}] done in {elapsed:.1f}s  last_loss={loss_str}", flush=True)

    return last_loss


# ── main loop ─────────────────────────────────────────────────────────────────

def train(args):
    # ── preflight ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.trainer_exe):
        print(f"ERROR: trainer not found: {args.trainer_exe}")
        print()
        print("Build it first:")
        print(f"  cd {_SCRIPT_DIR}")
        print("  cmake -B build -S . && cmake --build build --config Release")
        print("  # or open the VS solution and build gpt2_trainer")
        raise SystemExit(1)

    if not os.path.exists(args.manifest):
        print(f"ERROR: manifest not found: {args.manifest}")
        print("Run split_tokens.py first.")
        raise SystemExit(1)

    reader = ShardReader(args.manifest)
    n_shards = reader.n_shards
    seq_len  = reader.seq_len
    n_total  = reader.n_seq_total

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"train_loop  shards={n_shards}  seq_len={seq_len}  "
          f"total_seqs={n_total:,}  epochs={args.epochs}")
    print(f"  trainer : {args.trainer_exe}")
    print(f"  manifest: {args.manifest}")
    print(f"  out_dir : {args.out_dir}")
    print()

    # steps per shard: auto-compute from dataset size, or use explicit override
    if args.steps_per_shard > 0:
        steps_per_shard = args.steps_per_shard
    else:
        steps_per_shard = max(args.min_steps, n_total // (n_shards * args.batch))

    current_model = args.base_model   # None -> trainer initialises randomly

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        print(f"=== EPOCH {epoch}/{args.epochs} ===", flush=True)

        for shard_idx, (tokens, meta) in enumerate(reader):
            n_seq = meta['n_seq']

            # Sub-chunk large shards to stay under --max-chunk-seqs (≈44 MB per file).
            chunk_size = min(args.max_chunk_seqs, n_seq)
            n_chunks   = max(1, (n_seq + chunk_size - 1) // chunk_size)

            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end   = min(chunk_start + chunk_size, n_seq)
                chunk_tokens = tokens[chunk_start:chunk_end]

                tag = f"e{epoch:02d}_s{shard_idx:05d}_c{chunk_idx:02d}"

                mini_path = write_mini_tokens(chunk_tokens, args.tmp_dir, tag)
                ckpt_path = os.path.join(args.out_dir, f'ckpt_{tag}.safetensors')

                print(f"  shard {shard_idx+1}/{n_shards}  chunk {chunk_idx+1}/{n_chunks}"
                      f"  seqs={len(chunk_tokens):,}  steps={steps_per_shard}"
                      f"  mini={mini_path}", flush=True)

                loss = run_trainer(
                    trainer_exe = args.trainer_exe,
                    data_path   = mini_path,
                    model_in    = current_model,
                    model_out   = ckpt_path,
                    steps       = steps_per_shard,
                    batch       = args.batch,
                    lr          = args.lr,
                    block       = seq_len,
                    tag         = tag,
                    shard_path  = meta.get('path') if chunk_idx == 0 else None,
                )

                if loss is not None:
                    epoch_losses.append(loss)

                if os.path.exists(ckpt_path):
                    # Repair empty-shape safetensors header written by gpt2_trainer.exe.
                    if args.base_model and os.path.exists(args.base_model):
                        _fix_shapes(ckpt_path, args.base_model)
                    current_model = ckpt_path
                else:
                    print(f"  WARNING: {ckpt_path} not produced — continuing with previous model",
                          flush=True)

                if os.path.exists(mini_path):
                    os.remove(mini_path)

        # end-of-epoch checkpoint (copy/symlink latest shard checkpoint)
        epoch_ckpt = os.path.join(args.out_dir, f'ckpt_epoch_{epoch:02d}.safetensors')
        if current_model and os.path.exists(current_model):
            import shutil
            shutil.copy2(current_model, epoch_ckpt)
            print(f"\n  epoch {epoch} checkpoint -> {epoch_ckpt}", flush=True)

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
        loss_str = f'{avg_loss:.4f}' if avg_loss else 'n/a'
        print(f"=== EPOCH {epoch} DONE  avg_loss={loss_str} ===\n", flush=True)

    print("Training complete.")
    if current_model:
        print(f"Final model: {current_model}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Stream SCXQDDS shards -> gpt2_trainer.exe (D3D11 GPU Adam)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_loop.py
  python train_loop.py --epochs 3 --batch 4 --lr 1e-4
  python train_loop.py --manifest ~/shards/shards.manifest.json \\
                       --base-model ~/checkpoints/init.safetensors \\
                       --out-dir ~/checkpoints
        """)

    p.add_argument('--manifest',    default=DEFAULT_MANIFEST,
                   help=f'shards.manifest.json  (default: {DEFAULT_MANIFEST})')
    p.add_argument('--trainer-exe', default=DEFAULT_TRAINER,
                   help=f'path to gpt2_trainer.exe  (default: {DEFAULT_TRAINER})')
    p.add_argument('--base-model',
                   default=None,
                   help='initial .safetensors checkpoint (optional — omit to start from random init)')
    p.add_argument('--out-dir',     default=DEFAULT_OUT_DIR,
                   help=f'checkpoint output dir  (default: {DEFAULT_OUT_DIR})')
    p.add_argument('--tmp-dir',     default=DEFAULT_TMP_DIR,
                   help=f'temp dir for mini tokens.bin  (default: {DEFAULT_TMP_DIR})')
    p.add_argument('--epochs',      type=int,   default=1,
                   help='number of full passes over the shard set (default: 1)')
    p.add_argument('--batch',       type=int,   default=4,
                   help='batch size passed to trainer  (default: 4)')
    p.add_argument('--lr',          type=float, default=3e-5,
                   help='learning rate  (default: 3e-5)')
    p.add_argument('--min-steps',   type=int,   default=50,
                   help='minimum training steps per shard  (default: 50)')
    p.add_argument('--steps-per-shard', type=int, default=0,
                   help='override auto-computed steps per shard (0 = auto)')
    p.add_argument('--max-chunk-seqs', type=int, default=DEFAULT_MAX_CHUNK_SEQS,
                   help=f'max sequences per mini-bin chunk (default: {DEFAULT_MAX_CHUNK_SEQS}, ~44 MB)')

    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
