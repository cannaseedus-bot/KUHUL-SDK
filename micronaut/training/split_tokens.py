#!/usr/bin/env python3
"""
split_tokens.py — tokens.bin → .scxqdds shards
================================================
Reads  : [uint32 n_seq][uint32 seq_len][int32 tokens...] × n_seq
Writes : shard-NNNNN.scxqdds  (SQDS container, 2 chunks each)
         shards.manifest.json  (shard list + metadata)

SQDS chunk layout per shard:
  chunk 0  type=0x00  META    — shard_idx, n_seq, seq_len, total_n_seq, global_offset  (5 × uint32)
  chunk 1  type=0x01  TOKENS  — n_seq × seq_len × int32  (raw token ids, little-endian)

Usage:
    python split_tokens.py
    python split_tokens.py --input path/to/tokens.bin --outdir path/to/shards --shard-mb 50
"""

import argparse
import json
import math
import os
import struct
import zlib

import numpy as np

# ── SQDS wire-format constants ────────────────────────────────────────────────

MAGIC   = b'SQDS'
VERSION = 1
FLAGS   = 0

CHUNK_META   = 0x00   # 5 × uint32 metadata
CHUNK_TOKENS = 0x01   # int32 token ids


# ── varint helpers (matches scxqdds.cpp decode_varint_u32) ────────────────────

def _varint(n: int) -> bytes:
    out = bytearray()
    while n > 0x7F:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n & 0x7F)
    return bytes(out)


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


# ── shard writer ──────────────────────────────────────────────────────────────

def write_shard(
    path: str,
    shard_idx: int,
    token_block: np.ndarray,   # shape (n_seq, seq_len), dtype int32
    total_n_seq: int,
    global_offset: int,
) -> int:
    """Write one .scxqdds shard.  Returns file size in bytes."""
    n_seq, seq_len = token_block.shape

    # ── payloads ──────────────────────────────────────────────────────────────
    meta_payload  = struct.pack('<5I',
                                shard_idx, n_seq, seq_len,
                                total_n_seq, global_offset)
    token_payload = token_block.astype('<i4').tobytes()

    payloads = [meta_payload, token_payload]
    types    = [CHUNK_META, CHUNK_TOKENS]

    # Reconstructed-space offsets: chunks laid out contiguously at offset 0
    recon_offsets = [0]
    for p in payloads[:-1]:
        recon_offsets.append(recon_offsets[-1] + len(p))

    # ── header ────────────────────────────────────────────────────────────────
    header = MAGIC + bytes([VERSION, FLAGS]) + _varint(len(payloads))

    # ── chunk header table ────────────────────────────────────────────────────
    chunk_headers = bytearray()
    for i, (p, typ, off) in enumerate(zip(payloads, types, recon_offsets)):
        chunk_headers += (
            _varint(i)          +   # id
            bytes([typ])        +   # type
            _varint(off)        +   # reconstructed offset
            _varint(len(p))     +   # payload length
            struct.pack('<I', _crc32(p))   # crc32 le
        )

    body = b''.join(payloads)

    # ── file CRC covers everything before the trailing 4 bytes ───────────────
    pre_crc = header + bytes(chunk_headers) + body
    file_crc = _crc32(pre_crc)

    with open(path, 'wb') as f:
        f.write(pre_crc)
        f.write(struct.pack('<I', file_crc))

    return os.path.getsize(path)


# ── main splitter ─────────────────────────────────────────────────────────────

def split(input_path: str, outdir: str, shard_mb: float) -> None:
    os.makedirs(outdir, exist_ok=True)

    # ── read entire tokens.bin into memory ───────────────────────────────────
    with open(input_path, 'rb') as f:
        n_seq_total = int(np.frombuffer(f.read(4), dtype='<u4')[0])
        seq_len     = int(np.frombuffer(f.read(4), dtype='<u4')[0])
        raw         = f.read()

    tokens_all = np.frombuffer(raw, dtype='<i4')

    expected = n_seq_total * seq_len
    if tokens_all.size != expected:
        actual_seqs = tokens_all.size // seq_len
        print(f"  WARNING: header says {n_seq_total} seqs but found "
              f"{tokens_all.size} tokens → using {actual_seqs} seqs")
        n_seq_total = actual_seqs

    tokens_all = tokens_all[:n_seq_total * seq_len].reshape(n_seq_total, seq_len)

    # ── shard sizing ─────────────────────────────────────────────────────────
    seq_bytes       = seq_len * 4          # bytes per sequence (int32 tokens)
    shard_target    = int(shard_mb * 1024 * 1024)
    seqs_per_shard  = max(1, shard_target // seq_bytes)
    n_shards        = math.ceil(n_seq_total / seqs_per_shard)

    actual_mb = seqs_per_shard * seq_bytes / (1024 * 1024)

    print(f"  input    : {input_path}")
    print(f"  sequences: {n_seq_total:,}  ×  {seq_len} tokens  ({seq_bytes} B each)")
    print(f"  target   : {shard_mb:.0f} MB/shard -> {seqs_per_shard:,} seqs/shard "
          f"({actual_mb:.1f} MB token payload)")
    print(f"  shards   : {n_shards}")
    print(f"  output   : {outdir}")
    print()

    # ── write shards ──────────────────────────────────────────────────────────
    shard_paths = []
    total_bytes = 0

    for shard_idx, start in enumerate(range(0, n_seq_total, seqs_per_shard)):
        end         = min(start + seqs_per_shard, n_seq_total)
        block       = tokens_all[start:end]               # numpy slice — no copy
        path        = os.path.join(outdir, f'shard-{shard_idx:05d}.scxqdds')
        size        = write_shard(path, shard_idx, block, n_seq_total, start)
        total_bytes += size

        tag = "  (partial)" if end < n_seq_total and (end - start) < seqs_per_shard else ""
        print(f"  shard-{shard_idx:05d}.scxqdds  "
              f"{end - start:>6,} seqs  "
              f"{size / (1024*1024):.2f} MB{tag}")
        shard_paths.append(path)

    # ── manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "version":        1,
        "n_shards":       len(shard_paths),
        "n_seq_total":    n_seq_total,
        "seq_len":        seq_len,
        "seqs_per_shard": seqs_per_shard,
        "shard_mb":       shard_mb,
        "total_mb":       round(total_bytes / (1024 * 1024), 2),
        "shards":         shard_paths,
    }
    manifest_path = os.path.join(outdir, 'shards.manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"  manifest : {manifest_path}")
    print(f"  total    : {total_bytes / (1024*1024):.1f} MB across {len(shard_paths)} shards")
    print()
    print("Done.")


# ── shard reader (for training loop) ─────────────────────────────────────────

class ShardReader:
    """
    Iterate .scxqdds shards produced by split_tokens.py.

    Usage:
        reader = ShardReader('path/to/shards.manifest.json')
        for epoch in range(num_epochs):
            for shard_idx, (tokens, meta) in enumerate(reader):
                # tokens: np.ndarray shape (n_seq, seq_len) dtype int32
                # meta:   dict with shard_idx, global_offset, n_seq, seq_len
                dispatch_to_gpu(tokens, meta)
    """

    def __init__(self, manifest_path: str):
        with open(manifest_path) as f:
            self._manifest = json.load(f)
        self._paths = self._manifest['shards']

    @property
    def n_shards(self) -> int:
        return self._manifest['n_shards']

    @property
    def seq_len(self) -> int:
        return self._manifest['seq_len']

    @property
    def n_seq_total(self) -> int:
        return self._manifest['n_seq_total']

    def __iter__(self):
        for path in self._paths:
            yield self._load(path)

    def __len__(self):
        return len(self._paths)

    def _load(self, path: str):
        with open(path, 'rb') as f:
            data = f.read()

        if data[:4] != MAGIC:
            raise ValueError(f"{path}: invalid SQDS magic")

        offset = 4
        _version = data[offset]; offset += 1
        _flags   = data[offset]; offset += 1

        chunk_count, offset = _read_varint(data, offset)

        headers = []
        for _ in range(chunk_count):
            c_id,  offset = _read_varint(data, offset)
            c_type        = data[offset]; offset += 1
            c_off, offset = _read_varint(data, offset)
            c_len, offset = _read_varint(data, offset)
            c_crc         = struct.unpack_from('<I', data, offset)[0]; offset += 4
            headers.append((c_id, c_type, c_off, c_len, c_crc))

        payload_start = offset

        meta_bytes   = None
        token_bytes  = None

        for c_id, c_type, c_off, c_len, c_crc in headers:
            payload = data[payload_start + c_off : payload_start + c_off + c_len]
            actual  = _crc32(payload)
            if actual != c_crc:
                raise ValueError(
                    f"{path} chunk {c_id}: CRC mismatch "
                    f"(expected {c_crc:#010x}, got {actual:#010x})")
            if c_type == CHUNK_META:
                meta_bytes = payload
            elif c_type == CHUNK_TOKENS:
                token_bytes = payload

        if meta_bytes is None or token_bytes is None:
            raise ValueError(f"{path}: missing META or TOKENS chunk")

        shard_idx, n_seq, seq_len, total_n_seq, global_offset = \
            struct.unpack('<5I', meta_bytes)

        tokens = np.frombuffer(token_bytes, dtype='<i4').reshape(n_seq, seq_len).copy()

        meta = {
            'shard_idx':     shard_idx,
            'n_seq':         n_seq,
            'seq_len':       seq_len,
            'total_n_seq':   total_n_seq,
            'global_offset': global_offset,
            'path':          path,
        }
        return tokens, meta


def _read_varint(data: bytes, offset: int):
    result = 0
    shift  = 0
    while offset < len(data):
        b = data[offset]; offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, offset
        shift += 7
        if shift > 28:
            raise ValueError("varint too large")
    raise ValueError("unexpected EOF in varint")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    default_input  = os.path.join(_here, '..', 'out', 'tokens.bin')
    default_outdir = os.path.join(_here, '..', 'shards')

    p = argparse.ArgumentParser(
        description='Split tokens.bin into SCXQDDS shards for iGPU streaming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_tokens.py
  python split_tokens.py --shard-mb 25
  python split_tokens.py --input tokens.bin --outdir ./shards --shard-mb 50

To generate tokens.bin first:
  python tokenize_dataset.py --data train_combined.jsonl --out tokens.bin
        """)
    p.add_argument('--input',    default=default_input,
                   help=f'Path to tokens.bin (default: {default_input})')
    p.add_argument('--outdir',   default=default_outdir,
                   help=f'Output directory (default: {default_outdir})')
    p.add_argument('--shard-mb', type=float, default=50.0,
                   help='Target shard size in MB (default: 50)')
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        print()
        print("Generate tokens.bin first:")
        print(f"  python tokenize_dataset.py "
              f"--data train_combined.jsonl --out {args.input}")
        raise SystemExit(1)

    split(args.input, args.outdir, args.shard_mb)


if __name__ == '__main__':
    main()
