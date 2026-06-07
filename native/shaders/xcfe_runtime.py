"""
XCFE Runtime — K'UHUL Shader Dispatch Engine
=============================================
Runtime glue for glyph_compute.hlsl and attention_compute.hlsl

Features:
- Compile K'UHUL → SCXQ2 INT4 lanes
- Load into gramBuffer
- Dispatch shaders (GlyphExec, AttnPass1, AttnPass2)
- Interpret outBuffer for XCFE routing
- Loop phases (GRAM → TENSOR → GRAM)

Usage:
    from xcfe_runtime import XCFERuntime

    runtime = XCFERuntime()
    runtime.load_shader("glyph_compute.cso")
    runtime.dispatch_gram(lanes, tokens, instructions)
    runtime.dispatch_attention(Q, K, V)
    results = runtime.get_output()
"""

import logging
import math
import struct
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

try:
    import wgpu
    _WGPU_AVAILABLE = True
except ImportError:
    wgpu = None
    _WGPU_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# WGSL SHADERS  (iGPU-safe; dispatched via wgpu OpenGL/D3D12 backend)
# ============================================================================

_WGSL_GLYPH = """
struct DispatchParams {
    laneCount     : u32,
    tokensPerLane : u32,
    mode          : u32,
    param1        : u32,
}

@group(0) @binding(0) var<uniform>            params      : DispatchParams;
@group(0) @binding(1) var<storage, read>      gramBuffer  : array<u32>;
@group(0) @binding(2) var<storage, read_write> stateBuffer : array<f32>;
@group(0) @binding(3) var<storage, read_write> outBuffer   : array<u32>;

fn decodeToken(word: u32, idx: u32) -> u32 {
    return (word >> (idx * 4u)) & 0xFu;
}

@compute @workgroup_size(64)
fn CS_GlyphExec(@builtin(global_invocation_id) id: vec3<u32>) {
    let lane = id.x;
    if (lane >= params.laneCount) { return; }

    let base = lane * ((params.tokensPerLane + 7u) / 8u);
    var r0  = stateBuffer[lane * 4u + 0u];
    var r1  = stateBuffer[lane * 4u + 1u];
    var acc = stateBuffer[lane * 4u + 2u];

    for (var i = 0u; i < params.tokensPerLane; i = i + 1u) {
        let word  = gramBuffer[base + (i / 8u)];
        let token = decodeToken(word, i % 8u);

        switch token {
            case 0x0u: {}
            case 0x1u: { r0 = stateBuffer[lane * 4u + 0u]; }
            case 0x2u: {
                stateBuffer[lane * 4u + 0u] = r0;
                stateBuffer[lane * 4u + 1u] = r1;
                stateBuffer[lane * 4u + 2u] = acc;
            }
            case 0x3u: { r0 = r0 + r1; }
            case 0x4u: { r0 = r0 * r1; }
            case 0x5u: { acc = r0 * r1; }
            case 0x6u: {
                let mag = sqrt(r0 * r0 + r1 * r1 + 0.000001f);
                r0 = r0 / mag;
                r1 = r1 / mag;
            }
            case 0x8u: { r0 = exp(r0); }
            case 0x9u: { acc = acc + r0; }
            case 0xAu: { r0 = max(r0, r1); }
            case 0xBu: { r0 = min(r0, r1); }
            case 0xCu: { r1 = r0; }
            case 0xFu: { outBuffer[lane] = lane | 0x80000000u; }
            default: {}
        }
    }

    stateBuffer[lane * 4u + 0u] = r0;
    stateBuffer[lane * 4u + 1u] = r1;
    stateBuffer[lane * 4u + 2u] = acc;
}
"""

_WGSL_ATTN_PASS1 = """
struct AttnParams {
    T      : u32,
    D      : u32,
    H      : u32,
    scale  : f32,
    causal : u32,
    _pad0  : u32,
    _pad1  : u32,
    _pad2  : u32,
}

@group(0) @binding(0) var<uniform>             params : AttnParams;
@group(0) @binding(1) var<storage, read>       Q      : array<f32>;
@group(0) @binding(2) var<storage, read>       K      : array<f32>;
@group(0) @binding(3) var<storage, read_write> scores : array<f32>;

@compute @workgroup_size(64)
fn CS_AttnPass1(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.T) { return; }

    for (var j = 0u; j < params.T; j = j + 1u) {
        if (params.causal != 0u && j > i) {
            scores[i * params.T + j] = -1e9f;
        } else {
            var dot = 0.0f;
            for (var d = 0u; d < params.D; d = d + 1u) {
                dot = dot + Q[i * params.D + d] * K[j * params.D + d];
            }
            scores[i * params.T + j] = dot * params.scale;
        }
    }
}
"""

_WGSL_ATTN_PASS2 = """
struct AttnParams {
    T      : u32,
    D      : u32,
    H      : u32,
    scale  : f32,
    causal : u32,
    _pad0  : u32,
    _pad1  : u32,
    _pad2  : u32,
}

@group(0) @binding(0) var<uniform>             params : AttnParams;
@group(0) @binding(1) var<storage, read>       scores : array<f32>;
@group(0) @binding(2) var<storage, read>       V      : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64)
fn CS_AttnPass2(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.T) { return; }

    var max_s = -1e20f;
    for (var j = 0u; j < params.T; j = j + 1u) {
        max_s = max(max_s, scores[i * params.T + j]);
    }

    var sum_exp = 0.0f;
    for (var j = 0u; j < params.T; j = j + 1u) {
        sum_exp = sum_exp + exp(scores[i * params.T + j] - max_s);
    }

    for (var d = 0u; d < params.D; d = d + 1u) {
        var acc = 0.0f;
        for (var j = 0u; j < params.T; j = j + 1u) {
            let w = exp(scores[i * params.T + j] - max_s) / (sum_exp + 1e-9f);
            acc = acc + w * V[j * params.D + d];
        }
        output[i * params.D + d] = acc;
    }
}
"""


# ============================================================================
# CONSTANTS
# ============================================================================
class ShaderMode(IntEnum):
    GRAM   = 0
    TENSOR = 1


class Opcode(IntEnum):
    """INT4 ISA opcodes (must match glyph_compute.hlsl)"""
    NOP   = 0x0
    LOAD  = 0x1
    STORE = 0x2
    ADD   = 0x3
    MUL   = 0x4
    DOT   = 0x5
    NORM  = 0x6
    EXP   = 0x8
    SUM   = 0x9
    MAX   = 0xA
    MIN   = 0xB
    MOV   = 0xC
    ESC   = 0xF


# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class DispatchParams:
    laneCount:     int
    tokensPerLane: int
    mode:          int
    param1:        int = 0

    def pack(self) -> bytes:
        return struct.pack('4I', self.laneCount, self.tokensPerLane,
                           self.mode, self.param1)


@dataclass
class AttentionParams:
    T:            int
    D:            int
    H:            int
    scale:        float
    dropout:      float = 0.0
    cache_offset: int   = 0

    def pack(self) -> bytes:
        return struct.pack('3IfI', self.T, self.D, self.H, self.scale,
                           self.dropout, self.cache_offset)


@dataclass
class LaneProgram:
    tokens: list[int]
    packed: bytes = None

    def pack(self) -> bytes:
        packed = []
        for i in range(0, len(self.tokens), 8):
            word = 0
            for j in range(8):
                if i + j < len(self.tokens):
                    word |= (self.tokens[i + j] & 0xF) << (j * 4)
            packed.append(word)
        return struct.pack(f'{len(packed)}I', *packed)


# ============================================================================
# XCFE RUNTIME
# ============================================================================
class XCFERuntime:
    """K'UHUL Shader Runtime — wgpu GPU path + CPU fallback."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and _WGPU_AVAILABLE
        self.device  = None

        # wgpu handles
        self._wgpu_device        = None
        self._wgpu_gram_pipeline = None
        self._wgpu_attn1_pipeline = None
        self._wgpu_attn2_pipeline = None

        # CPU-side mirror buffers (always allocated for fallback + readback)
        self.gram_buffer  = None
        self.state_buffer = None
        self.out_buffer   = None
        self.debug_buffer = None

        self.q_buffer      = None
        self.k_buffer      = None
        self.v_buffer      = None
        self.kv_cache      = None
        self.scores_buffer = None
        self.logits_buffer = None
        self.attn_out      = None

        self.max_lanes         = 64
        self.max_tokens_per_lane = 256
        self.max_seq_len       = 2048
        self.max_dim           = 4096

        if self.use_gpu:
            self._init_wgpu()

        logger.info(f"XCFERuntime ready (GPU={self.use_gpu})")

    # -------------------------------------------------------------------------
    # wgpu initialisation
    # -------------------------------------------------------------------------
    def _init_wgpu(self):
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
            self._wgpu_device = adapter.request_device_sync()

            dev = self._wgpu_device
            self._wgpu_gram_pipeline  = dev.create_compute_pipeline(
                layout="auto",
                compute={"module": dev.create_shader_module(code=_WGSL_GLYPH),
                         "entry_point": "CS_GlyphExec"})
            self._wgpu_attn1_pipeline = dev.create_compute_pipeline(
                layout="auto",
                compute={"module": dev.create_shader_module(code=_WGSL_ATTN_PASS1),
                         "entry_point": "CS_AttnPass1"})
            self._wgpu_attn2_pipeline = dev.create_compute_pipeline(
                layout="auto",
                compute={"module": dev.create_shader_module(code=_WGSL_ATTN_PASS2),
                         "entry_point": "CS_AttnPass2"})

            logger.info(f"wgpu device: {adapter.summary}")
        except Exception as exc:
            logger.warning(f"wgpu init failed, falling back to CPU: {exc}")
            self.use_gpu      = False
            self._wgpu_device = None

    def _wgpu_buf(self, data: np.ndarray, usage):
        """Upload a numpy array to a wgpu buffer."""
        raw = bytes(data)
        return self._wgpu_device.create_buffer_with_data(data=raw, usage=usage)

    def _wgpu_empty(self, size: int, usage):
        """Create a zero-initialised wgpu buffer."""
        return self._wgpu_device.create_buffer(size=size, usage=usage)

    # -------------------------------------------------------------------------
    # Shader loading (legacy / CSO path — for reference)
    # -------------------------------------------------------------------------
    def load_shader(self, cso_path: str) -> bool:
        try:
            with open(cso_path, 'rb') as f:
                f.read()
            self.current_shader = cso_path
            logger.info(f"CSO loaded (reference): {cso_path}")
            return True
        except Exception as exc:
            logger.error(f"load_shader failed: {exc}")
            return False

    def compile_shader(self, hlsl_path: str, entry_point: str,
                       target: str = "cs_6_0") -> str | None:
        import subprocess
        cso_path = hlsl_path.replace('.hlsl', '.cso')
        try:
            result = subprocess.run(
                ['dxc', '-T', target, '-E', entry_point, '-O3', hlsl_path,
                 '-Fo', cso_path],
                capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"DXC failed:\n{result.stderr}")
                return None
            logger.info(f"Compiled {hlsl_path} → {cso_path}")
            return cso_path
        except FileNotFoundError:
            logger.error("DXC not found.")
            return None

    # -------------------------------------------------------------------------
    # Buffer management
    # -------------------------------------------------------------------------
    def initialize_buffers(self, lane_count: int, tokens_per_lane: int):
        self.max_lanes           = max(self.max_lanes, lane_count)
        self.max_tokens_per_lane = max(self.max_tokens_per_lane, tokens_per_lane)

        gram_size  = lane_count * ((tokens_per_lane + 7) // 8) * 4
        state_size = lane_count * 4 * 4
        out_size   = lane_count * 4
        debug_size = lane_count * 64 * 4

        self.gram_buffer  = np.zeros(gram_size  // 4, dtype=np.uint32)
        self.state_buffer = np.zeros(state_size // 4, dtype=np.float32)
        self.out_buffer   = np.zeros(out_size   // 4, dtype=np.uint32)
        self.debug_buffer = np.zeros(debug_size // 4, dtype=np.uint32)

        # stride used by load_program so it matches the shader's base calculation
        self._lane_stride_words = (tokens_per_lane + 7) // 8

        logger.info(f"Buffers: {lane_count} lanes × {tokens_per_lane} tokens")

    def initialize_attention_buffers(self, seq_len: int, dim: int, heads: int = 1):
        self.max_seq_len = max(self.max_seq_len, seq_len)
        self.max_dim     = max(self.max_dim, dim)

        self.q_buffer      = np.zeros(seq_len * dim,          dtype=np.float32)
        self.k_buffer      = np.zeros(seq_len * dim,          dtype=np.float32)
        self.v_buffer      = np.zeros(seq_len * dim,          dtype=np.float32)
        self.kv_cache      = np.zeros(self.max_seq_len * dim * 2, dtype=np.float32)
        self.scores_buffer = np.zeros(seq_len * seq_len,      dtype=np.float32)
        self.logits_buffer = np.zeros(seq_len * seq_len,      dtype=np.float32)
        self.attn_out      = np.zeros(seq_len * dim,          dtype=np.float32)

        logger.info(f"Attn buffers: T={seq_len} D={dim} H={heads}")

    # -------------------------------------------------------------------------
    # Program loading
    # -------------------------------------------------------------------------
    def load_program(self, lane: int, program: LaneProgram):
        packed       = program.pack()
        packed_array = np.frombuffer(packed, dtype=np.uint32)
        # offset must match the shader's `base = lane * ((tokensPerLane+7)/8)`
        stride = getattr(self, '_lane_stride_words',
                         (len(program.tokens) + 7) // 8)
        offset = lane * stride
        self.gram_buffer[offset:offset + len(packed_array)] = packed_array

    def set_state(self, lane: int, r0: float = 0.0, r1: float = 0.0,
                  acc: float = 0.0):
        base = lane * 4
        self.state_buffer[base + 0] = r0
        self.state_buffer[base + 1] = r1
        self.state_buffer[base + 2] = acc
        self.state_buffer[base + 3] = 0.0

    # -------------------------------------------------------------------------
    # GRAM dispatch
    # -------------------------------------------------------------------------
    def dispatch_gram(self, lane_count: int, tokens_per_lane: int,
                      programs: list[LaneProgram] | None = None) -> np.ndarray:
        words_needed = lane_count * ((tokens_per_lane + 7) // 8)
        if self.gram_buffer is None or len(self.gram_buffer) < words_needed:
            self.initialize_buffers(lane_count, tokens_per_lane)

        if programs:
            for i, prog in enumerate(programs):
                if i < lane_count:
                    self.load_program(i, prog)

        logger.info(f"GRAM dispatch: {lane_count} lanes × {tokens_per_lane} tokens "
                    f"({'GPU' if self.use_gpu else 'CPU'})")

        if self.use_gpu and self._wgpu_device:
            self._dispatch_gram_gpu(lane_count, tokens_per_lane)
        else:
            self._simulate_glyph_exec(lane_count, tokens_per_lane)

        return self.out_buffer[:lane_count].copy()

    def _dispatch_gram_gpu(self, lane_count: int, tokens_per_lane: int):
        dev      = self._wgpu_device
        pipeline = self._wgpu_gram_pipeline
        BU       = wgpu.BufferUsage

        uniform_data = struct.pack("4I", lane_count, tokens_per_lane, 0, 0)

        gram_buf    = self._wgpu_buf(self.gram_buffer,  BU.STORAGE)
        state_buf   = self._wgpu_buf(self.state_buffer,
                                     BU.STORAGE | BU.COPY_SRC)
        out_buf     = self._wgpu_empty(lane_count * 4,  BU.STORAGE | BU.COPY_SRC)
        uniform_buf = dev.create_buffer_with_data(data=uniform_data,
                                                  usage=BU.UNIFORM)

        bg = dev.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": uniform_buf, "offset": 0,
                                            "size": uniform_buf.size}},
                {"binding": 1, "resource": {"buffer": gram_buf,    "offset": 0,
                                            "size": gram_buf.size}},
                {"binding": 2, "resource": {"buffer": state_buf,   "offset": 0,
                                            "size": state_buf.size}},
                {"binding": 3, "resource": {"buffer": out_buf,     "offset": 0,
                                            "size": out_buf.size}},
            ])

        enc   = dev.create_command_encoder()
        pass_ = enc.begin_compute_pass()
        pass_.set_pipeline(pipeline)
        pass_.set_bind_group(0, bg)
        pass_.dispatch_workgroups(math.ceil(lane_count / 64))
        pass_.end()
        dev.queue.submit([enc.finish()])

        state_back = dev.queue.read_buffer(state_buf)
        out_back   = dev.queue.read_buffer(out_buf)

        self.state_buffer[:] = np.frombuffer(bytes(state_back), dtype=np.float32)
        self.out_buffer[:lane_count] = np.frombuffer(bytes(out_back), dtype=np.uint32)

    # -------------------------------------------------------------------------
    # TENSOR dispatch
    # -------------------------------------------------------------------------
    def dispatch_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                           scale: float | None = None,
                           causal: bool = False) -> np.ndarray:
        T, D = Q.shape

        if self.q_buffer is None or len(self.q_buffer) < T * D:
            self.initialize_attention_buffers(T, D)

        self.q_buffer[:T * D] = Q.flatten()
        self.k_buffer[:T * D] = K.flatten()
        self.v_buffer[:T * D] = V.flatten()

        if scale is None:
            scale = 1.0 / math.sqrt(D)

        logger.info(f"Attn dispatch: T={T} D={D} causal={causal} "
                    f"({'GPU' if self.use_gpu else 'CPU'})")

        if self.use_gpu and self._wgpu_device:
            self._dispatch_attention_gpu(T, D, scale, causal)
        else:
            self._simulate_attention(T, D, -scale if causal else scale)

        return self.attn_out[:T * D].reshape(T, D).copy()

    def _dispatch_attention_gpu(self, T: int, D: int, scale: float, causal: bool):
        dev = self._wgpu_device
        BU  = wgpu.BufferUsage

        uniform_data = struct.pack("3IfIIII",
                                   T, D, 1, scale,
                                   1 if causal else 0, 0, 0, 0)

        q_buf      = self._wgpu_buf(self.q_buffer[:T * D], BU.STORAGE)
        k_buf      = self._wgpu_buf(self.k_buffer[:T * D], BU.STORAGE)
        v_buf      = self._wgpu_buf(self.v_buffer[:T * D], BU.STORAGE)
        scores_buf = self._wgpu_empty(T * T * 4,           BU.STORAGE | BU.COPY_SRC)
        out_buf    = self._wgpu_empty(T * D * 4,           BU.STORAGE | BU.COPY_SRC)
        uni_buf    = dev.create_buffer_with_data(data=uniform_data, usage=BU.UNIFORM)

        # Pass 1: Q·K → scores
        p1 = self._wgpu_attn1_pipeline
        bg1 = dev.create_bind_group(
            layout=p1.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": uni_buf,    "offset": 0, "size": uni_buf.size}},
                {"binding": 1, "resource": {"buffer": q_buf,      "offset": 0, "size": q_buf.size}},
                {"binding": 2, "resource": {"buffer": k_buf,      "offset": 0, "size": k_buf.size}},
                {"binding": 3, "resource": {"buffer": scores_buf, "offset": 0, "size": scores_buf.size}},
            ])

        enc   = dev.create_command_encoder()
        pass_ = enc.begin_compute_pass()
        pass_.set_pipeline(p1)
        pass_.set_bind_group(0, bg1)
        pass_.dispatch_workgroups(math.ceil(T / 64))
        pass_.end()
        dev.queue.submit([enc.finish()])

        # Pass 2: softmax(scores) · V → output
        p2 = self._wgpu_attn2_pipeline
        bg2 = dev.create_bind_group(
            layout=p2.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": uni_buf,    "offset": 0, "size": uni_buf.size}},
                {"binding": 1, "resource": {"buffer": scores_buf, "offset": 0, "size": scores_buf.size}},
                {"binding": 2, "resource": {"buffer": v_buf,      "offset": 0, "size": v_buf.size}},
                {"binding": 3, "resource": {"buffer": out_buf,    "offset": 0, "size": out_buf.size}},
            ])

        enc   = dev.create_command_encoder()
        pass_ = enc.begin_compute_pass()
        pass_.set_pipeline(p2)
        pass_.set_bind_group(0, bg2)
        pass_.dispatch_workgroups(math.ceil(T / 64))
        pass_.end()
        dev.queue.submit([enc.finish()])

        out_back = dev.queue.read_buffer(out_buf)
        self.attn_out[:T * D] = np.frombuffer(bytes(out_back), dtype=np.float32)

    # -------------------------------------------------------------------------
    # CPU fallback (reference — exact Python mirror of HLSL)
    # -------------------------------------------------------------------------
    def _simulate_glyph_exec(self, lane_count: int, tokens_per_lane: int):
        for lane in range(lane_count):
            r0  = self.state_buffer[lane * 4 + 0]
            r1  = self.state_buffer[lane * 4 + 1]
            acc = self.state_buffer[lane * 4 + 2]

            words_per_lane = (tokens_per_lane + 7) // 8
            base           = lane * words_per_lane

            for i in range(tokens_per_lane):
                word  = self.gram_buffer[base + (i >> 3)]
                token = (word >> ((i & 7) * 4)) & 0xF

                if   token == Opcode.NOP:   pass
                elif token == Opcode.LOAD:  r0 = self.state_buffer[lane * 4 + 0]
                elif token == Opcode.STORE:
                    self.state_buffer[lane * 4 + 0] = r0
                    self.state_buffer[lane * 4 + 1] = r1
                    self.state_buffer[lane * 4 + 2] = acc
                elif token == Opcode.ADD:   r0 = r0 + r1
                elif token == Opcode.MUL:   r0 = r0 * r1
                elif token == Opcode.DOT:   acc = r0 * r1
                elif token == Opcode.NORM:
                    mag = np.sqrt(r0 * r0 + r1 * r1 + 1e-6)
                    r0  = r0 / mag
                    r1  = r1 / mag
                elif token == Opcode.EXP:   r0 = np.exp(r0)
                elif token == Opcode.SUM:   acc += r0
                elif token == Opcode.MAX:   r0 = max(r0, r1)
                elif token == Opcode.MIN:   r0 = min(r0, r1)
                elif token == Opcode.MOV:   r1 = r0
                elif token == Opcode.ESC:   self.out_buffer[lane] = lane | 0x80000000

            self.state_buffer[lane * 4 + 0] = r0
            self.state_buffer[lane * 4 + 1] = r1
            self.state_buffer[lane * 4 + 2] = acc

    def _simulate_attention(self, T: int, D: int, scale: float):
        Q = self.q_buffer[:T * D].reshape(T, D)
        K = self.k_buffer[:T * D].reshape(T, D)
        V = self.v_buffer[:T * D].reshape(T, D)

        causal = scale < 0
        scale  = abs(scale)

        scores = np.zeros((T, T), dtype=np.float32)
        for i in range(T):
            for j in range(T):
                dot = np.dot(Q[i], K[j])
                scores[i, j] = np.exp(dot * scale) if not (causal and j > i) else 0.0

        self.scores_buffer[:T * T] = scores.flatten()

        out = np.zeros((T, D), dtype=np.float32)
        for i in range(T):
            denom   = np.sum(scores[i]) + 1e-9
            weights = scores[i] / denom
            for j in range(T):
                out[i] += weights[j] * V[j]

        self.attn_out[:T * D] = out.flatten()

    # -------------------------------------------------------------------------
    # Output helpers
    # -------------------------------------------------------------------------
    def get_escape_signals(self) -> list[int]:
        return [int(v & 0x7FFFFFFF)
                for v in self.out_buffer if v & 0x80000000]

    def get_state(self, lane: int) -> tuple[float, float, float]:
        b = lane * 4
        return (float(self.state_buffer[b]),
                float(self.state_buffer[b + 1]),
                float(self.state_buffer[b + 2]))

    def get_debug_trace(self, lane: int) -> list[int]:
        s = lane * 64
        return self.debug_buffer[s:s + 64].tolist()


# ============================================================================
# SMOKE TEST
# ============================================================================
def smoke_test():
    print("=" * 60)
    print("XCFE Runtime — smoke test")
    print("=" * 60)

    runtime = XCFERuntime(use_gpu=True)

    program = LaneProgram(tokens=[
        Opcode.LOAD,
        Opcode.MOV,
        Opcode.ADD,
        Opcode.NORM,
        Opcode.STORE,
        Opcode.ESC,
    ])

    runtime.initialize_buffers(lane_count=4, tokens_per_lane=64)
    for lane in range(4):
        runtime.load_program(lane, program)
        runtime.set_state(lane, r0=float(lane + 1), r1=0.0, acc=0.0)

    output = runtime.dispatch_gram(lane_count=4, tokens_per_lane=64)
    print("out_buffer :", output)
    print("esc signals:", runtime.get_escape_signals())
    for lane in range(4):
        r0, r1, acc = runtime.get_state(lane)
        print(f"  lane {lane}: r0={r0:.4f} r1={r1:.4f} acc={acc:.4f}")

    print("\n--- attention ---")
    T, D = 8, 64
    Q = np.random.randn(T, D).astype(np.float32)
    K = np.random.randn(T, D).astype(np.float32)
    V = np.random.randn(T, D).astype(np.float32)
    out = runtime.dispatch_attention(Q, K, V, causal=True)
    print(f"shape={out.shape}  mean={out.mean():.4f}  std={out.std():.4f}")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    smoke_test()
