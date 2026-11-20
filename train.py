"""
GPU Neural Network Training Script for  Coin Counter (v7.2: C4E, A=16, 8x8 Bilinear Chroma).


Features:
  - Auto-parses ALL non-overlapping images in ./sampleimages/ (e.g., group-1, group-2, group-4, group-5),
    estimating exact per-coin class & rotation angle verified against .ans.
  - Builds real-on-real COMBINED training scenes (layer stacking, mosaic, dense overlap) directly
    from existing ./sampleimages/ files with zero resampling blur (chirality-safe rot90 + shifts).
  - Builds a held-out COMBINED end-to-end test suite combining all ./sampleimages/ (including
    group-3 / group-6 + white-space real coin fill with mathematically exact sums).
  - Runs Stage-2 Hard-Example Mining using the actual CoinDetector from generate_solution.py on
    separate combined training scenes, fine-tunes CoarseNet & AcceptNet, and keeps the best model.
  - Exports 100% lossless Predictive + Canonical Huffman + Base85 weights to ./trained_weights.b85
    and generates ./solution.py via generate_solution.py.

Usage:
    python train_coins.py
    python train_coins.py --save-tests
    python train_coins.py --no-inject
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gc
import io
import re
import sys
import glob
import math
import time
import heapq
import base64
import struct
import ctypes
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sampleimages")
SOLUTION_PATH = os.path.join(BASE_DIR, "solution.py")
COMBINED_TESTS_DIR = os.path.join(BASE_DIR, "combined_testcases")

CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64"
if os.path.isdir(CUDA_BIN):
    os.add_dll_directory(CUDA_BIN)

nvcuda = ctypes.WinDLL("nvcuda.dll")
nvrtc = ctypes.WinDLL("nvrtc64_130_0.dll")
cublas = ctypes.WinDLL("cublas64_13.dll")

nvcuda.cuInit.argtypes = [ctypes.c_uint]
nvcuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
nvcuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
nvcuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
nvcuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
nvcuda.cuCtxSynchronize.argtypes = []
nvcuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
nvcuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
nvcuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
nvcuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
nvcuda.cuMemsetD32_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint, ctypes.c_size_t]
nvcuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
nvcuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
nvcuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
nvcuda.cuLaunchKernel.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
]
nvrtc.nvrtcCreateProgram.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
]
nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
nvrtc.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
cublas.cublasSgemm_v2.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint64, ctypes.c_int,
    ctypes.c_uint64, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint64, ctypes.c_int,
]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
COIN_VALUES = (1, 5, 10)


def _ck(rc: int, label: str):
    if rc != 0:
        raise RuntimeError(f"CUDA error in {label}: code={rc}")


KERNELS_CU = r"""
extern "C" {

__global__ void gather_rows_f32(const float* src, const int* idx, float* dst, int B, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tot = B * stride;
    if (tid >= tot) return;
    int b = tid / stride;
    int j = tid - b * stride;
    dst[tid] = src[(size_t)idx[b] * (size_t)stride + (size_t)j];
}

__global__ void gather_rows_i32(const int* src, const int* idx, int* dst, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) dst[b] = src[idx[b]];
}

__global__ void add_bias(float* y, const float* b, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) y[idx] += b[idx % cols];
}

__global__ void sum_cols(const float* dy, float* db, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) {
        float s = 0.0f;
        for (int r = 0; r < rows; ++r) s += dy[r * cols + c];
        db[c] = s;
    }
}

__global__ void relu_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v > 0.0f ? v : 0.0f; }
}

__global__ void relu_bwd(const float* x, const float* dy, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
}

__global__ void bce_loss_fwd_bwd(const float* logits, const float* targets, const float* weights, float* dlogits, float* losses, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    float l = logits[b];
    float y = targets[b];
    float w = weights[b];
    float m = l > 0.0f ? l : 0.0f;
    losses[b] = w * (m - y * l + logf(expf(-m) + expf(l - m)));
    float sig = 1.0f / (1.0f + expf(-l));
    dlogits[b] = w * (sig - y) / (float)B;
}

__global__ void rim_loss_fwd_bwd(const float* pred, const float* tgt, const float* w_rim, float* dpred, float* losses, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    float l = pred[b * 3 + 0];
    float vx = pred[b * 3 + 1];
    float vy = pred[b * 3 + 2];
    float y = tgt[b * 3 + 0];
    float tvx = tgt[b * 3 + 1];
    float tvy = tgt[b * 3 + 2];
    float wp = w_rim[b];
    float m = l > 0.0f ? l : 0.0f;
    float bce = wp * (m - y * l + logf(expf(-m) + expf(l - m)));
    float sig = 1.0f / (1.0f + expf(-l));
    float scale = 1.0f / (float)B;
    dpred[b * 3 + 0] = wp * (sig - y) * scale;
    float reg_w = (y > 0.5f) ? 5.0f : 0.0f;
    float delta = 0.2f;
    float dx = vx - tvx, dy = vy - tvy;
    float ax = fabsf(dx), ay = fabsf(dy);
    float gx = (ax <= delta) ? dx : delta * (dx > 0.0f ? 1.0f : -1.0f);
    float gy = (ay <= delta) ? dy : delta * (dy > 0.0f ? 1.0f : -1.0f);
    float hx = (ax <= delta) ? 0.5f * dx * dx : delta * (ax - 0.5f * delta);
    float hy = (ay <= delta) ? 0.5f * dy * dy : delta * (ay - 0.5f * delta);
    dpred[b * 3 + 1] = reg_w * gx * scale;
    dpred[b * 3 + 2] = reg_w * gy * scale;
    losses[b] = bce + reg_w * (hx + hy);
}

__global__ void ce_weighted_fwd_bwd(const float* logits, const int* targets, const float* weights, float* dlogits, float* losses, int B, int C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const float* row = logits + b * C;
    float* drow = dlogits + b * C;
    float mx = row[0];
    for (int c = 1; c < C; ++c) if (row[c] > mx) mx = row[c];
    float sum_e = 0.0f;
    for (int c = 0; c < C; ++c) sum_e += expf(row[c] - mx);
    float inv_s = 1.0f / sum_e;
    int t = targets[b];
    float w = weights[b];
    losses[b] = -w * (row[t] - mx - logf(sum_e));
    float scale = w / (float)B;
    for (int c = 0; c < C; ++c) {
        float p = expf(row[c] - mx) * inv_s;
        drow[c] = (p - (c == t ? 1.0f : 0.0f)) * scale;
    }
}

__global__ void polar_denF_fwd(const float* F, const float* vis, float* out_denF, int B, int C, int R) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    int CR = C * R;
    const float* Fb = F + (size_t)b * CR * 16;
    const float* vb = vis + (size_t)b * R * 16;
    float denF = 1e-5f;
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            const float* frow = Fb + (c * R + r) * 16;
            const float* vrow = vb + r * 16;
            for (int a = 0; a < 16; ++a) {
                float vm = vrow[a];
                if (vm > 0.0f) {
                    float fv = frow[a];
                    denF += vm * fv * fv;
                }
            }
        }
    }
    out_denF[b] = denF;
}

__global__ void polar_sim_fwd(
    const float* F, const float* vis, const float* W,
    float* out_num, float* out_denW,
    int B, int C, int R
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * 48) return;
    int s = idx & 15;
    int tmp = idx >> 4;
    int cls = tmp % 3;
    int b = tmp / 3;
    int CR = C * R;
    const float* Fb = F + (size_t)b * CR * 16;
    const float* vb = vis + (size_t)b * R * 16;
    const float* Wc = W + (size_t)cls * CR * 16;
    float num = 0.0f, denW = 1e-5f;
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            int cr = c * R + r;
            const float* frow = Fb + cr * 16;
            const float* wrow = Wc + cr * 16;
            const float* vrow = vb + r * 16;
            for (int a = 0; a < 16; ++a) {
                float vm = vrow[a];
                if (vm > 0.0f) {
                    float fv = frow[a];
                    float wv = wrow[(a - s) & 15];
                    num += vm * fv * wv;
                    denW += vm * wv * wv;
                }
            }
        }
    }
    out_num[idx] = num;
    out_denW[idx] = denW;
}

__global__ void polar_sim_norm_ce_or_neg(
    const float* num, const float* denF, const float* denW,
    const int* target_cls, const float* target_s,
    float* sim_out, float* dsim, float* losses,
    int B, float tau
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    float df = denF[b];
    const float* nb = num + b * 48;
    const float* dwb = denW + b * 48;
    float* sb = sim_out + b * 48;
    float* dsb = dsim + b * 48;

    float mx = -1e9f;
    for (int i = 0; i < 48; ++i) {
        float sraw = nb[i] * rsqrtf(df * dwb[i]);
        sb[i] = sraw;
        float lg = sraw * tau;
        if (lg > mx) mx = lg;
    }
    float sum_e = 0.0f;
    for (int i = 0; i < 48; ++i) sum_e += expf(sb[i] * tau - mx);
    float inv_s = 1.0f / sum_e;
    float scale = 1.0f / (float)B;

    int cls = target_cls[b];
    if (cls >= 0) {
        float sf = target_s[b];
        float fl = floorf(sf);
        int s0 = ((int)fl) & 15;
        int s1 = (s0 + 1) & 15;
        float w1 = sf - fl;
        float w0 = 1.0f - w1;
        int t0 = cls * 16 + s0;
        int t1 = cls * 16 + s1;
        losses[b] = -(w0 * (sb[t0] * tau - mx) + w1 * (sb[t1] * tau - mx) - logf(sum_e));
        for (int i = 0; i < 48; ++i) {
            float p = expf(sb[i] * tau - mx) * inv_s;
            float yi = (i == t0 ? w0 : 0.0f) + (i == t1 ? w1 : 0.0f);
            dsb[i] = (p - yi) * tau * scale;
        }
    } else {
        float lse = (mx + logf(sum_e)) / tau;
        float diff = lse - 0.22f;
        if (diff > 0.0f) {
            losses[b] = 2.0f * diff * diff;
            float g_lse = 4.0f * diff * scale;
            for (int i = 0; i < 48; ++i) {
                float p = expf(sb[i] * tau - mx) * inv_s;
                dsb[i] = g_lse * p;
            }
        } else {
            losses[b] = 0.0f;
            for (int i = 0; i < 48; ++i) dsb[i] = 0.0f;
        }
    }
}

__global__ void polar_sim_bwd_W_partial(
    const float* F, const float* vis, const float* W,
    const float* dsim, const float* num, const float* denF, const float* denW,
    float* partial, int B, int C, int R, int tiles
) {
    int CR = C * R;
    int W_sz = 3 * CR * 16;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= tiles * W_sz) return;
    int tile = gid / W_sz;
    int idx = gid - tile * W_sz;
    int phi = idx & 15;
    int cr = (idx >> 4) % CR;
    int cls = idx / (CR * 16);
    int r = cr % R;
    float wv = W[idx];
    float acc = 0.0f;
    for (int b = tile; b < B; b += tiles) {
        float df = denF[b];
        const float* frow = F + ((size_t)b * CR + cr) * 16;
        const float* vrow = vis + ((size_t)b * R + r) * 16;
        for (int s = 0; s < 16; ++s) {
            int a = (phi + s) & 15;
            float vm = vrow[a];
            if (vm <= 0.0f) continue;
            int j = (b * 3 + cls) * 16 + s;
            float g = dsim[j];
            if (g == 0.0f) continue;
            float inv = rsqrtf(df * denW[j]);
            float nm = num[j];
            float fv = frow[a];
            acc += g * vm * inv * (fv - (nm / denW[j]) * wv);
        }
    }
    partial[gid] = acc;
}

__global__ void polar_sim_bwd_W_reduce(
    const float* partial, const float* W, float* dW,
    int C, int R, int tiles, float ang_reg
) {
    int CR = C * R;
    int W_sz = 3 * CR * 16;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W_sz) return;
    float sum = 0.0f;
    for (int t = 0; t < tiles; ++t) sum += partial[t * W_sz + idx];
    if (ang_reg > 0.0f) {
        int phi = idx & 15;
        int base = idx & ~15;
        float lap = 2.0f * W[base + phi] - W[base + ((phi - 1) & 15)] - W[base + ((phi + 1) & 15)];
        sum += ang_reg * lap;
    }
    dW[idx] = sum;
}

__global__ void adamw_step(
    float* p, const float* g, float* m, float* v,
    int n, float lr, float b1, float b2, float eps, float wd, float bc1, float bc2, float clip_val
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    if (clip_val > 0.0f) {
        gi = fmaxf(-clip_val, fminf(clip_val, gi));
    }
    float mi = b1 * m[i] + (1.0f - b1) * gi;
    float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    p[i] = p[i] * (1.0f - lr * wd) - lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
}

}
"""


class CudaEngine:
    def __init__(self):
        _ck(nvcuda.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        _ck(nvcuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        name = ctypes.create_string_buffer(128)
        nvcuda.cuDeviceGetName(name, 128, dev)
        self.device_name = name.value.decode()
        self.ctx = ctypes.c_void_p()
        _ck(nvcuda.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, dev), "cuCtxCreate_v2")
        self.blas = ctypes.c_void_p()
        _ck(cublas.cublasCreate_v2(ctypes.byref(self.blas)), "cublasCreate_v2")
        self.mod = self._compile(KERNELS_CU)
        self.fns = {}
        self.closed = False

    def _compile(self, code: str):
        prog = ctypes.c_void_p()
        _ck(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), code.encode("utf-8"), b"k.cu", 0, None, None), "nvrtcCreateProgram")
        try:
            opts = (ctypes.c_char_p * 1)(b"--use_fast_math")
            if nvrtc.nvrtcCompileProgram(prog, 1, opts) != 0:
                sz = ctypes.c_size_t()
                nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
                buf = ctypes.create_string_buffer(sz.value)
                nvrtc.nvrtcGetProgramLog(prog, buf)
                raise RuntimeError(buf.value.decode())
            ptx_sz = ctypes.c_size_t()
            nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_sz))
            ptx = ctypes.create_string_buffer(ptx_sz.value)
            nvrtc.nvrtcGetPTX(prog, ptx)
            mod = ctypes.c_void_p()
            _ck(nvcuda.cuModuleLoadData(ctypes.byref(mod), ptx), "cuModuleLoadData")
            return mod
        finally:
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    def fn(self, name: str):
        if name not in self.fns:
            f = ctypes.c_void_p()
            _ck(nvcuda.cuModuleGetFunction(ctypes.byref(f), self.mod, name.encode("ascii")), f"cuModuleGetFunction({name})")
            self.fns[name] = f
        return self.fns[name]

    def launch(self, name: str, n: int, *args, block=256):
        if n <= 0:
            return
        grid = (n + block - 1) // block
        c_args = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
        _ck(nvcuda.cuLaunchKernel(self.fn(name), grid, 1, 1, block, 1, 1, 0, None, c_args, None), f"cuLaunchKernel({name})")

    def sgemm(self, transA: bool, transB: bool, M: int, N: int, K: int, A, B, C, beta=0.0):
        opA = CUBLAS_OP_T if transA else CUBLAS_OP_N
        opB = CUBLAS_OP_T if transB else CUBLAS_OP_N
        lda = M if transA else K
        ldb = K if transB else N
        alpha_c = ctypes.c_float(1.0)
        beta_c = ctypes.c_float(beta)
        _ck(cublas.cublasSgemm_v2(
            self.blas, opB, opA,
            N, M, K,
            ctypes.byref(alpha_c), B.ptr, ldb, A.ptr, lda,
            ctypes.byref(beta_c), C.ptr, N
        ), "cublasSgemm_v2")

    def sync(self):
        _ck(nvcuda.cuCtxSynchronize(), "cuCtxSynchronize")

    def close(self):
        if not self.closed:
            self.closed = True
            if self.mod:
                nvcuda.cuModuleUnload(self.mod)
                self.mod = None
            if self.blas:
                cublas.cublasDestroy_v2(self.blas)
                self.blas = None
            if self.ctx:
                nvcuda.cuCtxDestroy_v2(self.ctx)
                self.ctx = None


class GpuTensor:
    def __init__(self, shape, dtype=np.float32, data=None):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape))
        self.nbytes = max(4, self.size * self.dtype.itemsize)
        self.ptr = ctypes.c_uint64()
        _ck(nvcuda.cuMemAlloc_v2(ctypes.byref(self.ptr), self.nbytes), "cuMemAlloc_v2")
        if data is not None:
            self.copy_from(data)
        else:
            _ck(nvcuda.cuMemsetD32_v2(self.ptr, 0, self.nbytes // 4), "cuMemsetD32_v2")

    def copy_from(self, arr, nbytes=None):
        arr = np.ascontiguousarray(arr, dtype=self.dtype)
        nb = arr.nbytes if nbytes is None else nbytes
        if nb > 0:
            _ck(nvcuda.cuMemcpyHtoD_v2(self.ptr, arr.ctypes.data, nb), "cuMemcpyHtoD_v2")

    def numpy(self, shape=None):
        shp = self.shape if shape is None else tuple(shape)
        out = np.empty(shp, dtype=self.dtype)
        if out.nbytes > 0:
            _ck(nvcuda.cuMemcpyDtoH_v2(out.ctypes.data, self.ptr, out.nbytes), "cuMemcpyDtoH_v2")
        return out

    def free(self):
        if hasattr(self, "ptr") and self.ptr.value:
            nvcuda.cuMemFree_v2(self.ptr)
            self.ptr.value = 0

    def __del__(self):
        self.free()


class Param:
    def __init__(self, arr, decay=True):
        self.p = GpuTensor(arr.shape, data=arr.astype(np.float32))
        self.g = GpuTensor(arr.shape)
        self.m = GpuTensor(arr.shape)
        self.v = GpuTensor(arr.shape)
        self.n = self.p.size
        self.decay = decay

    def free(self):
        self.p.free(); self.g.free(); self.m.free(); self.v.free()


class GpuLinear:
    def __init__(self, in_f: int, out_f: int, rng, scale=None, init_wb=None):
        if init_wb is not None:
            w_arr, b_arr = init_wb
            self.W = Param(np.asarray(w_arr, dtype=np.float32).reshape(out_f, in_f), decay=True)
            self.b = Param(np.asarray(b_arr, dtype=np.float32).reshape(out_f,), decay=False)
        else:
            s = math.sqrt(2.0 / in_f) if scale is None else scale
            self.W = Param(rng.normal(0.0, s, (out_f, in_f)).astype(np.float32), decay=True)
            self.b = Param(np.zeros((out_f,), dtype=np.float32), decay=False)
        self.in_f = in_f
        self.out_f = out_f
        self._x = None

    def params(self):
        return [self.W, self.b]

    def free(self):
        self.W.free(); self.b.free()

    def forward(self, eng: CudaEngine, x: GpuTensor, out: GpuTensor, rows: int = None):
        r = x.shape[0] if rows is None else rows
        self._x = x
        eng.sgemm(False, True, r, self.out_f, self.in_f, x, self.W.p, out, beta=0.0)
        eng.launch("add_bias", r * self.out_f, out.ptr, self.b.p.ptr, ctypes.c_int(r), ctypes.c_int(self.out_f))
        return out

    def backward(self, eng: CudaEngine, dy: GpuTensor, dx: GpuTensor = None, rows: int = None):
        r = dy.shape[0] if rows is None else rows
        eng.sgemm(True, False, self.out_f, self.in_f, r, dy, self._x, self.W.g, beta=0.0)
        eng.launch("sum_cols", self.out_f, dy.ptr, self.b.g.ptr, ctypes.c_int(r), ctypes.c_int(self.out_f))
        if dx is not None:
            eng.sgemm(False, False, r, self.in_f, self.out_f, dy, self.W.p, dx, beta=0.0)
        return dx


class AdamW:
    def __init__(self, params, lr=1e-3, wd=1e-4, b1=0.9, b2=0.999, eps=1e-8, clip_val=5.0):
        self.params = params
        self.lr = lr
        self.wd = wd
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.clip_val = clip_val
        self.t = 0

    def step(self, eng: CudaEngine, lr=None):
        if lr is not None:
            self.lr = lr
        self.t += 1
        bc1 = 1.0 - (self.b1 ** self.t)
        bc2 = 1.0 - (self.b2 ** self.t)
        for pr in self.params:
            wd_cur = self.wd if pr.decay else 0.0
            eng.launch(
                "adamw_step", pr.n,
                pr.p.ptr, pr.g.ptr, pr.m.ptr, pr.v.ptr,
                ctypes.c_int(pr.n),
                ctypes.c_float(self.lr), ctypes.c_float(self.b1), ctypes.c_float(self.b2),
                ctypes.c_float(self.eps), ctypes.c_float(wd_cur),
                ctypes.c_float(bc1), ctypes.c_float(bc2), ctypes.c_float(self.clip_val),
            )


def lr_warmup_cosine(ep: int, total_ep: int, lr_max: float, lr_min: float = 1e-4, warmup_ep: int = 3) -> float:
    if ep < warmup_ep:
        return lr_max * float(ep + 1) / float(max(1, warmup_ep))
    prog = float(ep - warmup_ep) / max(1.0, float(total_ep - warmup_ep))
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))


def f16_roundtrip(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float16).astype(np.float32)


def run_mlp_cpu_f16(weights, X: np.ndarray) -> np.ndarray:
    h = X
    for i, (W, b) in enumerate(weights):
        Wf = f16_roundtrip(W)
        bf = f16_roundtrip(b)
        h = h @ Wf.T + bf
        if i < len(weights) - 1:
            h = np.maximum(h, 0.0)
    return h


# ==============================================================================
# Exact Shared Geometry (A=16, C=4 [Yn, Cbn, Crn, EdgeMag], 8x8 Full-Res Chroma)
# ==============================================================================

RADII = np.array([3.6, 8.5, 13.5, 17.4], dtype=np.float32)
C_CNT = 4
R_CNT = 4
A_CNT = 16
CR_CNT = C_CNT * R_CNT
ANGLES = np.arange(A_CNT, dtype=np.float32) * (2.0 * math.pi / A_CNT)
POLAR_DX = (RADII[:, None] * np.cos(ANGLES)[None, :]).astype(np.float32)
POLAR_DY = (RADII[:, None] * np.sin(ANGLES)[None, :]).astype(np.float32)

OUTER_R = 20.0
OUTER_A = np.arange(12, dtype=np.float32) * (2.0 * math.pi / 12.0)
OUTER_DX = (OUTER_R * np.cos(OUTER_A)).astype(np.float32)
OUTER_DY = (OUTER_R * np.sin(OUTER_A)).astype(np.float32)
COARSE_SPEC = [(3.5, 6), (8.5, 8), (13.5, 10), (17.5, 10)]
COARSE_OFFS = []
for _r, _na in COARSE_SPEC:
    _ang = np.arange(_na, dtype=np.float32) * (2.0 * math.pi / _na)
    COARSE_OFFS.append(((_r * np.cos(_ang)).astype(np.float32), (_r * np.sin(_ang)).astype(np.float32)))


def _make_c4_mat():
    C8 = np.zeros((8, 8), dtype=np.float32)
    for u in range(8):
        cu = math.sqrt(1.0 / 8.0) if u == 0 else math.sqrt(2.0 / 8.0)
        for x in range(8):
            C8[u, x] = cu * math.cos((2 * x + 1) * u * math.pi / 16.0)
    return C8[:4, :]


C4_MAT = _make_c4_mat()


def encode_jpeg_bytes(img_u8: np.ndarray, q: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_u8, "RGB").save(buf, format="JPEG", quality=int(q), subsampling=2)
    return buf.getvalue()


def rgb_to_half_ycbcr_sim(img_u8: np.ndarray, q: int = 85, return_jpeg: bool = False):
    """Exact match to 4x4 Y half-res IDCT + full 8x8 500x500 Cb/Cr IDCT."""
    jpg = encode_jpeg_bytes(img_u8, q=q)
    with Image.open(io.BytesIO(jpg)) as pil_im:
        im_ycb = np.asarray(pil_im.convert("YCbCr"), dtype=np.float32)
    B_y = (im_ycb[:, :, 0] - 128.0).reshape(125, 8, 125, 8).transpose(0, 2, 1, 3)
    dct_4 = np.matmul(np.matmul(C4_MAT, B_y), C4_MAT.T)
    rec_8 = np.matmul(np.matmul(C4_MAT.T, dct_4), C4_MAT).transpose(0, 2, 1, 3).reshape(1000, 1000) + 128.0
    Y_half = np.clip(np.round(rec_8.reshape(500, 2, 500, 2).mean(axis=(1, 3))), 0.0, 255.0).astype(np.float32)
    Cb_half = np.clip(np.round(im_ycb[:, :, 1].reshape(500, 2, 500, 2).mean(axis=(1, 3))), 0.0, 255.0).astype(np.float32)
    Cr_half = np.clip(np.round(im_ycb[:, :, 2].reshape(500, 2, 500, 2).mean(axis=(1, 3))), 0.0, 255.0).astype(np.float32)
    if return_jpeg:
        return Y_half, Cb_half, Cr_half, jpg
    return Y_half, Cb_half, Cr_half


# ==============================================================================
# Full sampleimages/ Bootstrapper (Uses ALL Images in ./sampleimages/!)
# ==============================================================================

def find_non_overlapping_peaks(img_u8: np.ndarray, min_dist=55):
    fg = (img_u8.min(axis=2) < 238).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    dil = cv2.dilate(dist, np.ones((15, 15), np.uint8))
    ys, xs = np.where((dist >= 15.0) & (dist >= dil - 1e-6))
    vals = dist[ys, xs]
    order = np.argsort(-vals)
    kept = []
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if all((x - kx) ** 2 + (y - ky) ** 2 >= min_dist ** 2 for kx, ky in kept):
            kept.append((float(x), float(y)))
    return fg, kept


def refine_subpixel_center_rgb(img_u8: np.ndarray, cx: float, cy: float):
    ix, iy = int(round(cx)), int(round(cy))
    x0, y0 = ix - 42, iy - 42
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(1000, x0 + 85), min(1000, y0 + 85)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    patch = np.full((85, 85, 3), 255.0, dtype=np.float32)
    patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img_u8[sy0:sy1, sx0:sx1].astype(np.float32)

    m = np.clip((242.0 - patch.min(axis=2)) / 20.0, 0.0, 1.0)
    yy, xx = np.mgrid[:85, :85]
    d = np.hypot(xx - 42.0, yy - 42.0)
    m *= (d <= 40.5)
    s = float(m.sum())
    if s < 1800.0:
        return None
    dx = float((xx * m).sum() / s) - 42.0
    dy = float((yy * m).sum() / s) - 42.0
    ncx, ncy = cx + dx, cy + dy
    gx = cv2.Sobel(patch.mean(axis=2), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch.mean(axis=2), cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    for _ in range(3):
        lx, ly = 42.0 + (ncx - ix), 42.0 + (ncy - iy)
        rx = xx - lx
        ry = yy - ly
        rr = np.hypot(rx, ry)
        ring = (rr >= 35.0) & (rr <= 39.5)
        w = gmag * ring
        if w.sum() < 1e-3:
            break
        ux = rx / np.maximum(rr, 1e-4)
        uy = ry / np.maximum(rr, 1e-4)
        r_est = float((rr * w).sum() / w.sum())
        err = rr - r_est
        ncx += 0.4 * float((err * ux * w).sum() / w.sum())
        ncy += 0.4 * float((err * uy * w).sum() / w.sum())
    return (ncx, ncy)


def _sample_polar_180(patch_u8: np.ndarray) -> np.ndarray:
    """High-res polar descriptor (3 channels x 8 radii x 180 angles = 2 deg steps)."""
    rs = np.array([6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 33.5], dtype=np.float32)
    angs = np.arange(180, dtype=np.float32) * (2.0 * math.pi / 180.0)
    xs = (42.0 + rs[:, None] * np.cos(angs)[None, :]).astype(np.float32)
    ys = (42.0 + rs[:, None] * np.sin(angs)[None, :]).astype(np.float32)
    pf = patch_u8.astype(np.float32)
    out = np.empty((3, 8, 180), dtype=np.float32)
    for c in range(3):
        out[c] = cv2.remap(pf[:, :, c], xs, ys, cv2.INTER_LINEAR, borderValue=255.0)
    return out


def _estimate_coin_cls_and_deg(patch_u8: np.ndarray, tpl_fft, tpl_norm, tpl_means, force_cls=None):
    """Estimates exact (cls, deg in OpenCV CCW convention) using polar FFT cross-correlation."""
    desc = _sample_polar_180(patch_u8)
    desc_c = desc - desc.mean(axis=(1, 2), keepdims=True)
    d_norm = float(np.linalg.norm(desc_c)) + 1e-6
    d_fft = np.fft.rfft(desc_c.reshape(24, 180), axis=1)
    p_mean = desc.mean(axis=(1, 2))

    cls_candidates = [force_cls] if force_cls is not None else [0, 1, 2]
    best_cls = cls_candidates[0]
    best_deg = 0.0
    best_score = -1e9
    scores_all = {}

    for cls in cls_candidates:
        # irfft(rfft(tpl) * conj(rfft(patch)))[s] peaks at s = deg / 2.0 (OpenCV CCW angle)
        corr = np.fft.irfft(np.sum(tpl_fft[cls] * np.conj(d_fft), axis=0), n=180)
        s0 = int(np.argmax(corr))
        # Parabolic sub-bin interpolation around s0
        ym = float(corr[(s0 - 1) % 180])
        y0 = float(corr[s0])
        yp = float(corr[(s0 + 1) % 180])
        denom = (ym - 2.0 * y0 + yp)
        frac = 0.5 * (ym - yp) / denom if abs(denom) > 1e-6 else 0.0
        frac = max(-0.5, min(0.5, frac))
        deg_est = float(((s0 + frac) * 2.0) % 360.0)
        ncc = y0 / (tpl_norm[cls] * d_norm)
        color_pen = float(np.linalg.norm(p_mean - tpl_means[cls])) / 60.0
        sc = ncc - 0.35 * color_pen
        scores_all[cls] = (sc, deg_est, ncc)
        if sc > best_score:
            best_score = sc
            best_cls = cls
            best_deg = deg_est

    return best_cls, best_deg, best_score, scores_all


def _is_non_overlapping_image(img_u8: np.ndarray, peaks, min_r: float, max_r: float) -> bool:
    if len(peaks) == 0:
        return False
    min_d2 = (2.0 * min_r - 5.0) ** 2
    for i in range(len(peaks)):
        x1, y1 = peaks[i]
        for j in range(i + 1, len(peaks)):
            x2, y2 = peaks[j]
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 < min_d2:
                return False
    fg = (img_u8.min(axis=2) < 222).astype(np.uint8)
    fg_sep = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(fg_sep, connectivity=4)
    max_single_area = int(1.38 * math.pi * max_r * max_r)
    for k in range(1, n_cc):
        if stats[k, cv2.CC_STAT_AREA] > max_single_area:
            return False
    cov = np.zeros((1000, 1000), dtype=np.uint8)
    r_cov = int(math.ceil(max_r + 4.0))
    for cx, cy in peaks:
        cv2.circle(cov, (int(round(cx)), int(round(cy))), r_cov, 1, -1)
    fg_tot = int(fg.sum())
    if fg_tot == 0:
        return False
    uncovered = int((fg & (1 - cov)).sum())
    return (uncovered / fg_tot) < 0.04


def extract_all_sample_data():
    """
    Bootstraps clean sprites from group-1 & group-4, then scans ALL *.in/*.ans in ./sampleimages/:
      - Non-overlapping images (group-1, group-2, group-4, group-5, ...) -> verified per-coin GT
        with exact (cls, cx, cy, deg, r, un-premultiplied 85x85 RGB patch).
      - Overlapping images (group-3, group-6, ...) -> stored for combined end-to-end test cases.
    """
    g1_path = os.path.join(SAMPLE_DIR, "group-1.in")
    g4_path = os.path.join(SAMPLE_DIR, "group-4.in")
    ans1 = int(open(os.path.join(SAMPLE_DIR, "group-1.ans")).read().strip())
    ans4 = int(open(os.path.join(SAMPLE_DIR, "group-4.ans")).read().strip())
    g1 = np.array(Image.open(g1_path))
    g4 = np.array(Image.open(g4_path))

    _, p1 = find_non_overlapping_peaks(g1)
    assert len(p1) == ans1
    fg4, p4 = find_non_overlapping_peaks(g4)

    yy30, xx30 = np.mgrid[:85, :85]
    core30 = ((yy30 - 42) ** 2 + (xx30 - 42) ** 2) <= 30 ** 2

    raw_by_cls = [[], [], []]
    for cx, cy in p1:
        sub = refine_subpixel_center_rgb(g1, cx, cy)
        if sub is None:
            continue
        sx, sy = sub
        if sx < 44.0 or sx > 956.0 or sy < 44.0 or sy > 956.0:
            continue
        M = np.float32([[1, 0, 42.0 - sx], [0, 1, 42.0 - sy]])
        warped = cv2.warpAffine(g1, M, (85, 85), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        raw_by_cls[0].append(warped)

    c1_rgb = np.mean([c[core30].mean(axis=0) for c in raw_by_cls[0]], axis=0)

    p4_feats = []
    for cx, cy in p4:
        cyi, cxi = int(round(cy)), int(round(cx))
        y0, y1 = max(0, cyi - 30), min(1000, cyi + 31)
        x0, x1 = max(0, cxi - 30), min(1000, cxi + 31)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m = ((yy - cyi) ** 2 + (xx - cxi) ** 2 <= 30 ** 2) & (fg4[y0:y1, x0:x1] > 0)
        p4_feats.append(g4[y0:y1, x0:x1][m].mean(axis=0))
    p4_feats = np.array(p4_feats, dtype=np.float32)

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, lbls, centers = cv2.kmeans(p4_feats, 3, None, crit, 30, cv2.KMEANS_PP_CENTERS)
    lbls = lbls.ravel()
    k1 = int(np.argmin(np.linalg.norm(centers - c1_rgb, axis=1)))
    rem = [k for k in range(3) if k != k1]
    cnts = [int((lbls == k).sum()) for k in range(3)]
    sum_A = cnts[k1] * 1 + cnts[rem[0]] * 5 + cnts[rem[1]] * 10
    sum_B = cnts[k1] * 1 + cnts[rem[0]] * 10 + cnts[rem[1]] * 5
    if sum_A == ans4:
        k5, k10 = rem[0], rem[1]
    elif sum_B == ans4:
        k5, k10 = rem[1], rem[0]
    else:
        raise RuntimeError(f"Could not assign 5/10 SEK against group-4.ans={ans4}")

    cluster_to_cls = {k1: 0, k5: 1, k10: 2}
    for (cx, cy), cl in zip(p4, lbls):
        sub = refine_subpixel_center_rgb(g4, cx, cy)
        if sub is None:
            continue
        sx, sy = sub
        if sx < 44.0 or sx > 956.0 or sy < 44.0 or sy > 956.0:
            continue
        M = np.float32([[1, 0, 42.0 - sx], [0, 1, 42.0 - sy]])
        warped = cv2.warpAffine(g4, M, (85, 85), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        raw_by_cls[cluster_to_cls[int(cl)]].append(warped)

    sprites, alphas, radii, clean_by_cls = [], [], [], [[], [], []]
    edge_masks = []
    yy, xx = np.mgrid[:85, :85]
    rr = np.hypot(xx - 42.0, yy - 42.0)
    for cls in range(3):
        stack = np.stack(raw_by_cls[cls], axis=0).astype(np.float32)
        med = np.median(stack, axis=0)
        angles = np.linspace(0, 2 * math.pi, 360, endpoint=False)
        rs = []
        for th in angles:
            r_grid = np.linspace(34.0, 41.0, 141)
            xs = 42.0 + r_grid * math.cos(th)
            ys = 42.0 + r_grid * math.sin(th)
            mn = cv2.remap(
                med.min(axis=2),
                xs.astype(np.float32)[None, :],
                ys.astype(np.float32)[None, :],
                cv2.INTER_LINEAR,
            )[0]
            idx = np.where(mn >= 230.0)[0]
            if len(idx) > 0:
                rs.append(float(r_grid[idx[0]]))
        r_med = float(np.median(rs))
        alpha = np.clip((r_med + 0.5 - rr) / 1.0, 0.0, 1.0).astype(np.float32)
        edge_m = (alpha > 0.15) & (alpha < 0.99)
        edge_masks.append(edge_m)

        sprite_rgb = med.copy()
        sprite_rgb[edge_m] = np.clip(
            (med[edge_m] - (1.0 - alpha[edge_m, None]) * 255.0) / alpha[edge_m, None],
            0.0,
            255.0,
        )
        sprites.append(np.clip(sprite_rgb, 0, 255).astype(np.uint8))
        alphas.append(alpha)
        radii.append(r_med)

        for raw_c in raw_by_cls[cls]:
            cf = raw_c.astype(np.float32)
            cf[edge_m] = np.clip(
                (cf[edge_m] - (1.0 - alpha[edge_m, None]) * 255.0) / alpha[edge_m, None],
                0.0,
                255.0,
            )
            clean_by_cls[cls].append(np.clip(np.round(cf), 0, 255).astype(np.uint8))

        print(f"Bootstrapped {[1, 5, 10][cls]} SEK: n={len(stack)}, R_500={r_med / 2:.3f}", flush=True)

    # Build 180-bin polar FFT templates for exact rotation & class recognition on ALL sampleimages
    tpl_fft, tpl_norm, tpl_means = [], [], []
    for cls in range(3):
        td = _sample_polar_180(sprites[cls])
        tpl_means.append(td.mean(axis=(1, 2)))
        td_c = td - td.mean(axis=(1, 2), keepdims=True)
        tpl_norm.append(float(np.linalg.norm(td_c)) + 1e-6)
        tpl_fft.append(np.fft.rfft(td_c.reshape(24, 180), axis=1))

    # Pre-verified class map for group-4 from bootstrap above
    g4_cls_by_peak = [cluster_to_cls[int(cl)] for cl in lbls]

    raw_samples = []
    verified_non_overlap = []
    overlap_samples = []
    min_r, max_r = min(radii), max(radii)

    for in_path in sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.in"))):
        name = os.path.splitext(os.path.basename(in_path))[0]
        ans_path = os.path.splitext(in_path)[0] + ".ans"
        if not os.path.exists(ans_path):
            continue
        ans = int(open(ans_path, "r", encoding="utf-8").read().strip())
        jpg_bytes = open(in_path, "rb").read()
        img_u8 = np.array(Image.open(io.BytesIO(jpg_bytes)).convert("RGB"))
        raw_samples.append((name, jpg_bytes, img_u8, ans))

        if img_u8.shape[:2] != (1000, 1000):
            overlap_samples.append((name, img_u8, ans))
            continue

        _, peaks = find_non_overlapping_peaks(img_u8)
        known_non_ov = name in ("group-1", "group-4")
        if known_non_ov or _is_non_overlapping_image(img_u8, peaks, min_r, max_r):
            force_c0 = (len(peaks) == ans)
            coins = []
            for pi, (cx, cy) in enumerate(peaks):
                sub = refine_subpixel_center_rgb(img_u8, cx, cy)
                sx, sy = sub if sub is not None else (cx, cy)
                M = np.float32([[1, 0, 42.0 - sx], [0, 1, 42.0 - sy]])
                patch = cv2.warpAffine(img_u8, M, (85, 85), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

                if name == "group-1":
                    cls, deg = 0, 0.0
                elif name == "group-4" and len(peaks) == len(g4_cls_by_peak):
                    cls, deg = g4_cls_by_peak[pi], 0.0
                else:
                    cls, deg, _sc, _all = _estimate_coin_cls_and_deg(
                        patch, tpl_fft, tpl_norm, tpl_means, force_cls=(0 if force_c0 else None)
                    )

                r_cls = radii[cls]
                is_clipped = (
                    sx < r_cls + 1.0 or sx > 999.0 - r_cls - 1.0
                    or sy < r_cls + 1.0 or sy > 999.0 - r_cls - 1.0
                )
                if is_clipped:
                    Mr = cv2.getRotationMatrix2D((42.0, 42.0), deg, 1.0)
                    clean_patch = np.clip(
                        np.round(
                            cv2.warpAffine(
                                sprites[cls].astype(np.float32),
                                Mr,
                                (85, 85),
                                flags=cv2.INTER_LINEAR,
                                borderValue=(255, 255, 255),
                            )
                        ),
                        0,
                        255,
                    ).astype(np.uint8)
                else:
                    cf = patch.astype(np.float32)
                    em = edge_masks[cls]
                    al = alphas[cls]
                    cf[em] = np.clip(
                        (cf[em] - (1.0 - al[em, None]) * 255.0) / al[em, None],
                        0.0,
                        255.0,
                    )
                    clean_patch = np.clip(np.round(cf), 0, 255).astype(np.uint8)
                coins.append((cls, sx, sy, deg, r_cls, clean_patch))

            pred_sum = sum(COIN_VALUES[c[0]] for c in coins)
            if pred_sum == ans:
                is_1sek_only = all(c[0] == 0 for c in coins)
                verified_non_overlap.append({
                    "name": name,
                    "img_u8": img_u8,
                    "ans": ans,
                    "coins": coins,
                    "only_1sek": is_1sek_only,
                })
                if name not in ("group-1", "group-4"):
                    for cls, sx, sy, deg, r_cls, cp in coins:
                        if sx < r_cls + 1.0 or sx > 999.0 - r_cls - 1.0 or sy < r_cls + 1.0 or sy > 999.0 - r_cls - 1.0:
                            continue
                        M_inv = cv2.getRotationMatrix2D((42.0, 42.0), -deg, 1.0)
                        derot = cv2.warpAffine(
                            cp.astype(np.float32),
                            M_inv,
                            (85, 85),
                            flags=cv2.INTER_LINEAR,
                            borderValue=(255, 255, 255),
                        )
                        clean_by_cls[cls].append(np.clip(np.round(derot), 0, 255).astype(np.uint8))
                print(
                    f"  [Sample Parse] {name:10s}: VERIFIED non-overlap ({len(coins)} coins, sum={pred_sum}=={ans}, "
                    f"rot_std={np.std([c[3] for c in coins]):.1f} deg)",
                    flush=True,
                )
            else:
                overlap_samples.append((name, img_u8, ans))
                print(f"  [Sample Parse] {name:10s}: kept as overlap sample (pred_sum={pred_sum}, ans={ans})", flush=True)
        else:
            overlap_samples.append((name, img_u8, ans))
            print(f"  [Sample Parse] {name:10s}: overlapping sample (ans={ans}) -> used for combined tests", flush=True)

    if not verified_non_overlap:
        raise RuntimeError(f"No verified non-overlapping samples found in {SAMPLE_DIR}!")

    return sprites, alphas, radii, clean_by_cls, raw_samples, verified_non_overlap, overlap_samples


# ==============================================================================
# Scene Synthesizers: (A) Parametric Synthetic + (B) Combined Real Sample Images
# ==============================================================================

def _try_place_coin_on_canvas(
    canvas: np.ndarray,
    owner: np.ndarray,
    placed: list,
    cls: int,
    cx: float,
    cy: float,
    deg: float,
    r: float,
    rot_rgb: np.ndarray,
    rot_a: np.ndarray,
    allow_ov: bool,
    min_vis_ratio: float,
) -> bool:
    if cx < -24.0 or cx > 1024.0 or cy < -24.0 or cy > 1024.0:
        return False
    if not allow_ov:
        if any((cx - ox) ** 2 + (cy - oy) ** 2 < (r + orad + 1.5) ** 2 for _, ox, oy, _, orad in placed):
            return False

    ix, iy = int(round(cx)), int(round(cy))
    x0, y0 = ix - 42, iy - 42
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(1000, x0 + 85), min(1000, y0 + 85)
    if sx1 <= sx0 or sy1 <= sy0:
        return False

    yy_sub, xx_sub = np.mgrid[sy0:sy1, sx0:sx1]
    new_disk = ((xx_sub - cx) ** 2 + (yy_sub - cy) ** 2) <= (r * r)
    if int(new_disk.sum()) < 450:
        return False

    if allow_ov and len(placed) > 0:
        affected_ids = np.unique(owner[sy0:sy1, sx0:sx1][new_disk])
        for aid in affected_ids:
            if aid < 0:
                continue
            _, ax, ay, _, ar = placed[int(aid)]
            ay0, ay1 = max(0, int(ay - ar - 2)), min(1000, int(ay + ar + 3))
            ax0, ax1 = max(0, int(ax - ar - 2)), min(1000, int(ax + ar + 3))
            sub_own = (owner[ay0:ay1, ax0:ax1] == aid).copy()
            oy0, oy1 = max(ay0, sy0), min(ay1, sy1)
            ox0, ox1 = max(ax0, sx0), min(ax1, sx1)
            if oy1 > oy0 and ox1 > ox0:
                sub_own[oy0 - ay0:oy1 - ay0, ox0 - ax0:ox1 - ax0] &= ~new_disk[oy0 - sy0:oy1 - sy0, ox0 - sx0:ox1 - sx0]
            if int(sub_own.sum()) < int(min_vis_ratio * math.pi * ar * ar):
                return False
            n_cc, _, stats_cc, _ = cv2.connectedComponentsWithStats(sub_own.astype(np.uint8), connectivity=4)
            if sum(1 for k in range(1, n_cc) if stats_cc[k, cv2.CC_STAT_AREA] >= 12) != 1:
                return False

    px0, py0 = sx0 - x0, sy0 - y0
    px1, py1 = px0 + (sx1 - sx0), py0 + (sy1 - sy0)
    a_slice = rot_a[py0:py1, px0:px1, None]
    dst = canvas[sy0:sy1, sx0:sx1].astype(np.float32)
    comp = a_slice * rot_rgb[py0:py1, px0:px1] + (1.0 - a_slice) * dst
    canvas[sy0:sy1, sx0:sx1] = np.clip(np.round(comp), 0, 255).astype(np.uint8)

    owner[sy0:sy1, sx0:sx1][new_disk] = len(placed)
    placed.append((cls, cx, cy, deg, r))
    return True


def _rot90_coords(cx: float, cy: float, k: int):
    k &= 3
    if k == 0:
        return cx, cy
    if k == 1:
        return cy, 999.0 - cx
    if k == 2:
        return 999.0 - cx, 999.0 - cy
    return 999.0 - cy, cx


def synthesize_combined_scene(
    verified_scenes: list,
    alphas: list,
    rng,
    mode: str = "combo_stack",
    n_layers: int = 3,
    only_1sek: bool = False,
    return_jpeg: bool = False,
):
    """
    Combines 2-4 EXISTING verified sample images from ./sampleimages/ onto a 1000x1000 canvas!
    Uses chirality-preserving rot90 (0/90/180/270 deg) + shifts so real coin pixels are composited
    with ZERO resampling blur, real overlaps, real border clips, and 100% exact ground truth.
    """
    pool_sc = [sc for sc in verified_scenes if sc["only_1sek"]] if only_1sek else verified_scenes
    if not pool_sc:
        pool_sc = verified_scenes

    canvas = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    owner = np.full((1000, 1000), -1, dtype=np.int16)
    placed = []
    min_vis_ratio = 0.14 if mode in ("combo_dense", "combo_mosaic") else 0.16

    if mode == "combo_mosaic":
        # 4-quadrant mosaic from 4 sample images + 1-2 shifted overlapping real layers on top
        quads = [(0, 500, 0, 500), (500, 1000, 0, 500), (0, 500, 500, 1000), (500, 1000, 500, 1000)]
        for qx0, qx1, qy0, qy1 in quads:
            sc = pool_sc[int(rng.integers(0, len(pool_sc)))]
            k_rot = int(rng.integers(0, 4))
            dx_sh = float(rng.uniform(-45.0, 45.0))
            dy_sh = float(rng.uniform(-45.0, 45.0))
            order = rng.permutation(len(sc["coins"]))
            for idx in order:
                cls, cx, cy, deg, r, patch = sc["coins"][idx]
                rx, ry = _rot90_coords(cx, cy, k_rot)
                ncx, ncy = rx + dx_sh, ry + dy_sh
                if not (qx0 - 15.0 <= ncx <= qx1 + 15.0 and qy0 - 15.0 <= ncy <= qy1 + 15.0):
                    continue
                ndeg = (deg + 90.0 * k_rot) % 360.0
                rot_rgb = np.rot90(patch, k_rot).astype(np.float32)
                rot_a = np.rot90(alphas[cls], k_rot)
                # Apply subpixel fractional shift cleanly
                ix, iy = int(round(ncx)), int(round(ncy))
                fx, fy = ncx - ix, ncy - iy
                if abs(fx) > 1e-3 or abs(fy) > 1e-3:
                    Ms = np.float32([[1.0, 0.0, fx], [0.0, 1.0, fy]])
                    rot_rgb = cv2.warpAffine(rot_rgb, Ms, (85, 85), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
                    rot_a = cv2.warpAffine(rot_a, Ms, (85, 85), flags=cv2.INTER_LINEAR, borderValue=0.0)
                _try_place_coin_on_canvas(canvas, owner, placed, cls, ncx, ncy, ndeg, r, rot_rgb, rot_a, True, min_vis_ratio)
        extra_layers = max(1, n_layers - 2)
    else:
        extra_layers = n_layers

    # Layer-by-layer real image stacking (prefers mixing DIFFERENT images from sampleimages/)
    picked_indices = []
    if len(pool_sc) >= extra_layers:
        picked_indices = list(rng.choice(len(pool_sc), size=extra_layers, replace=False))
    else:
        picked_indices = list(rng.permutation(len(pool_sc)))
        while len(picked_indices) < extra_layers:
            picked_indices.append(int(rng.integers(0, len(pool_sc))))

    for li, sc_idx in enumerate(picked_indices):
        sc = pool_sc[sc_idx]
        k_rot = int(rng.integers(0, 4))
        # First layer near original alignment; upper layers shifted so real coins overlap realistically
        if li == 0 and mode != "combo_mosaic":
            dx_layer = float(rng.uniform(-35.0, 35.0))
            dy_layer = float(rng.uniform(-35.0, 35.0))
        else:
            dx_layer = float(rng.uniform(-90.0, 90.0))
            dy_layer = float(rng.uniform(-90.0, 90.0))

        order = rng.permutation(len(sc["coins"]))
        for idx in order:
            cls, cx, cy, deg, r, patch = sc["coins"][idx]
            rx, ry = _rot90_coords(cx, cy, k_rot)
            # Small per-coin jitter diversifies overlap offsets while keeping real coin layout density
            jx = float(rng.uniform(-8.0, 8.0)) if li > 0 else float(rng.uniform(-3.0, 3.0))
            jy = float(rng.uniform(-8.0, 8.0)) if li > 0 else float(rng.uniform(-3.0, 3.0))
            ncx = rx + dx_layer + jx
            ncy = ry + dy_layer + jy
            ndeg = (deg + 90.0 * k_rot) % 360.0

            rot_rgb = np.rot90(patch, k_rot).astype(np.float32)
            rot_a = np.rot90(alphas[cls], k_rot)
            ix, iy = int(round(ncx)), int(round(ncy))
            fx, fy = ncx - ix, ncy - iy
            if abs(fx) > 1e-3 or abs(fy) > 1e-3:
                Ms = np.float32([[1.0, 0.0, fx], [0.0, 1.0, fy]])
                rot_rgb = cv2.warpAffine(rot_rgb, Ms, (85, 85), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
                rot_a = cv2.warpAffine(rot_a, Ms, (85, 85), flags=cv2.INTER_LINEAR, borderValue=0.0)

            _try_place_coin_on_canvas(canvas, owner, placed, cls, ncx, ncy, ndeg, r, rot_rgb, rot_a, True, min_vis_ratio)

    q = 75 if rng.random() < 0.25 else int(rng.integers(72, 97))
    placed_500 = [(cls, (cx - 0.5) * 0.5, (cy - 0.5) * 0.5, deg, r * 0.5) for cls, cx, cy, deg, r in placed]
    owner_500 = owner.reshape(500, 2, 500, 2).max(axis=(1, 3)).copy()
    if return_jpeg:
        Y, Cb, Cr, jpg = rgb_to_half_ycbcr_sim(canvas, q=q, return_jpeg=True)
        exact_ans = sum(COIN_VALUES[c[0]] for c in placed)
        return Y, Cb, Cr, placed_500, owner_500, jpg, exact_ans
    Y, Cb, Cr = rgb_to_half_ycbcr_sim(canvas, q=q)
    return Y, Cb, Cr, placed_500, owner_500


def synthesize_scene(sprites, alphas, radii, coins_by_cls, rng, mode: str, n_target: int):
    canvas = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    placed = []
    allow_rot = mode in ("g2", "g3", "g5", "g6", "dense", "dense1", "dense6")
    allow_ov = mode in ("g3", "g6", "dense", "dense1", "dense6")
    cls_choices = (0,) if mode in ("g1", "g2", "g3", "dense1") else (0, 1, 2)
    is_super = mode in ("dense1", "dense6")
    min_vis_ratio = 0.14 if is_super else 0.16

    owner = np.full((1000, 1000), -1, dtype=np.int16)
    attempts = 0
    max_attempts = n_target * (18 if is_super else 14)
    while len(placed) < n_target and attempts < max_attempts:
        attempts += 1
        cls = int(rng.choice(cls_choices))
        r = radii[cls]
        deg = float(rng.uniform(0.0, 360.0)) if allow_rot else 0.0

        if allow_ov and len(placed) > 0 and rng.random() < (0.76 if is_super else 0.72):
            b_idx = int(rng.integers(0, len(placed)))
            _, bx, by, _, br = placed[b_idx]
            ang = float(rng.uniform(0.0, 2 * math.pi))
            dist = float(rng.uniform(21.5 if is_super else 22.0, r + br - 2.0))
            cx = bx + dist * math.cos(ang)
            cy = by + dist * math.sin(ang)
        elif rng.random() < 0.14:
            side = int(rng.integers(0, 4))
            margin = float(rng.uniform(-22.0, 22.0))
            along = float(rng.uniform(40.0, 960.0))
            cx, cy = [(along, margin), (along, 1000.0 - margin), (margin, along), (1000.0 - margin, along)][side]
        else:
            cx = float(rng.uniform(30.0, 970.0))
            cy = float(rng.uniform(30.0, 970.0))

        if len(coins_by_cls[cls]) > 0 and rng.random() < 0.55:
            sp_src = coins_by_cls[cls][int(rng.integers(0, len(coins_by_cls[cls])))]
        else:
            sp_src = sprites[cls]

        ix, iy = int(round(cx)), int(round(cy))
        M = cv2.getRotationMatrix2D((42.0, 42.0), deg, 1.0)
        M[0, 2] += (cx - ix)
        M[1, 2] += (cy - iy)
        rot_rgb = cv2.warpAffine(sp_src.astype(np.float32), M, (85, 85), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        rot_a = cv2.warpAffine(alphas[cls], M, (85, 85), flags=cv2.INTER_LINEAR, borderValue=0.0)
        _try_place_coin_on_canvas(canvas, owner, placed, cls, cx, cy, deg, r, rot_rgb, rot_a, allow_ov, min_vis_ratio)

    q = 75 if rng.random() < 0.25 else int(rng.integers(70, 97))
    Y, Cb, Cr = rgb_to_half_ycbcr_sim(canvas, q=q)
    placed_500 = [(cls, (cx - 0.5) * 0.5, (cy - 0.5) * 0.5, deg, r * 0.5) for cls, cx, cy, deg, r in placed]
    owner_500 = owner.reshape(500, 2, 500, 2).max(axis=(1, 3)).copy()
    return Y, Cb, Cr, placed_500, owner_500


# ==============================================================================
# Combined End-to-End Test Suite Builder (from ALL files in ./sampleimages/)
# ==============================================================================

def _build_overlap_plus_fill_case(base_name, base_img, base_ans, verified_scenes, alphas, k_rot, rng, max_add=35, q=82):
    """
    Takes an existing overlapping image (e.g. group-3 or group-6), applies chirality-safe rot90,
    and injects real coins from non-overlapping sample images strictly into unoccupied white regions.
    Guaranteed exact answer = base_ans + sum(added_coins).
    """
    canvas = np.rot90(base_img, k_rot).copy()
    occ = (canvas.min(axis=2) < 244).astype(np.uint8)
    occ = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    all_real_coins = [c for sc in verified_scenes for c in sc["coins"]]
    total_ans = base_ans
    added = 0
    for _ in range(max_add * 25):
        if added >= max_add:
            break
        cls, _sx, _sy, _deg, r, patch = all_real_coins[int(rng.integers(0, len(all_real_coins)))]
        cx = int(rng.integers(48, 952))
        cy = int(rng.integers(48, 952))
        rad_chk = int(math.ceil(r + 4.0))
        sub_occ = occ[cy - rad_chk:cy + rad_chk + 1, cx - rad_chk:cx + rad_chk + 1]
        if sub_occ.any():
            continue

        ck_rot = int(rng.integers(0, 4))
        rot_rgb = np.rot90(patch, ck_rot).astype(np.float32)
        rot_a = np.rot90(alphas[cls], ck_rot)
        y0, x0 = cy - 42, cx - 42
        a_sl = rot_a[:, :, None]
        dst = canvas[y0:y0 + 85, x0:x0 + 85].astype(np.float32)
        canvas[y0:y0 + 85, x0:x0 + 85] = np.clip(np.round(a_sl * rot_rgb + (1.0 - a_sl) * dst), 0, 255).astype(np.uint8)
        cv2.circle(occ, (cx, cy), rad_chk, 1, -1)
        total_ans += COIN_VALUES[cls]
        added += 1

    name = f"comb_{base_name}_r{k_rot * 90}_plus{added}"
    return name, encode_jpeg_bytes(canvas, q=q), total_ans


def build_combined_test_suite(raw_samples, verified_scenes, overlap_samples, alphas, save_dir=None):
    """
    Creates a comprehensive held-out test suite from ./sampleimages/:
      1. All original sampleimages/*.in
      2. Layered & mosaic real-on-real combined scenes (group-1/2/4/5 stacked with overlap)
      3. Real overlap images (group-3/group-6 + rot90) combined with real coins in white space
    """
    suite = []
    for name, jpg_bytes, _img, ans in raw_samples:
        suite.append((f"orig_{name}", jpg_bytes, ans))

    rng_t = np.random.default_rng(909001)
    combo_specs = [
        ("comb_stack2_1sek", "combo_stack", 2, True),
        ("comb_stack3_1sek", "combo_dense", 3, True),
        ("comb_stack2_mix", "combo_stack", 2, False),
        ("comb_stack3_mix", "combo_stack", 3, False),
        ("comb_dense4_mix", "combo_dense", 4, False),
        ("comb_mosaic4_mix", "combo_mosaic", 3, False),
    ]
    for cname, mode, nl, only1 in combo_specs:
        _, _, _, placed_500, _, jpg, exact_ans = synthesize_combined_scene(
            verified_scenes, alphas, rng_t, mode=mode, n_layers=nl, only_1sek=only1, return_jpeg=True
        )
        suite.append((f"{cname}_n{len(placed_500)}", jpg, exact_ans))

    for ov_name, ov_img, ov_ans in overlap_samples:
        for k_rot in (0, 1, 2):
            cname, jpg, exact_ans = _build_overlap_plus_fill_case(
                ov_name, ov_img, ov_ans, verified_scenes, alphas, k_rot, rng_t, max_add=30, q=int(rng_t.integers(76, 94))
            )
            suite.append((cname, jpg, exact_ans))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for cname, jpg, ans in suite:
            with open(os.path.join(save_dir, f"{cname}.in"), "wb") as f:
                f.write(jpg)
            with open(os.path.join(save_dir, f"{cname}.ans"), "w", encoding="ascii") as f:
                f.write(f"{ans}\n")
        print(f"Saved {len(suite)} combined test cases -> {save_dir}", flush=True)

    return suite


# ==============================================================================
# Feature Extraction (100% Bit-Matched to solution.py!)
# ==============================================================================

def bilinear_sample(plane: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    x0 = np.clip(np.floor(px).astype(np.int32), 0, 498)
    y0 = np.clip(np.floor(py).astype(np.int32), 0, 498)
    fx = np.clip(px - x0, 0.0, 1.0)
    fy = np.clip(py - y0, 0.0, 1.0)
    return (plane[y0, x0] * (1.0 - fx) + plane[y0, x0 + 1] * fx) * (1.0 - fy) + (
        plane[y0 + 1, x0] * (1.0 - fx) + plane[y0 + 1, x0 + 1] * fx
    ) * fy


def compute_is_fg(Y, Cb, Cr):
    return (Y < 240.0) | ((np.abs(Cb - 128.0) + np.abs(Cr - 128.0)) > 10.0)


def extract_rim_features_for_pixels(Y, Cb, Cr, ys, xs):
    Yc = Y[ys, xs]; Cbc = Cb[ys, xs]; Crc = Cr[ys, xs]
    gxY1 = (Y[ys, xs + 1] - Y[ys, xs - 1]) / 64.0
    gyY1 = (Y[ys + 1, xs] - Y[ys - 1, xs]) / 64.0
    gxY2 = (Y[ys, xs + 2] - Y[ys, xs - 2]) / 80.0
    gyY2 = (Y[ys + 2, xs] - Y[ys - 2, xs]) / 80.0
    gxCb2 = (Cb[ys, xs + 2] - Cb[ys, xs - 2]) / 32.0
    gyCb2 = (Cb[ys + 2, xs] - Cb[ys - 2, xs]) / 32.0
    gxCr2 = (Cr[ys, xs + 2] - Cr[ys, xs - 2]) / 32.0
    gyCr2 = (Cr[ys + 2, xs] - Cr[ys - 2, xs]) / 32.0
    lapY2 = (4.0 * Yc - Y[ys, xs - 2] - Y[ys, xs + 2] - Y[ys - 2, xs] - Y[ys + 2, xs]) / 128.0
    maxYnb = np.maximum(np.maximum(Y[ys, xs - 2], Y[ys, xs + 2]), np.maximum(Y[ys - 2, xs], Y[ys + 2, xs]))
    f12 = (maxYnb - 220.0) / 40.0
    f13 = np.hypot(gxY2, gyY2) + np.hypot(gxCb2, gyCb2) + np.hypot(gxCr2, gyCr2)
    return np.stack([
        (Yc - 185.0) / 60.0, (Cbc - 95.0) / 35.0, (Crc - 145.0) / 30.0,
        gxY1, gyY1, gxY2, gyY2, gxCb2, gyCb2, gxCr2, gyCr2, lapY2, f12, f13,
    ], axis=1).astype(np.float32)


def extract_coarse_features(Y, Cb, Cr, fg_map, claimed, cx: float, cy: float) -> np.ndarray:
    feats = []
    tot_vis = tot_fg = tot_oob = 0
    for dxs, dys in COARSE_OFFS:
        ix = np.floor(cx + dxs + 0.5).astype(np.int32)
        iy = np.floor(cy + dys + 0.5).astype(np.int32)
        inb = (ix >= 0) & (ix < 500) & (iy >= 0) & (iy < 500)
        tot_oob += int((~inb).sum())
        ixc = np.clip(ix, 0, 499)
        iyc = np.clip(iy, 0, 499)
        vis = inb & (claimed[iyc, ixc] == 0)
        nv = int(vis.sum())
        tot_vis += nv
        if nv > 0:
            vx, vy = ixc[vis], iyc[vis]
            yv = Y[vy, vx]; cbv = Cb[vy, vx]; crv = Cr[vy, vx]
            fgs = fg_map[vy, vx]
            tot_fg += int(fgs.sum())
            feats.extend([
                (float(yv.mean()) - 185.0) / 45.0,
                (float(cbv.mean()) - 92.0) / 25.0,
                (float(crv.mean()) - 148.0) / 25.0,
                float(yv.std()) / 35.0,
                float(fgs.mean()),
            ])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    oix = np.floor(cx + OUTER_DX + 0.5).astype(np.int32)
    oiy = np.floor(cy + OUTER_DY + 0.5).astype(np.int32)
    oinb = (oix >= 0) & (oix < 500) & (oiy >= 0) & (oiy < 500)
    if oinb.any():
        ovx, ovy = oix[oinb], oiy[oinb]
        omean_y = (float(Y[ovy, ovx].mean()) - 210.0) / 50.0
        obg = float(((~fg_map[ovy, ovx]) | (claimed[ovy, ovx] > 0)).mean())
    else:
        omean_y, obg = 0.0, 1.0

    feats.extend([tot_vis / 34.0, tot_fg / 34.0, tot_oob / 34.0, omean_y, obg])
    return np.array(feats, dtype=np.float32)


def extract_polar_raw(Y, Cb, Cr, fg_map, claimed, cx: float, cy: float):
    px = cx + POLAR_DX
    py = cy + POLAR_DY
    inb = (px >= 0.0) & (px <= 499.0) & (py >= 0.0) & (py <= 499.0)
    ix = np.clip(np.floor(px + 0.5).astype(np.int32), 0, 499)
    iy = np.clip(np.floor(py + 0.5).astype(np.int32), 0, 499)
    vis = (inb & (claimed[iy, ix] == 0)).astype(np.float32)

    pxc = np.clip(px, 0.0, 499.0)
    pyc = np.clip(py, 0.0, 499.0)
    Yp = bilinear_sample(Y, pxc, pyc)
    Cbp = bilinear_sample(Cb, pxc, pyc)
    Crp = bilinear_sample(Cr, pxc, pyc)

    fg_p = fg_map[iy, ix] & inb & (claimed[iy, ix] == 0)
    oix = np.floor(cx + OUTER_DX + 0.5).astype(np.int32)
    oiy = np.floor(cy + OUTER_DY + 0.5).astype(np.int32)
    oinb = (oix >= 0) & (oix < 500) & (oiy >= 0) & (oiy < 500)
    if oinb.any():
        obg = float(((~fg_map[oiy[oinb], oix[oinb]]) | (claimed[oiy[oinb], oix[oinb]] > 0)).mean())
    else:
        obg = 1.0

    vm_b = vis > 0.5
    n_vis = int(vm_b.sum())
    n_inb = int(inb.sum())
    if n_vis > 0:
        yv = Yp[vm_b]; cbv = Cbp[vm_b]; crv = Crp[vm_b]
        inner_v = vm_b[:2]; rim_v = vm_b[2:]
        stats12 = np.array([
            n_vis / 64.0,
            n_inb / 64.0,
            float(fg_p[vm_b].mean()),
            float(fg_p[:2][inner_v].mean()) if inner_v.any() else 0.0,
            float(fg_p[2:][rim_v].mean()) if rim_v.any() else 0.0,
            (float(yv.mean()) - 185.0) / 45.0,
            (float(cbv.mean()) - 92.0) / 25.0,
            (float(crv.mean()) - 148.0) / 25.0,
            float(yv.std()) / 40.0,
            float(cbv.std()) / 15.0,
            float(crv.std()) / 15.0,
            obg,
        ], dtype=np.float32)
    else:
        stats12 = np.array([0.0, n_inb / 64.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, obg], dtype=np.float32)

    raw_ycc = np.stack([Yp, Cbp, Crp], axis=0).astype(np.float32)
    return raw_ycc, vis, stats12


def build_polar_features_c4e(raw_ycc: np.ndarray, vis: np.ndarray) -> np.ndarray:
    N = len(raw_ycc)
    Yp = raw_ycc[:, 0]
    Cbp = raw_ycc[:, 1]
    Crp = raw_ycc[:, 2]
    Yn = (Yp - 185.0) * (1.0 / 45.0)
    Cbn = (Cbp - 92.0) * (1.0 / 25.0)
    Crn = (Crp - 148.0) * (1.0 / 25.0)

    gr = np.zeros((N, 4, 16), dtype=np.float32)
    vp0 = (vis[:, 1] > 0.0) & (vis[:, 0] > 0.0)
    gr[:, 0] = np.where(vp0, Yp[:, 1] - Yp[:, 0], 0.0)
    for r in (1, 2):
        vp = (vis[:, r + 1] > 0.0) & (vis[:, r] > 0.0)
        vm = (vis[:, r - 1] > 0.0) & (vis[:, r] > 0.0)
        both = vp & vm
        gr[:, r] = np.where(both, 0.5 * (Yp[:, r + 1] - Yp[:, r - 1]),
                   np.where(vp, Yp[:, r + 1] - Yp[:, r],
                   np.where(vm, Yp[:, r] - Yp[:, r - 1], 0.0)))
    vm3 = (vis[:, 2] > 0.0) & (vis[:, 3] > 0.0)
    gr[:, 3] = np.where(vm3, Yp[:, 3] - Yp[:, 2], 0.0)

    Yp_p = np.roll(Yp, -1, axis=2)
    Yp_m = np.roll(Yp, 1, axis=2)
    vap = (np.roll(vis, -1, axis=2) > 0.0) & (vis > 0.0)
    vam = (np.roll(vis, 1, axis=2) > 0.0) & (vis > 0.0)
    both_a = vap & vam
    ga = np.where(both_a, 0.5 * (Yp_p - Yp_m),
         np.where(vap, Yp_p - Yp,
         np.where(vam, Yp - Yp_m, 0.0))).astype(np.float32)

    edge = (np.abs(gr) + np.abs(ga)) * (1.0 / 50.0) - 0.4
    F = np.stack([Yn, Cbn, Crn, edge], axis=1) * vis[:, None, :, :]
    return F.reshape(N, CR_CNT, A_CNT).astype(np.float32)


# ==============================================================================
# Dataset Prep (Scene-Separated -> Zero Train/Val Leakage)
# ==============================================================================

def extract_rim_from_scenes(scenes, rng):
    X_list, tgt_list, w_list = [], [], []
    for Y, Cb, Cr, placed, owner in scenes:
        ys_grid, xs_grid = np.mgrid[2:498:3, 2:498:3]
        g = (
            np.abs(Y[ys_grid, xs_grid + 1] - Y[ys_grid, xs_grid - 1])
            + np.abs(Y[ys_grid + 1, xs_grid] - Y[ys_grid - 1, xs_grid])
            + np.abs(Y[ys_grid, xs_grid + 2] - Y[ys_grid, xs_grid - 2])
            + np.abs(Y[ys_grid + 2, xs_grid] - Y[ys_grid - 2, xs_grid])
            + 2.0 * (
                np.abs(Cb[ys_grid, xs_grid + 2] - Cb[ys_grid, xs_grid - 2])
                + np.abs(Cb[ys_grid + 2, xs_grid] - Cb[ys_grid - 2, xs_grid])
                + np.abs(Cr[ys_grid, xs_grid + 2] - Cr[ys_grid, xs_grid - 2])
                + np.abs(Cr[ys_grid + 2, xs_grid] - Cr[ys_grid - 2, xs_grid])
            )
        )
        m = g >= 46.0
        ys = ys_grid[m]; xs = xs_grid[m]
        if len(ys) == 0:
            continue
        X = extract_rim_features_for_pixels(Y, Cb, Cr, ys, xs)
        own_id = owner[ys, xs]
        is_rim = np.zeros(len(ys), dtype=np.float32)
        tvx = np.zeros(len(ys), dtype=np.float32)
        tvy = np.zeros(len(ys), dtype=np.float32)
        pos_mask = own_id >= 0
        pcx = np.array([p[1] for p in placed], dtype=np.float32)
        pcy = np.array([p[2] for p in placed], dtype=np.float32)
        pr = np.array([p[4] for p in placed], dtype=np.float32)
        if pos_mask.any():
            oid = own_id[pos_mask]
            dx = pcx[oid] - xs[pos_mask].astype(np.float32)
            dy = pcy[oid] - ys[pos_mask].astype(np.float32)
            dd = np.hypot(dx, dy)
            rr = pr[oid]
            ok = (dd >= rr - 1.4) & (dd <= rr + 1.0)
            tmp_rim = np.zeros(pos_mask.sum(), dtype=np.float32)
            tmp_vx = np.zeros(pos_mask.sum(), dtype=np.float32)
            tmp_vy = np.zeros(pos_mask.sum(), dtype=np.float32)
            tmp_rim[ok] = 1.0
            inv_d = np.where(ok, 1.0 / np.maximum(dd, 1e-4), 0.0)
            tmp_vx[ok] = dx[ok] * inv_d[ok]
            tmp_vy[ok] = dy[ok] * inv_d[ok]
            is_rim[pos_mask] = tmp_rim; tvx[pos_mask] = tmp_vx; tvy[pos_mask] = tmp_vy

        pidx = np.where(is_rim > 0.5)[0]
        nidx = np.where(is_rim < 0.5)[0]
        if len(nidx) > len(pidx) * 2:
            nidx = rng.choice(nidx, size=len(pidx) * 2, replace=False)
        keep = np.concatenate([pidx, nidx])
        X_list.append(X[keep])
        tgt_list.append(np.stack([is_rim[keep], tvx[keep], tvy[keep]], axis=1).astype(np.float32))
        w_list.append(np.where(is_rim[keep] > 0.5, 1.5, 1.0).astype(np.float32))

    return (
        np.ascontiguousarray(np.concatenate(X_list, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(tgt_list, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(w_list, axis=0), dtype=np.float32),
    )


def _extract_patches_one_scene(scene, seed):
    rng = np.random.default_rng(seed)
    Y, Cb, Cr, placed, owner = scene
    zeros_mask = np.zeros((500, 500), dtype=np.uint8)
    fg = compute_is_fg(Y, Cb, Cr)

    coarse_X, coarse_Y, coarse_W = [], [], []
    raw_YCC, pV, pCls, pShift, pStats, pLbl, pWt, pTight = [], [], [], [], [], [], [], []
    n_coins = len(placed)

    for k in range(n_coins - 1, -1, -1):
        cls, cx, cy, deg, r = placed[k]
        claimed_top = (owner > k).astype(np.uint8)
        masks = [claimed_top]
        vis_frac = float((owner == k).sum()) / max(1.0, math.pi * r * r)
        if vis_frac >= 0.48 and (claimed_top.sum() > 0) and rng.random() < 0.35:
            masks.append(zeros_mask)
        if rng.random() < 0.30:
            ang_occ = float(rng.uniform(0, 2 * math.pi))
            d_occ = float(rng.uniform(13.8, 24.0))
            ox = cx + d_occ * math.cos(ang_occ); oy = cy + d_occ * math.sin(ang_occ)
            y0, y1 = max(0, int(cy - 25)), min(500, int(cy + 26))
            x0, x1 = max(0, int(cx - 25)), min(500, int(cx + 26))
            if y1 > y0 and x1 > x0:
                yy, xx = np.mgrid[y0:y1, x0:x1]
                m_extra = claimed_top.copy()
                disk = ((xx - ox) ** 2 + (yy - oy) ** 2) <= (18.8 ** 2)
                m_extra[y0:y1, x0:x1][disk] = 1
                masks.append(m_extra)

        s_cont = float(((-deg % 360.0) / (360.0 / A_CNT)) % A_CNT)
        for mi, m_cl in enumerate(masks):
            ang_j1 = float(rng.uniform(0, 2 * math.pi))
            rad_j1 = float(rng.uniform(0.0, 0.55))
            jcx = cx + rad_j1 * math.cos(ang_j1); jcy = cy + rad_j1 * math.sin(ang_j1)
            cf = extract_coarse_features(Y, Cb, Cr, fg, m_cl, jcx, jcy)
            if cf[21] < 0.08:
                continue
            coarse_X.append(cf); coarse_Y.append(cls + 1); coarse_W.append(1.2)

            ang_j2 = float(rng.uniform(0, 2 * math.pi))
            rad_j2 = float(rng.uniform(0.55, 1.85))
            wcx = cx + rad_j2 * math.cos(ang_j2); wcy = cy + rad_j2 * math.sin(ang_j2)
            cf2 = extract_coarse_features(Y, Cb, Cr, fg, m_cl, wcx, wcy)
            if cf2[21] >= 0.08:
                coarse_X.append(cf2); coarse_Y.append(cls + 1); coarse_W.append(0.9)

            if m_cl is not zeros_mask:
                ycc, vis, stats12 = extract_polar_raw(Y, Cb, Cr, fg, m_cl, jcx, jcy)
                if vis.sum() >= 6:
                    raw_YCC.append(ycc); pV.append(vis); pCls.append(cls); pShift.append(s_cont)
                    pStats.append(stats12); pLbl.append(cls + 1)
                    pWt.append(1.2); pTight.append(True)

                if mi == 0 and rng.random() < 0.72:
                    ang_jm = float(rng.uniform(0, 2 * math.pi))
                    rad_jm = float(rng.uniform(0.55, 1.38))
                    mcx = cx + rad_jm * math.cos(ang_jm); mcy = cy + rad_jm * math.sin(ang_jm)
                    ycc_m, vis_m, stats_m = extract_polar_raw(Y, Cb, Cr, fg, m_cl, mcx, mcy)
                    if vis_m.sum() >= 6:
                        raw_YCC.append(ycc_m); pV.append(vis_m); pCls.append(cls); pShift.append(s_cont)
                        pStats.append(stats_m); pLbl.append(cls + 1)
                        pWt.append(0.85); pTight.append(False)

        n_neg_shifts = 4 if (cx < 22.0 or cx > 478.0 or cy < 22.0 or cy > 478.0) else 2
        for _ in range(n_neg_shifts):
            n_ang = float(rng.uniform(0, 2 * math.pi))
            n_dist = float(rng.uniform(2.3, 22.0))
            nx = cx + n_dist * math.cos(n_ang); ny = cy + n_dist * math.sin(n_ang)
            if -22 < nx < 522 and -22 < ny < 522:
                if n_dist >= 3.8:
                    cf = extract_coarse_features(Y, Cb, Cr, fg, claimed_top, nx, ny)
                    coarse_X.append(cf); coarse_Y.append(0); coarse_W.append(1.2)
                if n_dist >= 2.3:
                    ycc, vis, stats12 = extract_polar_raw(Y, Cb, Cr, fg, claimed_top, nx, ny)
                    if vis.sum() >= 6:
                        raw_YCC.append(ycc); pV.append(vis); pCls.append(-1); pShift.append(0.0)
                        pStats.append(stats12); pLbl.append(0)
                        pWt.append(1.4); pTight.append(True)

        if rng.random() < 0.8:
            claimed_incl = (owner >= k).astype(np.uint8)
            hang = float(rng.uniform(0, 2 * math.pi))
            hdist = 0.0 if rng.random() < 0.3 else float(rng.uniform(0.5, 20.0))
            hx, hy = cx + hdist * math.cos(hang), cy + hdist * math.sin(hang)
            cf = extract_coarse_features(Y, Cb, Cr, fg, claimed_incl, hx, hy)
            coarse_X.append(cf); coarse_Y.append(0); coarse_W.append(1.2)
            ycc, vis, stats12 = extract_polar_raw(Y, Cb, Cr, fg, claimed_incl, hx, hy)
            if vis.sum() >= 5:
                raw_YCC.append(ycc); pV.append(vis); pCls.append(-1); pShift.append(0.0)
                pStats.append(stats12); pLbl.append(0)
                pWt.append(1.5); pTight.append(True)

        if cx < 20.0 or cx > 480.0 or cy < 20.0 or cy > 480.0:
            ang_b = float(rng.uniform(0, 2 * math.pi))
            dx_b = -abs(math.cos(ang_b)) if cx < 20.0 else (abs(math.cos(ang_b)) if cx > 480.0 else math.cos(ang_b))
            dy_b = -abs(math.sin(ang_b)) if cy < 20.0 else (abs(math.sin(ang_b)) if cy > 480.0 else math.sin(ang_b))
            dist_b = float(rng.uniform(8.5, 16.0))
            scx = cx + dist_b * dx_b; scy = cy + dist_b * dy_b
            if (scx < -1.5 or scx > 500.5 or scy < -1.5 or scy > 500.5) and min((scx - ox) ** 2 + (scy - oy) ** 2 for _, ox, oy, _, _ in placed) >= 81.0:
                ycc, vis, stats12 = extract_polar_raw(Y, Cb, Cr, fg, claimed_top, scx, scy)
                if vis.sum() >= 5:
                    raw_YCC.append(ycc); pV.append(vis); pCls.append(-1); pShift.append(0.0)
                    pStats.append(stats12); pLbl.append(0)
                    pWt.append(1.4); pTight.append(True)

    return (
        np.array(coarse_X, dtype=np.float32),
        np.array(coarse_Y, dtype=np.int32),
        np.array(coarse_W, dtype=np.float32),
        np.array(raw_YCC, dtype=np.float32),
        np.array(pV, dtype=np.float32),
        np.array(pCls, dtype=np.int32),
        np.array(pShift, dtype=np.float32),
        np.array(pStats, dtype=np.float32),
        np.array(pLbl, dtype=np.int32),
        np.array(pWt, dtype=np.float32),
        np.array(pTight, dtype=bool),
    )


def extract_patches_from_scenes(scenes, seed_base: int, n_workers: int):
    seeds = np.random.SeedSequence(seed_base).spawn(len(scenes))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_extract_patches_one_scene, scenes, seeds))
    return tuple(np.ascontiguousarray(np.concatenate([p[i] for p in parts], axis=0)) for i in range(11))


# ==============================================================================
# GPU Training Loops (Pre-Allocated Workspace + GPU-Resident Gather + Best FP16)
# ==============================================================================

def extract_model_weights(layers):
    return [(l.W.p.numpy(), l.b.p.numpy()) for l in layers]


def train_rim_models(eng: CudaEngine, rim_tr, rim_va, rng):
    X_tr, Y_tr, W_tr = rim_tr
    X_val, Y_val, W_val = rim_va
    N_tr = len(X_tr)
    print(f"Rim dataset (scene-split): train={N_tr}, val={len(X_val)}, pos_frac={(Y_tr[:,0]>0.5).mean():.3f}", flush=True)

    X_tr_g = GpuTensor(X_tr.shape, data=X_tr)
    Y_tr_g = GpuTensor(Y_tr.shape, data=Y_tr)
    Y0_tr_g = GpuTensor((N_tr,), data=np.ascontiguousarray(Y_tr[:, 0]))
    W_tr_g = GpuTensor(W_tr.shape, data=W_tr)
    Wg_tr_g = GpuTensor((N_tr,), data=np.where(Y_tr[:, 0] > 0.5, 0.70, 1.0).astype(np.float32))

    B = 4096
    idx_g = GpuTensor((B,), dtype=np.int32)
    gx = GpuTensor((B, 14))

    gate = [GpuLinear(14, 1, rng)]
    opt_g = AdamW(gate[0].params(), lr=3e-2, wd=1e-4)
    gt = GpuTensor((B,)); gw = GpuTensor((B,))
    go = GpuTensor((B, 1)); gd = GpuTensor((B, 1)); gl = GpuTensor((B,))

    best_gate_w = None
    best_gate_loss = 1e9
    ep_g = 24
    for ep in range(ep_g):
        p = rng.permutation(N_tr).astype(np.int32)
        lr_c = lr_warmup_cosine(ep, ep_g, 3e-2, 5e-4, warmup_ep=2)
        for s in range(0, N_tr, B):
            idx = p[s:s + B]; bs = len(idx)
            idx_g.copy_from(idx, bs * 4)
            eng.launch("gather_rows_f32", bs * 14, X_tr_g.ptr, idx_g.ptr, gx.ptr, ctypes.c_int(bs), ctypes.c_int(14))
            eng.launch("gather_rows_f32", bs, Y0_tr_g.ptr, idx_g.ptr, gt.ptr, ctypes.c_int(bs), ctypes.c_int(1))
            eng.launch("gather_rows_f32", bs, Wg_tr_g.ptr, idx_g.ptr, gw.ptr, ctypes.c_int(bs), ctypes.c_int(1))
            gate[0].forward(eng, gx, go, rows=bs)
            eng.launch("bce_loss_fwd_bwd", bs, go.ptr, gt.ptr, gw.ptr, gd.ptr, gl.ptr, ctypes.c_int(bs))
            gate[0].backward(eng, gd, rows=bs)
            opt_g.step(eng, lr=lr_c)
        if (ep + 1) % 4 == 0 or ep == ep_g - 1:
            eng.sync()
            cur_w = extract_model_weights(gate)
            lg = run_mlp_cpu_f16(cur_w, X_val).ravel()
            m = np.maximum(lg, 0.0)
            y0v = Y_val[:, 0]
            wgv = np.where(y0v > 0.5, 0.70, 1.0)
            vloss = float(np.mean(wgv * (m - y0v * lg + np.log(np.exp(-m) + np.exp(lg - m)))))
            if vloss < best_gate_loss:
                best_gate_loss = vloss
                best_gate_w = cur_w

    g_val_logits = run_mlp_cpu_f16(best_gate_w, X_val).ravel()
    gate_thr = float(np.percentile(g_val_logits[Y_val[:, 0] > 0.5], 4.0))
    print(f"RimGate (FP16 val): loss={best_gate_loss:.4f} | thr={gate_thr:.3f}, pass_rate={(g_val_logits >= gate_thr).mean():.3f}", flush=True)

    for t in (gt, gw, go, gd, gl):
        t.free()
    gate[0].free()

    rim = [GpuLinear(14, 12, rng), GpuLinear(12, 8, rng), GpuLinear(8, 3, rng)]
    opt = AdamW(rim[0].params() + rim[1].params() + rim[2].params(), lr=8e-3, wd=1e-4)

    t0 = GpuTensor((B, 3)); w0 = GpuTensor((B,))
    z1 = GpuTensor((B, 12)); a1 = GpuTensor((B, 12))
    z2 = GpuTensor((B, 8)); a2 = GpuTensor((B, 8))
    z3 = GpuTensor((B, 3)); dz3 = GpuTensor((B, 3))
    da2 = GpuTensor((B, 8)); dz2 = GpuTensor((B, 8))
    da1 = GpuTensor((B, 12)); dz1 = GpuTensor((B, 12)); losses = GpuTensor((B,))

    best_rim_w = None
    best_rim_loss = 1e9
    epochs = 50
    for ep in range(epochs):
        p = rng.permutation(N_tr).astype(np.int32)
        lr_cur = lr_warmup_cosine(ep, epochs, 8e-3, 1e-4, warmup_ep=3)
        for s in range(0, N_tr, B):
            idx = p[s:s + B]; bs = len(idx)
            idx_g.copy_from(idx, bs * 4)
            eng.launch("gather_rows_f32", bs * 14, X_tr_g.ptr, idx_g.ptr, gx.ptr, ctypes.c_int(bs), ctypes.c_int(14))
            eng.launch("gather_rows_f32", bs * 3, Y_tr_g.ptr, idx_g.ptr, t0.ptr, ctypes.c_int(bs), ctypes.c_int(3))
            eng.launch("gather_rows_f32", bs, W_tr_g.ptr, idx_g.ptr, w0.ptr, ctypes.c_int(bs), ctypes.c_int(1))
            rim[0].forward(eng, gx, z1, rows=bs); eng.launch("relu_fwd", bs * 12, z1.ptr, a1.ptr, ctypes.c_int(bs * 12))
            rim[1].forward(eng, a1, z2, rows=bs); eng.launch("relu_fwd", bs * 8, z2.ptr, a2.ptr, ctypes.c_int(bs * 8))
            rim[2].forward(eng, a2, z3, rows=bs)
            eng.launch("rim_loss_fwd_bwd", bs, z3.ptr, t0.ptr, w0.ptr, dz3.ptr, losses.ptr, ctypes.c_int(bs))
            rim[2].backward(eng, dz3, da2, rows=bs); eng.launch("relu_bwd", bs * 8, z2.ptr, da2.ptr, dz2.ptr, ctypes.c_int(bs * 8))
            rim[1].backward(eng, dz2, da1, rows=bs); eng.launch("relu_bwd", bs * 12, z1.ptr, da1.ptr, dz1.ptr, ctypes.c_int(bs * 12))
            rim[0].backward(eng, dz1, rows=bs)
            opt.step(eng, lr=lr_cur)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            eng.sync()
            cur_w = extract_model_weights(rim)
            vp = run_mlp_cpu_f16(cur_w, X_val)
            lg = vp[:, 0]; y0v = Y_val[:, 0]
            m = np.maximum(lg, 0.0)
            bce = W_val * (m - y0v * lg + np.log(np.exp(-m) + np.exp(lg - m)))
            dx = np.abs(vp[:, 1] - Y_val[:, 1]); dy = np.abs(vp[:, 2] - Y_val[:, 2])
            hx = np.where(dx <= 0.2, 0.5 * dx * dx, 0.2 * (dx - 0.1))
            hy = np.where(dy <= 0.2, 0.5 * dy * dy, 0.2 * (dy - 0.1))
            vloss = float(np.mean(bce + np.where(y0v > 0.5, 5.0, 0.0) * (hx + hy)))
            if vloss < best_rim_loss:
                best_rim_loss = vloss
                best_rim_w = cur_w

    vp = run_mlp_cpu_f16(best_rim_w, X_val)
    pm = Y_val[:, 0] > 0.5
    ang_err = np.degrees(np.arccos(np.clip(
        (vp[pm, 1] * Y_val[pm, 1] + vp[pm, 2] * Y_val[pm, 2]) / np.maximum(np.hypot(vp[pm, 1], vp[pm, 2]), 1e-4),
        -1.0, 1.0
    )))
    print(f"RimNet (FP16 val): loss={best_rim_loss:.4f} | rim_acc={((vp[:,0]>0)==pm).mean():.4f} | dir_med_err={np.median(ang_err):.2f} deg", flush=True)

    for l in rim:
        l.free()
    for t in (X_tr_g, Y_tr_g, Y0_tr_g, W_tr_g, Wg_tr_g, idx_g, gx, t0, w0, z1, a1, z2, a2, z3, dz3, da2, dz2, da1, dz1, losses):
        t.free()
    return best_gate_w[0], gate_thr, best_rim_w


def train_mlp_ce(eng, layers, X_tr, Y_tr, W_tr, X_val, Y_val, W_val, rng, epochs=50, lr_max=7e-3, bs=2048, wd=1e-4, warmup_ep=3, name="MLP"):
    N_tr = len(X_tr)
    in_f = layers[0].in_f
    n_cls = layers[-1].out_f
    params = []
    for l in layers:
        params += l.params()
    opt = AdamW(params, lr=lr_max, wd=wd)

    X_g = GpuTensor(X_tr.shape, data=X_tr)
    Y_g = GpuTensor(Y_tr.shape, dtype=np.int32, data=Y_tr)
    W_g = GpuTensor(W_tr.shape, data=W_tr)
    idx_g = GpuTensor((bs,), dtype=np.int32)
    x0 = GpuTensor((bs, in_f))
    y0 = GpuTensor((bs,), dtype=np.int32)
    w0 = GpuTensor((bs,))
    zs = [GpuTensor((bs, l.out_f)) for l in layers]
    acts = [GpuTensor((bs, l.out_f)) for l in layers[:-1]]
    dl = GpuTensor((bs, n_cls))
    losses = GpuTensor((bs,))
    das = [GpuTensor((bs, layers[li].in_f)) for li in range(1, len(layers))]
    dzs = [GpuTensor((bs, layers[li].in_f)) for li in range(1, len(layers))]

    best_w = None
    best_vloss = 1e9
    for ep in range(epochs):
        p = rng.permutation(N_tr).astype(np.int32)
        lr_c = lr_warmup_cosine(ep, epochs, lr_max, 1e-4, warmup_ep=warmup_ep)
        for s in range(0, N_tr, bs):
            idx = p[s:s + bs]; b = len(idx)
            idx_g.copy_from(idx, b * 4)
            eng.launch("gather_rows_f32", b * in_f, X_g.ptr, idx_g.ptr, x0.ptr, ctypes.c_int(b), ctypes.c_int(in_f))
            eng.launch("gather_rows_i32", b, Y_g.ptr, idx_g.ptr, y0.ptr, ctypes.c_int(b))
            eng.launch("gather_rows_f32", b, W_g.ptr, idx_g.ptr, w0.ptr, ctypes.c_int(b), ctypes.c_int(1))

            cur_in = x0
            for li, l in enumerate(layers):
                l.forward(eng, cur_in, zs[li], rows=b)
                if li < len(layers) - 1:
                    eng.launch("relu_fwd", b * l.out_f, zs[li].ptr, acts[li].ptr, ctypes.c_int(b * l.out_f))
                    cur_in = acts[li]

            eng.launch("ce_weighted_fwd_bwd", b, zs[-1].ptr, y0.ptr, w0.ptr, dl.ptr, losses.ptr, ctypes.c_int(b), ctypes.c_int(n_cls))

            grad = dl
            for li in range(len(layers) - 1, -1, -1):
                if li > 0:
                    da = das[li - 1]; dz = dzs[li - 1]
                    layers[li].backward(eng, grad, da, rows=b)
                    eng.launch("relu_bwd", b * layers[li].in_f, zs[li - 1].ptr, da.ptr, dz.ptr, ctypes.c_int(b * layers[li].in_f))
                    grad = dz
                else:
                    layers[li].backward(eng, grad, rows=b)
            opt.step(eng, lr=lr_c)

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            eng.sync()
            cur_w = extract_model_weights(layers)
            vp = run_mlp_cpu_f16(cur_w, X_val)
            mx = vp.max(axis=1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(vp - mx).sum(axis=1))
            vloss = float(np.mean(W_val * (lse - vp[np.arange(len(Y_val)), Y_val])))
            if vloss < best_vloss:
                best_vloss = vloss
                best_w = cur_w

    vp = run_mlp_cpu_f16(best_w, X_val)
    pred = np.argmax(vp, axis=1)
    pos_m = Y_val > 0; neg_m = Y_val == 0
    val_acc = float((pred == Y_val).mean())
    print(
        f"  {name} (FP16 val): loss={best_vloss:.4f} | acc={val_acc:.4f} | "
        f"pos_cls={(pred[pos_m]==Y_val[pos_m]).mean():.4f} | pos_rec={(pred[pos_m]>0).mean():.4f} | neg_spec={(pred[neg_m]==0).mean():.4f}",
        flush=True,
    )

    for l in layers:
        l.free()
    for t in [X_g, Y_g, W_g, idx_g, x0, y0, w0, dl, losses] + zs + acts + das + dzs:
        t.free()
    return best_w, val_acc


def train_polar_w_cls(eng: CudaEngine, F_tr, V_tr, Cls_tr, Sh_tr, F_va, V_va, Cls_va, rng_v):
    N_tr = len(F_tr)
    N_LOGITS = 3 * A_CNT
    W_sz = 3 * CR_CNT * A_CNT
    TILES = 16
    bs_p = 512
    ep_p = 50
    tau = 14.0
    ang_reg = 4e-4

    W_cls = Param(rng_v.normal(0.0, 0.08, (3, CR_CNT, A_CNT)).astype(np.float32), decay=True)
    opt_p = AdamW([W_cls], lr=1.2e-2, wd=1e-4, clip_val=5.0)

    F_tr_g = GpuTensor(F_tr.shape, data=F_tr)
    V_tr_g = GpuTensor(V_tr.shape, data=V_tr)
    Cls_tr_g = GpuTensor(Cls_tr.shape, dtype=np.int32, data=Cls_tr)
    Sh_tr_g = GpuTensor(Sh_tr.shape, data=Sh_tr)

    idx_g = GpuTensor((bs_p,), dtype=np.int32)
    Fg = GpuTensor((bs_p, CR_CNT, A_CNT))
    Vg = GpuTensor((bs_p, R_CNT, A_CNT))
    Clsg = GpuTensor((bs_p,), dtype=np.int32)
    Shg = GpuTensor((bs_p,))
    num_g = GpuTensor((bs_p, N_LOGITS))
    df_g = GpuTensor((bs_p,))
    dw_g = GpuTensor((bs_p, N_LOGITS))
    sim_g = GpuTensor((bs_p, N_LOGITS))
    dsim_g = GpuTensor((bs_p, N_LOGITS))
    los_g = GpuTensor((bs_p,))
    part_g = GpuTensor((TILES, W_sz))

    pos_va = np.where(Cls_va >= 0)[0]
    bs_v = len(pos_va)
    vF = GpuTensor((bs_v, CR_CNT, A_CNT), data=F_va[pos_va])
    vV = GpuTensor((bs_v, R_CNT, A_CNT), data=V_va[pos_va])
    vW_f16 = GpuTensor((3, CR_CNT, A_CNT))
    vdf = GpuTensor((bs_v,))
    vnum = GpuTensor((bs_v, N_LOGITS))
    vdw = GpuTensor((bs_v, N_LOGITS))
    eng.launch("polar_denF_fwd", bs_v, vF.ptr, vV.ptr, vdf.ptr, ctypes.c_int(bs_v), ctypes.c_int(C_CNT), ctypes.c_int(R_CNT))
    vdf_np = vdf.numpy()
    va_cls_gt = Cls_va[pos_va]

    best_W_f16 = None
    best_acc = -1.0
    for ep in range(ep_p):
        perm = rng_v.permutation(N_tr).astype(np.int32)
        lr_p = lr_warmup_cosine(ep, ep_p, 1.2e-2, 8e-4, warmup_ep=3)
        for s in range(0, N_tr, bs_p):
            idx = perm[s:s + bs_p]; bs = len(idx)
            idx_g.copy_from(idx, bs * 4)
            eng.launch("gather_rows_f32", bs * CR_CNT * A_CNT, F_tr_g.ptr, idx_g.ptr, Fg.ptr, ctypes.c_int(bs), ctypes.c_int(CR_CNT * A_CNT))
            eng.launch("gather_rows_f32", bs * R_CNT * A_CNT, V_tr_g.ptr, idx_g.ptr, Vg.ptr, ctypes.c_int(bs), ctypes.c_int(R_CNT * A_CNT))
            eng.launch("gather_rows_i32", bs, Cls_tr_g.ptr, idx_g.ptr, Clsg.ptr, ctypes.c_int(bs))
            eng.launch("gather_rows_f32", bs, Sh_tr_g.ptr, idx_g.ptr, Shg.ptr, ctypes.c_int(bs), ctypes.c_int(1))

            eng.launch("polar_denF_fwd", bs, Fg.ptr, Vg.ptr, df_g.ptr, ctypes.c_int(bs), ctypes.c_int(C_CNT), ctypes.c_int(R_CNT))
            eng.launch("polar_sim_fwd", bs * N_LOGITS, Fg.ptr, Vg.ptr, W_cls.p.ptr, num_g.ptr, dw_g.ptr, ctypes.c_int(bs), ctypes.c_int(C_CNT), ctypes.c_int(R_CNT))
            eng.launch("polar_sim_norm_ce_or_neg", bs, num_g.ptr, df_g.ptr, dw_g.ptr, Clsg.ptr, Shg.ptr, sim_g.ptr, dsim_g.ptr, los_g.ptr, ctypes.c_int(bs), ctypes.c_float(tau))
            eng.launch("polar_sim_bwd_W_partial", TILES * W_sz, Fg.ptr, Vg.ptr, W_cls.p.ptr, dsim_g.ptr, num_g.ptr, df_g.ptr, dw_g.ptr, part_g.ptr, ctypes.c_int(bs), ctypes.c_int(C_CNT), ctypes.c_int(R_CNT), ctypes.c_int(TILES))
            eng.launch("polar_sim_bwd_W_reduce", W_sz, part_g.ptr, W_cls.p.ptr, W_cls.g.ptr, ctypes.c_int(C_CNT), ctypes.c_int(R_CNT), ctypes.c_int(TILES), ctypes.c_float(ang_reg))
            opt_p.step(eng, lr=lr_p)

        if (ep + 1) % 5 == 0 or ep == ep_p - 1:
            eng.sync()
            cur_W = f16_roundtrip(W_cls.p.numpy())
            vW_f16.copy_from(cur_W)
            eng.launch("polar_sim_fwd", bs_v * N_LOGITS, vF.ptr, vV.ptr, vW_f16.ptr, vnum.ptr, vdw.ptr, ctypes.c_int(bs_v), ctypes.c_int(C_CNT), ctypes.c_int(R_CNT))
            eng.sync()
            sims = (vnum.numpy() / np.sqrt(vdf_np[:, None] * vdw.numpy())).reshape(bs_v, 3, A_CNT)
            acc = float((np.argmax(sims.max(axis=2), axis=1) == va_cls_gt).mean())
            if acc > best_acc:
                best_acc = acc
                best_W_f16 = cur_W.reshape(3, C_CNT, R_CNT, A_CNT)

    print(f"  [C4E] Polar FP16 val_pos_cls_acc={best_acc:.4f}", flush=True)
    W_cls.free()
    for t in (F_tr_g, V_tr_g, Cls_tr_g, Sh_tr_g, idx_g, Fg, Vg, Clsg, Shg, num_g, df_g, dw_g, sim_g, dsim_g, los_g, part_g, vF, vV, vW_f16, vdf, vnum, vdw):
        t.free()
    return best_W_f16


# ==============================================================================
# Fully Vectorized AcceptNet Feature Builder
# ==============================================================================

def build_accept_features_batch(W_cls_np, F_arr, vis_arr, stats_arr):
    N = F_arr.shape[0]
    if N == 0:
        return np.empty((0, 33), dtype=np.float32)
    feats = np.empty((N, 33), dtype=np.float32)
    W_fft = np.fft.rfft(W_cls_np.reshape(3, CR_CNT, A_CNT), axis=2)
    W2_fft = np.fft.rfft((W_cls_np ** 2).reshape(3, CR_CNT, A_CNT), axis=2)

    W_rot = np.stack(
        [np.stack([np.roll(W_cls_np[k], sh, axis=2) for sh in range(A_CNT)], axis=0) for k in range(3)],
        axis=0,
    ).astype(np.float32)
    W_hp = 0.5 * (W_rot + np.roll(W_rot, -1, axis=1))
    W_hm = 0.5 * (W_rot + np.roll(W_rot, 1, axis=1))

    batch = 1024
    for s in range(0, N, batch):
        e = min(N, s + batch)
        B = e - s
        fb = F_arr[s:e]; vb = vis_arr[s:e]
        vis_cr = np.repeat(vb[:, None, :, :], C_CNT, axis=1).reshape(B, CR_CNT, A_CNT)
        denF = np.sum(vis_cr * fb * fb, axis=(1, 2)) + 1e-5
        F_fft = np.fft.rfft(vis_cr * fb, axis=2)
        V_fft = np.fft.rfft(vis_cr, axis=2)
        num = np.fft.irfft(np.einsum("bdk,cdk->bck", F_fft, np.conj(W_fft)), n=A_CNT, axis=2)
        denW = np.maximum(0.0, np.fft.irfft(np.einsum("bdk,cdk->bck", V_fft, np.conj(W2_fft)), n=A_CNT, axis=2)) + 1e-5
        sim = num / np.sqrt(denF[:, None, None] * denW)

        s_best = np.argmax(sim, axis=2)
        b_idx = np.arange(B)
        sim_best = sim[b_idx[:, None], np.arange(3)[None, :], s_best]
        sim_mean = sim.mean(axis=2)

        out = np.empty((B, 33), dtype=np.float32)
        out[:, 21:33] = stats_arr[s:e]

        fb_r = fb.reshape(B, C_CNT, R_CNT, A_CNT)
        vb_e = vb[:, None, :, :]
        vfb = vb_e * fb_r
        df0 = np.sum(vfb[:, :, :2] * fb_r[:, :, :2], axis=(1, 2, 3)) + 1e-5
        df1 = np.sum(vfb[:, :, 2:] * fb_r[:, :, 2:], axis=(1, 2, 3)) + 1e-5

        for k in range(3):
            sk = s_best[:, k]
            W_sh = W_rot[k, sk]
            Wp = W_hp[k, sk]
            Wm = W_hm[k, sk]

            c_sub_p = np.sum(vfb * Wp, axis=(1, 2, 3)) / np.sqrt(denF * (np.sum(vb_e * (Wp * Wp), axis=(1, 2, 3)) + 1e-5))
            c_sub_m = np.sum(vfb * Wm, axis=(1, 2, 3)) / np.sqrt(denF * (np.sum(vb_e * (Wm * Wm), axis=(1, 2, 3)) + 1e-5))
            sb_k = sim_best[:, k]
            sim_sub = np.maximum(sb_k, np.maximum(c_sub_p, c_sub_m))

            j0, j1 = (1, 2) if k == 0 else ((0, 2) if k == 1 else (0, 1))
            other_max = np.maximum(sim_best[:, j0], sim_best[:, j1])

            dw_in = np.sum(vb_e[:, :, :2] * (W_sh[:, :, :2] ** 2), axis=(1, 2, 3)) + 1e-5
            dw_out = np.sum(vb_e[:, :, 2:] * (W_sh[:, :, 2:] ** 2), axis=(1, 2, 3)) + 1e-5
            dw_all = dw_in + dw_out - 1e-5
            num_in = np.sum(vfb[:, :, :2] * W_sh[:, :, :2], axis=(1, 2, 3))
            num_out = num[b_idx, k, sk] - num_in
            log_e = np.clip(0.5 * np.log(denF / dw_all), -3.0, 3.0)

            base = k * 7
            out[:, base + 0] = sb_k
            out[:, base + 1] = sim_sub
            out[:, base + 2] = sb_k - sim_mean[:, k]
            out[:, base + 3] = sb_k - other_max
            out[:, base + 4] = num_in / np.sqrt(df0 * dw_in)
            out[:, base + 5] = num_out / np.sqrt(df1 * dw_out)
            out[:, base + 6] = log_e

        feats[s:e] = out
    return feats


# ==============================================================================
# 100% Lossless Predictive + Canonical Huffman Weight Packer & Verifier
# ==============================================================================

WEIGHT_SCHEMA = [
    ("rim_gate_W", (1, 14)), ("rim_gate_b", (1,)), ("rim_gate_thr", (1,)),
    ("rim0_W", (12, 14)), ("rim0_b", (12,)),
    ("rim1_W", (8, 12)), ("rim1_b", (8,)),
    ("rim2_W", (3, 8)), ("rim2_b", (3,)),
    ("coarse0_W", (12, 25)), ("coarse0_b", (12,)),
    ("coarse1_W", (8, 12)), ("coarse1_b", (8,)),
    ("coarse2_W", (4, 8)), ("coarse2_b", (4,)),
    ("W_cls", (3, C_CNT, R_CNT, A_CNT)),
    ("acc0_W", (20, 33)), ("acc0_b", (20,)),
    ("acc1_W", (10, 20)), ("acc1_b", (10,)),
    ("acc2_W", (4, 10)), ("acc2_b", (4,)),
    ("radii_500", (3,)),
]


def _build_canonical_huff_lengths(freqs, n_syms, max_len=15):
    while True:
        heap = []
        uid = 0
        for sym in range(n_syms):
            f = freqs.get(sym, 0)
            if f > 0:
                heap.append((f, uid, sym))
                uid += 1
        if not heap:
            return [0] * n_syms
        if len(heap) == 1:
            lens = [0] * n_syms; lens[heap[0][2]] = 1; return lens
        heapq.heapify(heap)
        while len(heap) > 1:
            f1, _, n1 = heapq.heappop(heap)
            f2, _, n2 = heapq.heappop(heap)
            heapq.heappush(heap, (f1 + f2, uid, (n1, n2)))
            uid += 1
        lens = [0] * n_syms

        def walk(node, d):
            if isinstance(node, int):
                lens[node] = d
            else:
                walk(node[0], d + 1); walk(node[1], d + 1)

        walk(heap[0][2], 0)
        if max(lens) <= max_len:
            return lens
        freqs = {s: (f + 1) // 2 if f > 1 else 1 for s, f in freqs.items() if f > 0}


def _huff_encode_stream(seq, n_syms):
    freqs = Counter(seq)
    lens = _build_canonical_huff_lengths(freqs, n_syms, max_len=15)
    pairs = sorted((l, s) for s, l in enumerate(lens) if l > 0)
    codes = {}; code = 0; prev_l = 0
    for l, s in pairs:
        code <<= (l - prev_l)
        codes[s] = (l, code)
        code += 1; prev_l = l
    hdr = bytearray(n_syms // 2)
    for i in range(0, n_syms, 2):
        hdr[i >> 1] = ((lens[i] & 0xF) << 4) | (lens[i + 1] & 0xF)
    bitbuf = 0; bitcnt = 0; out = bytearray()
    for sym in seq:
        blen, c = codes[sym]
        bitbuf = (bitbuf << blen) | c; bitcnt += blen
        while bitcnt >= 8:
            bitcnt -= 8; out.append((bitbuf >> bitcnt) & 0xFF)
    if bitcnt > 0:
        out.append((bitbuf << (8 - bitcnt)) & 0xFF)
    return bytes(hdr), bytes(out)


def _huff_decode_stream(hdr: bytes, data: bytes, n_syms: int, n_out: int):
    lens = [0] * n_syms
    for i in range(0, n_syms, 2):
        b = hdr[i >> 1]
        lens[i] = (b >> 4) & 0xF
        lens[i + 1] = b & 0xF
    pairs = sorted((l, s) for s, l in enumerate(lens) if l > 0)
    lut = {}; code = 0; prev_l = 0
    for l, s in pairs:
        code <<= (l - prev_l)
        lut[(l, code)] = s
        code += 1; prev_l = l

    out = []
    cur_code = 0; cur_len = 0
    for byte in data:
        for shift in range(7, -1, -1):
            cur_code = (cur_code << 1) | ((byte >> shift) & 1)
            cur_len += 1
            sym = lut.get((cur_len, cur_code))
            if sym is not None:
                out.append(sym)
                cur_code = 0; cur_len = 0
                if len(out) == n_out:
                    return out
    raise RuntimeError(f"Huffman decode underrun: got {len(out)}/{n_out}")


def pack_weights_lossless(bundle: dict) -> str:
    se_stream = []; m8_stream = []; lo2_bits = []
    n_tot = 0
    for k, shp in WEIGHT_SCHEMA:
        arr = np.asarray(bundle[k], dtype=np.float16).reshape(shp)
        vals = [int(x) for x in np.frombuffer(arr.tobytes(), dtype=np.uint16)]
        sz = len(vals); n_tot += sz
        if k == "W_cls":
            for row0 in range(0, sz, A_CNT):
                prev_se = 0; prev_m8 = 0
                for a in range(A_CNT):
                    v = vals[row0 + a]
                    se = (v >> 10) & 0x3F; m8 = (v >> 2) & 0xFF
                    se_stream.append(se)
                    m8_stream.append(((m8 - prev_m8) & 0xFF) if se == prev_se else m8)
                    lo2_bits.append(v & 3)
                    prev_se = se; prev_m8 = m8
        else:
            for v in vals:
                se_stream.append((v >> 10) & 0x3F)
                m8_stream.append((v >> 2) & 0xFF)
                lo2_bits.append(v & 3)
    hdr_se, bs_se = _huff_encode_stream(se_stream, 64)
    hdr_m8, bs_m8 = _huff_encode_stream(m8_stream, 256)
    lo2_bytes = bytearray((len(lo2_bits) + 3) // 4)
    for i, b2 in enumerate(lo2_bits):
        lo2_bytes[i >> 2] |= (b2 & 3) << ((3 - (i & 3)) * 2)
    blob = struct.pack("<HHH", n_tot, len(bs_se), len(bs_m8)) + hdr_se + hdr_m8 + bs_se + bs_m8 + bytes(lo2_bytes)
    b85 = base64.b85encode(blob).decode("ascii")
    unpack_weights_verify(b85, bundle)
    return b85


def unpack_weights_verify(b85: str, bundle: dict):
    blob = base64.b85decode(b85.encode("ascii"))
    n_tot, len_se, len_m8 = struct.unpack_from("<HHH", blob, 0)
    off = 6
    hdr_se = blob[off:off + 32]; off += 32
    hdr_m8 = blob[off:off + 128]; off += 128
    bs_se = blob[off:off + len_se]; off += len_se
    bs_m8 = blob[off:off + len_m8]; off += len_m8
    lo2_bytes = blob[off:]

    se_dec = _huff_decode_stream(hdr_se, bs_se, 64, n_tot)
    m8_dec = _huff_decode_stream(hdr_m8, bs_m8, 256, n_tot)
    pos = 0
    for k, shp in WEIGHT_SCHEMA:
        sz = int(np.prod(shp))
        u16 = np.empty(sz, dtype=np.uint16)
        if k == "W_cls":
            for row0 in range(0, sz, A_CNT):
                prev_se = 0; prev_m8 = 0
                for a in range(A_CNT):
                    se = se_dec[pos]; m8_raw = m8_dec[pos]
                    lo2 = (lo2_bytes[pos >> 2] >> ((3 - (pos & 3)) * 2)) & 3
                    pos += 1
                    m8 = ((prev_m8 + m8_raw) & 0xFF) if se == prev_se else m8_raw
                    u16[row0 + a] = (se << 10) | (m8 << 2) | lo2
                    prev_se = se; prev_m8 = m8
        else:
            for j in range(sz):
                se = se_dec[pos]; m8 = m8_dec[pos]
                lo2 = (lo2_bytes[pos >> 2] >> ((3 - (pos & 3)) * 2)) & 3
                pos += 1
                u16[j] = (se << 10) | (m8 << 2) | lo2
        expected = np.asarray(bundle[k], dtype=np.float16).reshape(shp).view(np.uint16)
        if not np.array_equal(u16.reshape(shp), expected):
            raise RuntimeError(f"Lossless pack verification FAILED for tensor '{k}'!")
    assert pos == n_tot


# ==============================================================================
# Stage-2 Hard-Example Mining & Combined End-to-End Evaluation (via generate_solution)
# ==============================================================================

def _build_runtime_module_from_template(b85: str):
    """Loads CoinDetector directly from generate_solution.SOLUTION_TEMPLATE in-memory."""
    from generate_solution import SOLUTION_TEMPLATE
    src = SOLUTION_TEMPLATE.replace("__WEIGHTS_B85__", b85, 1)
    ns = {}
    exec(compile(src, "<solution_mem>", "exec"), ns)
    return ns


def evaluate_on_combined_suite(b85: str, test_suite: list, label: str = "Eval"):
    ns = _build_runtime_module_from_template(b85)
    weights = ns["load_weights"](b85)
    det = ns["CoinDetector"](weights)
    exact = 0
    tot_abs_err = 0
    print(f"\n=== {label}: End-to-End Combined Test Suite ({len(test_suite)} cases) ===", flush=True)
    for name, jpg_bytes, expected in test_suite:
        t0 = time.perf_counter()
        pred = det.solve(jpg_bytes, time_budget_s=1.80)
        dt = time.perf_counter() - t0
        err = pred - expected
        tot_abs_err += abs(err)
        ok = (err == 0)
        exact += int(ok)
        st = "OK" if ok else f"DIFF {err:+d}"
        print(f"  {name:28s} pred={pred:5d}  ans={expected:5d}  [{st:10s}]  {dt:.3f}s", flush=True)
    print(f"  -> {label} Summary: exact={exact}/{len(test_suite)} | total |err|={tot_abs_err}", flush=True)
    return exact, tot_abs_err


def mine_hard_examples(b85: str, mining_scenes: list, W_cls_np: np.ndarray, rng):
    """
    Runs the REAL CoinDetector on separate combined training scenes (with exact GT coordinates)
    to harvest runtime False Positives, Wrong-Class Detections, and Missed Coins!
    """
    ns = _build_runtime_module_from_template(b85)
    weights = ns["load_weights"](b85)
    BaseDetector = ns["CoinDetector"]

    class MiningDetector(BaseDetector):
        def _propose_centers(self, Y, Cb, Cr, fg, fg_rows=None, deadline=None):
            self._cap_planes = (Y, Cb, Cr, fg)
            self._cap_events = []
            self._cap_final_claimed = None
            return super()._propose_centers(Y, Cb, Cr, fg, fg_rows=fg_rows, deadline=deadline)

        def _mark_claimed(self, claimed, claimed_rows, cx, cy, r):
            self._cap_events.append((cx, cy, r, bytes(claimed)))
            self._cap_final_claimed = claimed
            return super()._mark_claimed(claimed, claimed_rows, cx, cy, r)

    det = MiningDetector(weights)

    mc_X, mc_Y, mc_W = [], [], []
    mp_YCC, mp_V, mp_Stats, mp_Lbl, mp_Wt = [], [], [], [], []
    n_fp = n_wc = n_fn = 0

    for _Y_s, _Cb_s, _Cr_s, placed_500, _owner_500, jpg_bytes, _exact_ans in mining_scenes:
        det.solve(jpg_bytes, time_budget_s=1.80)
        if not hasattr(det, "_cap_planes") or det._cap_final_claimed is None:
            continue
        Y_b, Cb_b, Cr_b, fg_b = det._cap_planes
        Y = np.frombuffer(Y_b, dtype=np.uint8).reshape(500, 500).astype(np.float32)
        Cb = np.frombuffer(Cb_b, dtype=np.uint8).reshape(500, 500).astype(np.float32)
        Cr = np.frombuffer(Cr_b, dtype=np.uint8).reshape(500, 500).astype(np.float32)
        fg = np.frombuffer(fg_b, dtype=np.uint8).reshape(500, 500).astype(bool)
        final_cl = np.frombuffer(det._cap_final_claimed, dtype=np.uint8).reshape(500, 500).copy()

        gt_matched = [False] * len(placed_500)
        for cx, cy, r_mark, pre_cl_bytes in det._cap_events:
            pre_cl = np.frombuffer(pre_cl_bytes, dtype=np.uint8).reshape(500, 500)
            # Determine exact predicted class from the runtime detector at pre_cl
            det._pt_cache.clear(); det._eval_cache.clear()
            ev = det._eval_polar_and_accept(Y_b, Cb_b, Cr_b, fg_b, bytearray(pre_cl_bytes), cx, cy)
            cls_pred = ev[1] if ev is not None else int(np.argmin([abs((r_mark - 0.12) - rk) for rk in det.radii_500]))

            best_gi = -1
            best_d2 = 16.0  # 4.0 px radius at 500x500
            for gi, (gcls, gcx, gcy, _gdeg, _gr) in enumerate(placed_500):
                if gt_matched[gi]:
                    continue
                d2 = (cx - gcx) ** 2 + (cy - gcy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_gi = gi

            if best_gi < 0:
                # Runtime False Positive -> hard negative (label=0)
                n_fp += 1
                for jx, jy in ((0.0, 0.0), (float(rng.uniform(-0.6, 0.6)), float(rng.uniform(-0.6, 0.6)))):
                    fx, fy = cx + jx, cy + jy
                    mc_X.append(extract_coarse_features(Y, Cb, Cr, fg, pre_cl, fx, fy))
                    mc_Y.append(0); mc_W.append(2.5)
                    ycc, vis, st12 = extract_polar_raw(Y, Cb, Cr, fg, pre_cl, fx, fy)
                    if vis.sum() >= 5:
                        mp_YCC.append(ycc); mp_V.append(vis); mp_Stats.append(st12)
                        mp_Lbl.append(0); mp_Wt.append(2.6)
            else:
                gt_matched[best_gi] = True
                gcls, gcx, gcy, _gdeg, _gr = placed_500[best_gi]
                if cls_pred != gcls:
                    # Runtime Wrong-Class Detection -> hard positive with true class
                    n_wc += 1
                    for jx, jy in ((0.0, 0.0), (cx - gcx, cy - gcy)):
                        fx, fy = gcx + jx, gcy + jy
                        mc_X.append(extract_coarse_features(Y, Cb, Cr, fg, pre_cl, fx, fy))
                        mc_Y.append(gcls + 1); mc_W.append(2.5)
                        ycc, vis, st12 = extract_polar_raw(Y, Cb, Cr, fg, pre_cl, fx, fy)
                        if vis.sum() >= 5:
                            mp_YCC.append(ycc); mp_V.append(vis); mp_Stats.append(st12)
                            mp_Lbl.append(gcls + 1); mp_Wt.append(2.8)

        # Runtime Missed Coins (False Negatives) under final_cl
        for gi, matched in enumerate(gt_matched):
            if matched:
                continue
            gcls, gcx, gcy, _gdeg, _gr = placed_500[gi]
            n_fn += 1
            for rad_j in (0.0, 0.45, 0.90):
                ang_j = float(rng.uniform(0.0, 2.0 * math.pi))
                fx = gcx + rad_j * math.cos(ang_j)
                fy = gcy + rad_j * math.sin(ang_j)
                cf = extract_coarse_features(Y, Cb, Cr, fg, final_cl, fx, fy)
                if cf[21] >= 0.06:
                    mc_X.append(cf); mc_Y.append(gcls + 1); mc_W.append(2.4)
                ycc, vis, st12 = extract_polar_raw(Y, Cb, Cr, fg, final_cl, fx, fy)
                if vis.sum() >= 5:
                    mp_YCC.append(ycc); mp_V.append(vis); mp_Stats.append(st12)
                    mp_Lbl.append(gcls + 1); mp_Wt.append(2.6)

    print(f"Stage-2 Mining across {len(mining_scenes)} combined train scenes: FP={n_fp}, WrongCls={n_wc}, Missed={n_fn} -> mined Coarse={len(mc_X)}, Accept={len(mp_Lbl)}", flush=True)

    if len(mp_YCC) > 0:
        F_m = build_polar_features_c4e(np.array(mp_YCC, dtype=np.float32), np.array(mp_V, dtype=np.float32))
        X33_m = build_accept_features_batch(W_cls_np, F_m, np.array(mp_V, dtype=np.float32), np.array(mp_Stats, dtype=np.float32))
        acc_mined = (X33_m, np.array(mp_Lbl, dtype=np.int32), np.array(mp_Wt, dtype=np.float32))
    else:
        acc_mined = None

    if len(mc_X) > 0:
        coarse_mined = (np.array(mc_X, dtype=np.float32), np.array(mc_Y, dtype=np.int32), np.array(mc_W, dtype=np.float32))
    else:
        coarse_mined = None

    return coarse_mined, acc_mined


def _assemble_bundle(rim_gate_w, gate_thr, rim_w, coarse_w, W_cls_np, acc_w, radii_500):
    return {
        "rim_gate_W": rim_gate_w[0], "rim_gate_b": rim_gate_w[1], "rim_gate_thr": np.array([gate_thr], dtype=np.float32),
        "rim0_W": rim_w[0][0], "rim0_b": rim_w[0][1],
        "rim1_W": rim_w[1][0], "rim1_b": rim_w[1][1],
        "rim2_W": rim_w[2][0], "rim2_b": rim_w[2][1],
        "coarse0_W": coarse_w[0][0], "coarse0_b": coarse_w[0][1],
        "coarse1_W": coarse_w[1][0], "coarse1_b": coarse_w[1][1],
        "coarse2_W": coarse_w[2][0], "coarse2_b": coarse_w[2][1],
        "W_cls": W_cls_np,
        "acc0_W": acc_w[0][0], "acc0_b": acc_w[0][1],
        "acc1_W": acc_w[1][0], "acc1_b": acc_w[1][1],
        "acc2_W": acc_w[2][0], "acc2_b": acc_w[2][1],
        "radii_500": radii_500,
    }


def main():
    no_inject = "--no-inject" in sys.argv[1:]
    save_tests = "--save-tests" in sys.argv[1:]
    t0 = time.perf_counter()
    n_workers = min(8, os.cpu_count() or 4)
    rng = np.random.default_rng(2026)

    sprites, alphas, radii, coins_by_cls, raw_samples, verified_non_overlap, overlap_samples = extract_all_sample_data()
    radii_500 = np.array([r * 0.5 for r in radii], dtype=np.float32)

    # Build held-out Combined End-to-End Test Suite from ALL ./sampleimages/
    test_suite = build_combined_test_suite(
        raw_samples, verified_non_overlap, overlap_samples, alphas,
        save_dir=(COMBINED_TESTS_DIR if save_tests else None),
    )

    # Mix of Parametric Synthetic Scenes + Real-on-Real Combined Sample Image Scenes
    synth_specs = [
        ("g1", 95, 4),
        ("g2", 95, 5),
        ("g3", 150, 10),
        ("dense1", 215, 6),
        ("g4", 95, 5),
        ("g5", 95, 6),
        ("g6", 155, 18),
        ("dense", 195, 9),
        ("dense6", 230, 8),
    ]
    combo_train_specs = [
        ("combo_stack", 2, True, 5),
        ("combo_dense", 3, True, 5),
        ("combo_stack", 2, False, 8),
        ("combo_stack", 3, False, 10),
        ("combo_dense", 4, False, 8),
        ("combo_mosaic", 3, False, 8),
    ]

    tasks = []
    for mode, nt, cnt in synth_specs:
        n_val_m = max(1, int(round(cnt * 0.15)))
        for idx_in_m in range(cnt):
            tasks.append(("synth", mode, nt, False, idx_in_m < n_val_m))
    for mode, n_layers, only1, cnt in combo_train_specs:
        n_val_m = max(1, int(round(cnt * 0.15)))
        for idx_in_m in range(cnt):
            tasks.append(("combo", mode, n_layers, only1, idx_in_m < n_val_m))

    scene_seeds = np.random.SeedSequence(202607).spawn(len(tasks))
    print(f"\nBuilding {len(tasks)} scenes ({sum(1 for t in tasks if t[0]=='combo')} real-combined + {sum(1 for t in tasks if t[0]=='synth')} synthetic) on {n_workers} threads...", flush=True)

    def _scene_worker(args):
        (kind, mode, param, only1, is_val), seed = args
        r_local = np.random.default_rng(seed)
        if kind == "combo":
            return synthesize_combined_scene(verified_non_overlap, alphas, r_local, mode=mode, n_layers=param, only_1sek=only1), is_val
        n_t = max(30, int(round(param * r_local.uniform(0.86, 1.10)))) if mode in ("dense1", "dense6", "dense") else param
        return synthesize_scene(sprites, alphas, radii, coins_by_cls, r_local, mode, n_t), is_val

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        synth_out = list(ex.map(_scene_worker, zip(tasks, scene_seeds)))

    train_scenes = [sc for sc, is_val in synth_out if not is_val]
    val_scenes = [sc for sc, is_val in synth_out if is_val]
    del synth_out
    print(
        f"  Built {len(train_scenes)} train scenes ({sum(len(s[3]) for s in train_scenes)} coins) + "
        f"{len(val_scenes)} val scenes ({sum(len(s[3]) for s in val_scenes)} coins) in {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    # Also build 18 dedicated combined training scenes (with JPEG bytes) for Stage-2 Hard-Example Mining
    rng_mine = np.random.default_rng(55501)
    mining_specs = [
        ("combo_stack", 2, True), ("combo_dense", 3, True), ("combo_dense", 4, True),
        ("combo_stack", 2, False), ("combo_stack", 3, False), ("combo_stack", 3, False),
        ("combo_dense", 3, False), ("combo_dense", 4, False), ("combo_dense", 4, False),
        ("combo_mosaic", 3, False), ("combo_mosaic", 4, False), ("combo_mosaic", 4, False),
    ]
    mining_scenes = [
        synthesize_combined_scene(verified_non_overlap, alphas, rng_mine, mode=m, n_layers=nl, only_1sek=o1, return_jpeg=True)
        for m, nl, o1 in mining_specs
    ]

    rim_tr = extract_rim_from_scenes(train_scenes, rng)
    rim_va = extract_rim_from_scenes(val_scenes, rng)

    tr_patches = extract_patches_from_scenes(train_scenes, seed_base=41001, n_workers=n_workers)
    va_patches = extract_patches_from_scenes(val_scenes, seed_base=42001, n_workers=n_workers)
    del train_scenes, val_scenes, sprites, alphas, coins_by_cls, verified_non_overlap, overlap_samples, raw_samples
    gc.collect()

    cX_tr, cY_tr, cW_tr, raw_tr, pV_tr, pCls_tr, pSh_tr, pStats_tr, pLbl_tr, pWt_tr, pTight_tr = tr_patches
    cX_va, cY_va, cW_va, raw_va, pV_va, pCls_va, pSh_va, pStats_va, pLbl_va, pWt_va, pTight_va = va_patches
    print(
        f"Extracted Coarse(tr={len(cX_tr)}, va={len(cX_va)}), "
        f"Polar(tr={len(raw_tr)} [tight={pTight_tr.sum()}], va={len(raw_va)} [tight={pTight_va.sum()}]) "
        f"in {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    eng = CudaEngine()
    print(f"CUDA GPU: {eng.device_name}", flush=True)

    try:
        rim_gate_w, gate_thr, rim_w = train_rim_models(eng, rim_tr, rim_va, rng)
        del rim_tr, rim_va

        print("\n=== Stage 1: Training CoarseNet (25 -> 12 -> 8 -> 4) ===", flush=True)
        coarse = [GpuLinear(25, 12, rng), GpuLinear(12, 8, rng), GpuLinear(8, 4, rng)]
        coarse_w_s1, _ = train_mlp_ce(
            eng, coarse, cX_tr, cY_tr, cW_tr, cX_va, cY_va, cW_va, rng,
            epochs=50, lr_max=8e-3, bs=2048, name="CoarseNet-S1",
        )

        print("\n=== Stage 1: Training Polar C4E W_cls (3, 4, 4, 16) ===", flush=True)
        F_tr = build_polar_features_c4e(raw_tr, pV_tr)
        F_va = build_polar_features_c4e(raw_va, pV_va)
        del raw_tr, raw_va
        gc.collect()

        rng_v = np.random.default_rng(3004)
        W_cls_np = train_polar_w_cls(
            eng,
            F_tr[pTight_tr], pV_tr[pTight_tr], pCls_tr[pTight_tr], pSh_tr[pTight_tr],
            F_va[pTight_va], pV_va[pTight_va], pCls_va[pTight_va],
            rng_v,
        )

        print("\n=== Stage 1: Training AcceptNet (33 -> 20 -> 10 -> 4) ===", flush=True)
        t_feat = time.perf_counter()
        X33_tr = build_accept_features_batch(W_cls_np, F_tr, pV_tr, pStats_tr)
        X33_va = build_accept_features_batch(W_cls_np, F_va, pV_va, pStats_va)
        print(f"  Built X33 features (tr={len(X33_tr)}, va={len(X33_va)}) in {time.perf_counter()-t_feat:.2f}s", flush=True)
        del F_tr, pV_tr, pStats_tr, F_va, pV_va, pStats_va
        gc.collect()

        acc_layers = [GpuLinear(33, 20, rng_v), GpuLinear(20, 10, rng_v), GpuLinear(10, 4, rng_v)]
        acc_w_s1, _ = train_mlp_ce(
            eng, acc_layers, X33_tr, pLbl_tr, pWt_tr, X33_va, pLbl_va, pWt_va, rng_v,
            epochs=80, lr_max=6e-3, bs=1024, wd=1e-4, name="AcceptNet-S1",
        )

        bundle_s1 = _assemble_bundle(rim_gate_w, gate_thr, rim_w, coarse_w_s1, W_cls_np, acc_w_s1, radii_500)
        b85_s1 = pack_weights_lossless(bundle_s1)

        # Stage-2: Hard-Example Mining on Combined Training Scenes & End-to-End Selection
        best_b85 = b85_s1
        has_generator = os.path.exists(os.path.join(BASE_DIR, "generate_solution.py"))
        if has_generator:
            ex_s1, err_s1 = evaluate_on_combined_suite(b85_s1, test_suite, label="Stage-1")
            print("\n=== Stage 2: Hard-Example Mining on Combined Training Scenes ===", flush=True)
            coarse_mined, acc_mined = mine_hard_examples(b85_s1, mining_scenes, W_cls_np, rng_v)

            coarse_w_s2 = coarse_w_s1
            if coarse_mined is not None and len(coarse_mined[0]) >= 32:
                cX_all = np.ascontiguousarray(np.concatenate([cX_tr, coarse_mined[0]], axis=0))
                cY_all = np.ascontiguousarray(np.concatenate([cY_tr, coarse_mined[1]], axis=0))
                cW_all = np.ascontiguousarray(np.concatenate([cW_tr, coarse_mined[2]], axis=0))
                c_ft = [
                    GpuLinear(25, 12, rng, init_wb=coarse_w_s1[0]),
                    GpuLinear(12, 8, rng, init_wb=coarse_w_s1[1]),
                    GpuLinear(8, 4, rng, init_wb=coarse_w_s1[2]),
                ]
                coarse_w_s2, _ = train_mlp_ce(
                    eng, c_ft, cX_all, cY_all, cW_all, cX_va, cY_va, cW_va, rng,
                    epochs=25, lr_max=2.5e-3, bs=2048, warmup_ep=2, name="CoarseNet-S2",
                )

            acc_w_s2 = acc_w_s1
            if acc_mined is not None and len(acc_mined[0]) >= 32:
                X33_all = np.ascontiguousarray(np.concatenate([X33_tr, acc_mined[0]], axis=0))
                pL_all = np.ascontiguousarray(np.concatenate([pLbl_tr, acc_mined[1]], axis=0))
                pW_all = np.ascontiguousarray(np.concatenate([pWt_tr, acc_mined[2]], axis=0))
                a_ft = [
                    GpuLinear(33, 20, rng_v, init_wb=acc_w_s1[0]),
                    GpuLinear(20, 10, rng_v, init_wb=acc_w_s1[1]),
                    GpuLinear(10, 4, rng_v, init_wb=acc_w_s1[2]),
                ]
                acc_w_s2, _ = train_mlp_ce(
                    eng, a_ft, X33_all, pL_all, pW_all, X33_va, pLbl_va, pWt_va, rng_v,
                    epochs=35, lr_max=2.0e-3, bs=1024, wd=1e-4, warmup_ep=2, name="AcceptNet-S2",
                )

            if (coarse_w_s2 is not coarse_w_s1) or (acc_w_s2 is not acc_w_s1):
                bundle_s2 = _assemble_bundle(rim_gate_w, gate_thr, rim_w, coarse_w_s2, W_cls_np, acc_w_s2, radii_500)
                b85_s2 = pack_weights_lossless(bundle_s2)
                ex_s2, err_s2 = evaluate_on_combined_suite(b85_s2, test_suite, label="Stage-2")
                if (err_s2 < err_s1) or (err_s2 == err_s1 and ex_s2 >= ex_s1):
                    print(f"  -> Selecting Stage-2 weights (|err| {err_s1} -> {err_s2}, exact {ex_s1} -> {ex_s2})", flush=True)
                    best_b85 = b85_s2
                else:
                    print(f"  -> Keeping Stage-1 weights (Stage-1 |err|={err_s1} vs Stage-2 |err|={err_s2})", flush=True)
        else:
            print("Note: generate_solution.py not found in directory; skipping Stage-2 runtime mining.", flush=True)
    finally:
        gc.collect()
        eng.close()

    out_w_path = os.path.join(BASE_DIR, "trained_weights.b85")
    with open(out_w_path, "w", encoding="ascii") as f:
        f.write(best_b85)
    print(f"\nSaved & verified lossless weights -> {out_w_path} ({len(best_b85)} chars) in {time.perf_counter()-t0:.1f}s", flush=True)

    if not no_inject:
        if os.path.exists(os.path.join(BASE_DIR, "generate_solution.py")):
            from generate_solution import generate_solution
            generate_solution(weights_b85=best_b85, out_path=SOLUTION_PATH, verify=True)
        elif os.path.exists(SOLUTION_PATH):
            with open(SOLUTION_PATH, "r", encoding="utf-8") as sf:
                sol_src = sf.read()
            sol_new = re.sub(
                r'WEIGHTS_B85\s*=\s*".*?"',
                lambda _m: f'WEIGHTS_B85="{best_b85}"',
                sol_src,
                count=1,
                flags=re.DOTALL,
            )
            with open(SOLUTION_PATH, "w", encoding="utf-8", newline="\n") as sf:
                sf.write(sol_new)
            print(f"Injected weights into {SOLUTION_PATH} ({len(sol_new.encode('utf-8'))} bytes total)", flush=True)


if __name__ == "__main__":
    main()
