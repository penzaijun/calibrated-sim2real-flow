// Fused residual-add + LayerNorm kernels for the transformer blocks (rows of exactly 256 bf16 values, one warp per row,
// 8 values per lane), and an in-place bias-add + SiLU for the width CNN.
// Semantics match bf16 autocast: the residual sum is rounded to bf16 (as a bf16 add), LayerNorm runs in fp32 with fp32
// gamma/beta (biased variance, rsqrt(var + eps)), and only the final output is rounded to bf16 (where autocast cast the
// fp32 LayerNorm output for the next Linear). The double variant applies two LayerNorms back to back in fp32 without
// rounding in between (the block's norm2 followed by the LayerNorm at the start of its MLP).
// Build: nvcc -arch=sm_80 -O3 -cubin -o fused_ln_sm80.cubin fused_ln.cu ; nvcc -arch=compute_80 -O3 -ptx -o fused_ln.ptx fused_ln.cu
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float* v) {
    uint4 u = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) { float2 f = __bfloat1622float2(h[i]); v[2 * i] = f.x; v[2 * i + 1] = f.y; }
}

__device__ __forceinline__ void store8(__nv_bfloat16* p, const float* v) {
    uint4 u; __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    *reinterpret_cast<uint4*>(p) = u;
}

__device__ __forceinline__ void layer_norm8(float* v, const float* __restrict__ g, const float* __restrict__ b, int c0, float eps) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += v[i];
    const float mean = warp_sum(s) * (1.0f / 256.0f);
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { const float d = v[i] - mean; q += d * d; }
    const float rstd = rsqrtf(warp_sum(q) * (1.0f / 256.0f) + eps);
    const float4 g0 = *reinterpret_cast<const float4*>(g + c0), g1 = *reinterpret_cast<const float4*>(g + c0 + 4);
    const float4 b0 = *reinterpret_cast<const float4*>(b + c0), b1 = *reinterpret_cast<const float4*>(b + c0 + 4);
    const float gg[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    const float bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = gg[i] * (rstd * (v[i] - mean)) + bb[i];
}

// x_out = bf16(x + add) (skipped when add == nullptr: x is used as is); y = bf16(LN(x_out; g, b))
extern "C" __global__ void __launch_bounds__(128) fast_ln_single(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ add, __nv_bfloat16* __restrict__ x_out,
    __nv_bfloat16* __restrict__ y, const float* __restrict__ g, const float* __restrict__ b, int rows, float eps) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int c0 = (threadIdx.x & 31) * 8;
    const int64_t off = (int64_t)row * 256 + c0;
    float v[8];
    load8(x + off, v);
    if (add != nullptr) {
        float a[8]; load8(add + off, a);
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(__float2bfloat16_rn(v[i] + a[i]));
        store8(x_out + off, v);
    }
    layer_norm8(v, g, b, c0, eps);
    store8(y + off, v);
}

// x_out = bf16(x + add); y = bf16(LN2(LN1(x_out; g1, b1); g2, b2)) with the intermediate kept in fp32
extern "C" __global__ void __launch_bounds__(128) fast_ln_double(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ add, __nv_bfloat16* __restrict__ x_out,
    __nv_bfloat16* __restrict__ y, const float* __restrict__ g1, const float* __restrict__ b1,
    const float* __restrict__ g2, const float* __restrict__ b2, int rows, float eps1, float eps2) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int c0 = (threadIdx.x & 31) * 8;
    const int64_t off = (int64_t)row * 256 + c0;
    float v[8], a[8];
    load8(x + off, v); load8(add + off, a);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(__float2bfloat16_rn(v[i] + a[i]));
    store8(x_out + off, v);
    layer_norm8(v, g1, b1, c0, eps1);
    layer_norm8(v, g2, b2, c0, eps2);
    store8(y + off, v);
}

// In-place x = bf16(silu(bf16(x + bias[c]))) on a channels-last (NHWC) bf16 tensor with C channels per pixel (C % 8 == 0),
// i.e. exactly conv-output.add_(bias) followed by silu as torch computes them (both rounded to bf16).
extern "C" __global__ void __launch_bounds__(256) fast_bias_silu(
    __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ bias, int C, long long n_vec) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vec) return;
    const int c0 = (int)((i % (C >> 3)) << 3);
    float v[8], b[8];
    load8(x + i * 8, v); load8(bias + c0, b);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float s = __bfloat162float(__float2bfloat16_rn(v[k] + b[k]));
        v[k] = s / (1.0f + expf(-s));
    }
    store8(x + i * 8, v);
}
