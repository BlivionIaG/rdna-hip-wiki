// Verbatim excerpts (MIT, ggml-org/llama.cpp authors + stew675/llama-cpp-rdna-boosts) from
// https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/vecdotq.cuh
// lines 8-14, 649-718, 1327-1435, 1446-1519, behind a shim (ggml_cuda_dp4a = its RDNA2 branch, common.cuh L915-L918).
#pragma once
// Shim for the resource probe: HIP runtime + ggml block layouts + the gfx1030 (RDNA2) branch of ggml_cuda_dp4a.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#define GGML_COMMON_DECL_HIP
#include "ggml-common.h"
#define GGML_USE_HIP
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
    c = __builtin_amdgcn_sdot4(a, b, c, false);  // common.cuh:917-918 branch for RDNA2
    return c;
}
template<int width = 32> static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) x += __shfl_xor(x, offset, width);
    return x;
}
static __device__ __forceinline__ int ggml_cuda_mul_small(const int a, const int b) {
#if defined(GGML_USE_HIP)
    return __mul24(a, b);
#else
    return a * b;
#endif
}
#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMVQ_MOE 4
#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * ggml_cuda_mul_small(dot1, sc[i]);
        sumf_m += d8[i] * ggml_cuda_mul_small(dot2, m[i]);

    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

// VDR=4 variant: processes two adjacent 16-element chunks (32 elements).
// Both chunks share the q8_1 block pair, the two sub-scales, the two mins and
// the d8 values, so the loads are amortized over twice the dp4a work of the
// VDR=2 kernel.
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq4(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < 2*QR5_K; ++i) {
        const int vl0i = (vl[2*(i>>1)]     >> (4*(i&1))) & 0x0F0F0F0F;
        const int vl1i = (vl[2*(i>>1) + 1] >> (4*(i&1))) & 0x0F0F0F0F;

        const int vh0i = ((vh[2*(i>>1)]     >> (i&1)) << 4) & 0x10101010;
        const int vh1i = ((vh[2*(i>>1) + 1] >> (i&1)) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i&1] * ggml_cuda_mul_small(dot1, sc[i&1]);
        sumf_m += d8[i&1] * ggml_cuda_mul_small(dot2, m[i&1]);

    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}
static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    // same as q4_K
    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;

    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];

    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);

    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));

    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

// VDR=4 entry point: iqs must be a multiple of 4 (the mmvq kernel strides kqs
// by VDR). Processes 32 elements per call, splitting the ql/qh/u loads over two
// 16-element chunks that share the q8_1 block pair, the sub-scale/min pair and
// the d8 values.
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_vdr4(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    int   vl[4];
    int   vh[4];
    int    u[4*QR5_K];
    float d8[2*QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int i8 = (iqs/2) % 4;

    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * i8);
    const int * qh = (const int *)(bq5_K->qh + 4 * i8);

    vl[0] = ql[0];
    vl[1] = ql[4];
    vl[2] = ql[1];
    vl[3] = ql[5];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    vh[2] = qh[1] >> bq8_offset;
    vh[3] = qh[5] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    d8[0] = __low2float(bq8_1[bq8_offset + 0].ds);
    d8[1] = __low2float(bq8_1[bq8_offset + 1].ds);

#pragma unroll
    for (int i = 0; i < 2*QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + (i&1);

        const int * q8 = (const int *)bq8i->qs + i8 + (i>>1);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq4(vl, vh, u, sc, m, bq5_K->dm, d8);
}

#if defined(GGML_CUDA_Q5K_DENSE_VDR4)
// gfx1100 (RDNA3_0) dense Q5_K mmvq, 32 elements per call (VDR 4): the dense Q4_K kernel plus the fifth bit.  The two
// 16-element chunks of a call belong to the same two sub-blocks, so each sub-block's 16 products are summed in ONE
// integer dot-product chain (with the two qh high bits of each value OR'd into its low nibble) before its single
// scale/convert/fma (|dot1| <= 16*31*127 = 63092, times the 6-bit scale < 2^22, exact); the mins term uses the same
// exact-product form as vec_dot_q5_K_q8_1 via the same 4-deep chain of ones.  That merge is a pure reordering of the
// MoE VDR=4 kernel's two half-chains (identical scale, identical d8), so it changes no arithmetic -- only how many
// times the per-(row, column) float work and the 16-element unpacking happen, which is what bounds the multi-row
// verify kernel on gfx1100.  Q5_K's scales/mins share Q4_K's layout, hence the identical extraction.
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_vdr4_dense(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    // iqs is in 0,4..28: chunks iqs and iqs+2 share bq8_offset and the scale pair (see vec_dot_q5_K_q8_1_vdr4)
    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int i8 = (iqs/2) % 4;

    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * i8);
    const int * qh = (const int *)(bq5_K->qh + 4 * i8);
    const int v0 = ql[0];
    const int v1 = ql[4];
    const int v2 = ql[1];
    const int v3 = ql[5];
    const int h0 = qh[0] >> bq8_offset;
    const int h1 = qh[4] >> bq8_offset;
    const int h2 = qh[1] >> bq8_offset;
    const int h3 = qh[5] >> bq8_offset;

    // scales/mins of sub-block pair j packed as in vec_dot_q4_K_q8_1 (Q5_K uses Q4_K's 6-bit scale/min layout)
    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;

    const uint32_t s02 = scales[jm + 0] | ((uint32_t) scales[jm + 2] << 16);
    const uint32_t s4  = scales[jm + 4];

    const uint32_t scm_lo = s02 & 0x3f3f3f3f;
    const uint32_t scm_hi = ((s4 | (s4 << 12)) & 0x0f0f0f0f) | ((s02 & 0xc0c0c0c0) >> 2);
    const uint32_t scm    = j >= 2 ? scm_hi : scm_lo;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        const int * q8 = (const int *)bq8i->qs + i8;
        const float d8 = __low2float(bq8i->ds);

        const int vi0 = ((v0 >> (4*i)) & 0x0F0F0F0F) | (((h0 >> i) << 4) & 0x10101010);
        const int vi1 = ((v1 >> (4*i)) & 0x0F0F0F0F) | (((h1 >> i) << 4) & 0x10101010);
        const int vi2 = ((v2 >> (4*i)) & 0x0F0F0F0F) | (((h2 >> i) << 4) & 0x10101010);
        const int vi3 = ((v3 >> (4*i)) & 0x0F0F0F0F) | (((h3 >> i) << 4) & 0x10101010);

        int dot1 = ggml_cuda_dp4a(vi0, q8[0], 0);
        dot1     = ggml_cuda_dp4a(vi1, q8[4], dot1);
        dot1     = ggml_cuda_dp4a(vi2, q8[1], dot1);
        dot1     = ggml_cuda_dp4a(vi3, q8[5], dot1);

        int dot2 = ggml_cuda_dp4a(0x01010101, q8[0], 0);
        dot2     = ggml_cuda_dp4a(0x01010101, q8[4], dot2);
        dot2     = ggml_cuda_dp4a(0x01010101, q8[1], dot2);
        dot2     = ggml_cuda_dp4a(0x01010101, q8[5], dot2);

        sumf_d += d8 * ggml_cuda_mul_small(dot1, (scm >> (8*i)) & 0xFF);
        sumf_m += (d8 * (float) dot2) * (float) ((scm >> (8*i + 16)) & 0xFF);
    }

    const float2 dm5f = __half22float2(bq5_K->dm);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}
#endif // defined(GGML_CUDA_Q5K_DENSE_VDR4)
