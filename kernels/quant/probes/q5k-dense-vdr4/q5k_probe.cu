// Resource probe only (not upstream code): mimics the mul_mat_vec_q inner loop (rows x ncols_dst accumulators,
// nwarps wave32 waves, LDS cross-wave reduce) around the two Q5_K dot products copied verbatim into q5k_src.cuh
// from llama.cpp tree bc7898a2 (stew675/llama.cpp af7a57ba, = llama-cpp-rdna-boosts r368 d829586b).
#include "q5k_src.cuh"

template <int vdr, bool dense, int ncols_dst, int nwarps, int rows>
__launch_bounds__(nwarps*32, 1)
__global__ void probe(const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
                      float * __restrict__ dst, int ncols_x, int stride_col_y) {
    constexpr int qk = QK_K, qi = QI5_K, warp_size = 32;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;
    const int tid = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = rows*blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    float tmp[ncols_dst][rows] = {{0.0f}};
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                const int kbxr = (row0 + i)*blocks_per_row_x + kbx;
                if constexpr (dense) tmp[j][i] += vec_dot_q5_K_q8_1_vdr4_dense(vx, &y[j*stride_col_y + kby], kbxr, kqs);
                else                 tmp[j][i] += vec_dot_q5_K_q8_1      (vx, &y[j*stride_col_y + kby], kbxr, kqs);
            }
        }
    }
    if constexpr (nwarps > 1) {
        __shared__ float tmp_shared[nwarps-1][ncols_dst][rows][warp_size];
        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j)
#pragma unroll
                for (int i = 0; i < rows; ++i) tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
        }
        __syncthreads();
        if (threadIdx.y > 0) return;
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j)
#pragma unroll
            for (int i = 0; i < rows; ++i)
#pragma unroll
                for (int l = 0; l < nwarps-1; ++l) tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
    }
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j)
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            const float s = warp_reduce_sum<warp_size>(tmp[j][i]);
            if (threadIdx.x == 0) dst[j*gridDim.x*rows + row0 + i] = s;
        }
}

#define INST(V,D,N,W,R) template __global__ void probe<V,D,N,W,R>(const void*, const block_q8_1*, float*, int, int);
// plugin mmvq shape (opengfx1030/vllm-gguf-plugin mmvq.cuh: one wave32 per row, one column per block)
INST(2,false,1,1,1) INST(4,true,1,1,1)
// one-row shapes
INST(2,false,1,4,1) INST(4,true,1,4,1)
INST(2,false,2,2,1) INST(4,true,2,2,1)
INST(2,false,4,2,1) INST(4,true,4,2,1)
INST(2,false,8,1,1) INST(4,true,8,1,1)
// Navi21-table shapes (calc_nwarps / calc_rows_per_block NAVI21): n=2..4 -> 2 waves x 2 rows; n=5..8 -> 1 wave x 4 rows
INST(2,false,2,2,2) INST(4,true,2,2,2)
INST(2,false,4,2,2) INST(4,true,4,2,2)
INST(2,false,8,1,4) INST(4,true,8,1,4)
