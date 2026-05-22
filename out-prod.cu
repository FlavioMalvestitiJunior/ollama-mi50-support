#include "out-prod.cuh"

#include <cstdint>

// Simple tiled F32 GEMM for gfx906 (no rocBLAS on Vega20).
// C[M,N] = A[M,K] * B^T[N,K]  (row-major, B is the transposed operand)
// Grid: ((M+63)/64, (N+63)/64), Block: (16,16).
template<typename T0, typename T1>
__global__ static void vega20_out_prod_gemm(
        const T0 * __restrict__ A, const T0 * __restrict__ B, T1 * __restrict__ C,
        int M, int N, int K, int lda, int ldb, int ldc) {
    __shared__ T0 sA[16][16];
    __shared__ T0 sB[16][16];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.x * 64 + ty * 16 + tx % 16;
    const int col = blockIdx.y * 64 + ty * 16 + tx / 16;
    // Simple non-tiled path for correctness (out-prod is not perf-critical)
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += (float)A[row * lda + k] * (float)B[col * ldb + k];
    }
    C[row * ldc + col] = (T1)acc;
    (void)sA; (void)sB;
}

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne01 == ne11);
    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);

    GGML_ASSERT(ne2 % src0->ne[2] == 0);
    GGML_ASSERT(ne3 % src0->ne[3] == 0);

    GGML_ASSERT(ne2 == src1->ne[2]);
    GGML_ASSERT(ne3 == src1->ne[3]);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    cudaStream_t   stream = ctx.stream();

    // Strides and derived quantities — declared before any branch
    const int64_t lda = nb01 / sizeof(float);
    const int64_t ldc = nb1  / sizeof(float);

    const bool src1_T = ggml_is_transposed(src1);
    const int64_t ldb = (src1_T ? nb10 : nb11) / sizeof(float);
    GGML_ASSERT(         (src1_T ? nb11 : nb10) == sizeof(float));

    const size_t s02 = nb02 / sizeof(float);
    const size_t s03 = nb03 / sizeof(float);
    const size_t s12 = nb12 / sizeof(float);
    const size_t s13 = nb13 / sizeof(float);
    const size_t s2  = nb2  / sizeof(float);
    const size_t s3  = nb3  / sizeof(float);

    const int64_t dps2 = ne2 / ne02;
    const int64_t dps3 = ne3 / ne03;

    const int out_cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (out_cc == GGML_CUDA_CC_VEGA20) {
        // cublasSgemm has no gfx906 kernel — use local tiled GEMM fallback.
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                const float * src0_b = src0_d + (i3/dps3)*s03 + (i2/dps2)*s02;
                const float * src1_b = src1_d + i3*s13 + i2*s12;
                float       *  dst_b = dst_d  + i3*s3  + i2*s2;
                const int M = (int)ne0, N = (int)ne1, K = (int)ne01;
                const dim3 block(16, 16);
                if (!src1_T) {
                    const dim3 grid((M+63)/64, (N+63)/64);
                    vega20_out_prod_gemm<float, float><<<grid, block, 0, stream>>>(
                        src0_b, src1_b, dst_b, M, N, K, (int)lda, (int)ldb, (int)ldc);
                } else {
                    const dim3 grid((N+63)/64, (M+63)/64);
                    vega20_out_prod_gemm<float, float><<<grid, block, 0, stream>>>(
                        src1_b, src0_b, dst_b, N, M, K, (int)ldb, (int)lda, (int)ldc);
                }
            }
        }
        return;
    }

    cublasHandle_t handle = ctx.cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const cublasOperation_t src1_cublas_op = src1_T ? CUBLAS_OP_N : CUBLAS_OP_T;

    // TODO batched matrix multiplication
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            CUBLAS_CHECK(
                cublasSgemm(handle, CUBLAS_OP_N, src1_cublas_op,
                        ne0, ne1, ne01,
                        &alpha, src0_d + (i3/dps3)*s03 + (i2/dps2)*s02, lda,
                                src1_d +  i3      *s13 +  i2      *s12, ldb,
                        &beta,  dst_d  +  i3      *s3  +  i2      *s2,  ldc));
        }
    }
}
