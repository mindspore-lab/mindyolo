#include <stdio.h>
#include "elementswise_op_impl.cu"
constexpr int THREADS = 256;

template<typename T>
__device__ T Max(T a, T b){
	if (a > b) {
		return a;
	} else {
		return b;
	}
}

template<typename T>
__device__ T Min(T a, T b){
	if (a < b) {
		return a;
	} else {
		return b;
	}
}

template <typename T>
struct FusedGetConvexDiagonalSquaredFunctor {
    FusedGetConvexDiagonalSquaredFunctor() {}
    __device__ __forceinline__ T operator()(T b1_x1, T b1_x2, T b2_x1, T b2_x2, T b1_y1, T b1_y2, T b2_y1, T b2_y2) const {
      float eps = 1e-7;
      const T cw = Max(b1_x2, b2_x2) - Min(b1_x1, b2_x1);
      const T ch = Max(b1_y2, b2_y2) - Min(b1_y1, b2_y1);
      return cw * cw + ch * ch + static_cast<T>(eps);
    }
};

template <typename T>
void FusedGetConvexDiagonalSquaredKernel(const T* b1_x1, const T* b1_x2, const T* b2_x1, const T* b2_x2, const T* b1_y1,
                                         const T* b1_y2, const T* b2_y1, const T* b2_y2,
                                         T *output, const size_t count, cudaStream_t cuda_stream) {
    FusedGetConvexDiagonalSquaredFunctor<T> functor;
    cuda::elementwise::EightInputs(functor, (uint)(count), output, b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2, cuda_stream);
}


template <typename T>
__global__ void FusedGetConvexDiagonalSquaredGradKernel(
    const int output_num, const T* b1_x1, const T* b1_x2, const T* b2_x1, const T* b2_x2, const T* b1_y1,
    const T* b1_y2, const T* b2_y1, const T* b2_y2, const T* d_c2, T* d_b1_x1, T* d_b1_x2,
    T* d_b2_x1, T* d_b2_x2, T* d_b1_y1, T* d_b1_y2, T* d_b2_y1, T* d_b2_y2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const T zero = static_cast<T>(0);
    const T one = static_cast<T>(1);
    const T two = static_cast<T>(2);
    for(int i = idx; i < output_num; i += blockDim.x * gridDim.x) {
      const T cw = Max(b1_x2[i], b2_x2[i]) - Min(b1_x1[i], b2_x1[i]);
      const T ch = Max(b1_y2[i], b2_y2[i]) - Min(b1_y1[i], b2_y1[i]);
      const T d_c2_cw = two * cw * d_c2[i];
      const T d_c2_ch = two * ch * d_c2[i];
      d_b1_x2[i] = d_c2_cw * (b1_x2[i] > b2_x2[i] ? one : zero);
      d_b2_x2[i] = d_c2_cw * (b1_x2[i] > b2_x2[i] ? zero : one);
      d_b1_x1[i] = -d_c2_cw * (b1_x1[i] < b2_x1[i] ? one : zero);
      d_b2_x1[i] = -d_c2_cw * (b1_x1[i] < b2_x1[i] ? zero : one);
      d_b1_y2[i] = d_c2_ch * (b1_y2[i] > b2_y2[i] ? one : zero);
      d_b2_y2[i] = d_c2_ch * (b1_y2[i] > b2_y2[i] ? zero : one);
      d_b1_y1[i] = -d_c2_ch * (b1_y1[i] < b2_y1[i] ? one : zero);
      d_b2_y1[i] = -d_c2_ch * (b1_y1[i] < b2_y1[i] ? zero : one);
    }
}

extern "C" int FusedGetConvexDiagonalSquared(int nparam, void **params, int *ndims, int64_t **shapes,
                                             const char **dtypes, void *stream, void *extra) {
    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    int expect_num_param = 9;
    if (nparam != expect_num_param) {
      printf("Param num is not equal to %d \n.", expect_num_param);
      return -1;
    }
    // check the data type is float32 or float16
    for (int i = 0; i < nparam; i++) {
        if (strcmp(dtypes[i], "float32") != 0 && strcmp(dtypes[i], "float16") != 0) {
            printf("dtypes is not equal to float32 and float16.");
            return -1;
        }
    }
    int output_index = 8;
    void *b1_x1 = params[0];
    void *b1_x2 = params[1];
    void *b2_x1 = params[2];
    void *b2_x2 = params[3];
    void *b1_y1 = params[4];
    void *b1_y2 = params[5];
    void *b2_y1 = params[6];
    void *b2_y2 = params[7];
    void *c2 = params[8];
    int output_num = 1;

    for (int i = 0; i < ndims[output_index]; i++) {
        output_num *= shapes[output_index][i];
    }
    FusedGetConvexDiagonalSquaredKernel(static_cast<float *>(b1_x1),
                                        static_cast<float *>(b1_x2),
                                        static_cast<float *>(b2_x1),
                                        static_cast<float *>(b2_x2),
                                        static_cast<float *>(b1_y1),
                                        static_cast<float *>(b1_y2),
                                        static_cast<float *>(b2_y1),
                                        static_cast<float *>(b2_y2),
                                        static_cast<float *>(c2), output_num, custream);
    return 0;
}


extern "C" int FusedGetConvexDiagonalSquaredGrad(int nparam, void **params, int *ndims, int64_t **shapes,
                                                 const char **dtypes, void *stream, void *extra) {
    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    int expect_num_param = 17;
    if (nparam != expect_num_param) {
        printf("Param num is not equal to %d \n.", expect_num_param);
        return -1;
    }
    // check the data type is float32 or float16
    for (int i = 0; i < nparam; i++) {
        if (strcmp(dtypes[i], "float32") != 0 && strcmp(dtypes[i], "float16") != 0) {
            printf("dtypes is not equal to float32 and float16.");
            return -1;
        }
    }
    void *b1_x1 = params[0];
    void *b1_x2 = params[1];
    void *b2_x1 = params[2];
    void *b2_x2 = params[3];
    void *b1_y1 = params[4];
    void *b1_y2 = params[5];
    void *b2_y1 = params[6];
    void *b2_y2 = params[7];
    void *d_c2 = params[8];
    void *d_b1_x1 = params[9];
    void *d_b1_x2 = params[10];
    void *d_b2_x1 = params[11];
    void *d_b2_x2 = params[12];
    void *d_b1_y1 = params[13];
    void *d_b1_y2 = params[14];
    void *d_b2_y1 = params[15];
    void *d_b2_y2 = params[16];
    int output_num = 1;

    int output_index = 9;
    for (int i = 0; i < ndims[output_index]; i++) {
        output_num *= shapes[output_index][i];
    }
    int block_num = output_num / THREADS + 1;
    FusedGetConvexDiagonalSquaredGradKernel<<<block_num, THREADS, 0, custream>>>(output_num,
                                                        static_cast<float *>(b1_x1), static_cast<float *>(b1_x2),
                                                        static_cast<float *>(b2_x1), static_cast<float *>(b2_x2),
                                                        static_cast<float *>(b1_y1), static_cast<float *>(b1_y2),
                                                        static_cast<float *>(b2_y1), static_cast<float *>(b2_y2),
                                                        static_cast<float *>(d_c2),  static_cast<float *>(d_b1_x1),
                                                        static_cast<float *>(d_b1_x2), static_cast<float *>(d_b2_x1),
                                                        static_cast<float *>(d_b2_x2), static_cast<float *>(d_b1_y1),
                                                        static_cast<float *>(d_b1_y2), static_cast<float *>(d_b2_y1),
                                                        static_cast<float *>(d_b2_y2));
    return 0;
}
