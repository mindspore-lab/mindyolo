#include <stdio.h>
#include "elementswise_op_impl.cu"
constexpr int THREADS = 256;

template <typename T>
struct FusedGetCiouDiagonalAngleFunctor {
    FusedGetCiouDiagonalAngleFunctor() {}
    __device__ __forceinline__ T operator()(T w1, T h1, T w2, T h2) const {
        T eps = static_cast<T>(1e-7);
        const T angle = atan(w2 / (h2 + eps)) - atan(w1 / (h1 + eps));
        return static_cast<T>(4.0 / (M_PI * M_PI) * angle * angle);
    }
};

template <typename T>
void FusedGetCiouDiagonalAngleKernel(const T *w1, const T *h1, const T *w2, const T *h2,
                                     T *output, const size_t count, cudaStream_t cuda_stream) {
    FusedGetCiouDiagonalAngleFunctor<T> functor;
    cuda::elementwise::FourInputs(functor, (uint)(count), output, w1, h1, w2, h2, cuda_stream);
}


template <typename T>
__global__ void FusedGetCiouDiagonalAngleGradKernel(
    const int output_num, const T* w1, const T* h1, const T* w2, const T* h2, const T* v_diff,
                                       T* w1_diff, T* h1_diff, T* w2_diff,
                                       T* h2_diff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T eps = static_cast<T>(1e-7);
    for(int i = idx; i < output_num; i += blockDim.x * gridDim.x) {
        const T w1_i = w1[i];
        const T h1_i = h1[i];
        const T w2_i = w2[i];
        const T h2_i = h2[i];
        const T v_diff_i = v_diff[i];
        const T angle_delta = static_cast<T>(8.0) * (atan(w2_i / (h2_i + eps)) - atan(w1_i / (h1_i + eps))) / static_cast<T>((M_PI * M_PI));
        const T angle1 = static_cast<T>(1.0) + (w1_i * w1_i / ((h1_i + eps) * (h1_i + eps)));
        const T angle2 = static_cast<T>(1.0) + (w2_i * w2_i / ((h2_i + eps) * (h2_i + eps)));
        w1_diff[i] = static_cast<T>(-1.0) * angle_delta / ((h1_i + eps) * angle1) * v_diff_i;
        w2_diff[i] = angle_delta / ((h2_i + eps) * angle2) * v_diff_i;
        h1_diff[i] = w1_i * angle_delta / ((h1_i + eps) * (h1_i + eps) * angle1) * v_diff_i;
        h2_diff[i] = static_cast<T>(-1.0) * w2_i * angle_delta / ((h2_i + eps) * (h2_i + eps) * angle2) * v_diff_i;
    }
}

extern "C" int FusedGetCiouDiagonalAngle(int nparam, void **params, int *ndims, int64_t **shapes,
                                         const char **dtypes, void *stream, void *extra) {
    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    int expect_num_param = 5;
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
    int output_index = 4;
    void *w1 = params[0];
    void *h1 = params[1];
    void *w2 = params[2];
    void *h2 = params[3];
    void *v = params[4];
    int output_num = 1;

    for (int i = 0; i < ndims[output_index]; i++) {
        output_num *= shapes[output_index][i];
    }
    FusedGetCiouDiagonalAngleKernel(static_cast<float *>(w1),
                                    static_cast<float *>(h1),
                                    static_cast<float *>(w2),
                                    static_cast<float *>(h2),
                                    static_cast<float *>(v), output_num, custream);
    return 0;
}


extern "C" int FusedGetCiouDiagonalAngleGrad(int nparam, void **params, int *ndims, int64_t **shapes,
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

    void *w1 = params[0];
    void *h1 = params[1];
    void *w2 = params[2];
    void *h2 = params[3];
    void *v_diff = params[4];
    void *w1_diff = params[5];
    void *h1_diff = params[6];
    void *w2_diff = params[7];
    void *h2_diff = params[8];
    int output_num = 1;

    int output_index = 4;
    for (int i = 0; i < ndims[output_index]; i++) {
        output_num *= shapes[output_index][i];
    }
    int block_num = output_num / THREADS + 1;
    FusedGetCiouDiagonalAngleGradKernel<<<block_num, THREADS, 0, custream>>>(output_num,
                                                        static_cast<float *>(w1), static_cast<float *>(h1),
                                                        static_cast<float *>(w2), static_cast<float *>(h2),
                                                        static_cast<float *>(v_diff), static_cast<float *>(w1_diff),
                                                        static_cast<float *>(h1_diff), static_cast<float *>(w2_diff),
                                                        static_cast<float *>(h2_diff));
    return 0;
}