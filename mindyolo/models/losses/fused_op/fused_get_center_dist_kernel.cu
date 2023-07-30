#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "elementswise_op_impl.cu"

constexpr int THREADS = 256;

template <typename T>
struct FusedGetCenterDistFunctor
{
  FusedGetCenterDistFunctor() {}
  __device__ __forceinline__ T operator()(T b1_x1, T b1_x2, T b1_y1, T b1_y2, T b2_x1, T b2_x2, T b2_y1, T b2_y2) const
  {
    T a = b2_x1 + b2_x2 - b1_x1 - b1_x2;
    T b = b2_y1 + b2_y2 - b1_y1 - b1_y2;
    return (a * a + b * b) / 4;
  }
};

template <typename T>
void FusedGetCenterDistKernel(const T *b1_x1, const T *b1_x2, const T *b1_y1, const T *b1_y2,
                              const T *b2_x1, const T *b2_x2, const T *b2_y1, const T *b2_y2,
                              T *output, const size_t count, cudaStream_t cuda_stream)
{
  FusedGetCenterDistFunctor<T> functor;
  cuda::elementwise::EightInputs(functor, (uint)(count), output, b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2, cuda_stream);
}

extern "C" int FusedGetCenterDist(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                  void *stream, void *extra)
{
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  constexpr int OUTPUT_INDEX = 8;
  constexpr int TOTAL_PARAM_NUM = 9;
  if (nparam != TOTAL_PARAM_NUM)
  {
    printf("[Error] nparam is %d", nparam);
    return 1;
  }
  // check the data type is float32
  for (int i = 0; i < nparam; i++)
  {
    if (strcmp(dtypes[i], "float32") != 0)
    {
      return 2;
    }
  }
  // read input & output parameters
  void *b1_x1 = params[0];
  void *b1_x2 = params[1];
  void *b1_y1 = params[2];
  void *b1_y2 = params[3];
  void *b2_x1 = params[4];
  void *b2_x2 = params[5];
  void *b2_y1 = params[6];
  void *b2_y2 = params[7];
  void *out = params[8];
  // calculate the size to data to be processed
  size_t size = 1;
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++)
  {
    size *= shapes[OUTPUT_INDEX][i];
  }
  FusedGetCenterDistKernel(static_cast<const float *>(b1_x1), static_cast<const float *>(b1_x2),
                           static_cast<const float *>(b1_y1), static_cast<const float *>(b1_y2),
                           static_cast<const float *>(b2_x1), static_cast<const float *>(b2_x2),
                           static_cast<const float *>(b2_y1), static_cast<const float *>(b2_y2),
                           static_cast<float *>(out), size, custream);
  return 0;
}

__global__ void FusedGetCenterDistBpropKernel(const float *b1_x1, const float *b1_x2, const float *b1_y1, const float *b1_y2,
                                              const float *b2_x1, const float *b2_x2, const float *b2_y1, const float *b2_y2,
                                              const float *d_out, float *d_b1_x1, float *d_b1_x2, float *d_b1_y1, float *d_b1_y2,
                                              float *d_b2_x1, float *d_b2_x2, float *d_b2_y1, float *d_b2_y2, const size_t size)
{
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; idx < size;
       idx += step)
  {
    const float b1_x1_i = b1_x1[idx];
    const float b1_x2_i = b1_x2[idx];
    const float b1_y1_i = b1_y1[idx];
    const float b1_y2_i = b1_y2[idx];
    const float b2_x1_i = b2_x1[idx];
    const float b2_x2_i = b2_x2[idx];
    const float b2_y1_i = b2_y1[idx];
    const float b2_y2_i = b2_y2[idx];
    const float d_out_i = d_out[idx];
    float dx_i = (b2_x1_i + b2_x2_i - b1_x1_i - b1_x2_i) * d_out_i / 2;
    d_b1_x1[idx] = -dx_i;
    d_b1_x2[idx] = -dx_i;
    d_b2_x1[idx] = dx_i;
    d_b2_x2[idx] = dx_i;
    float dy_i = (b2_y1_i + b2_y2_i - b1_y1_i - b1_y2_i) * d_out_i / 2;
    d_b1_y1[idx] = -dy_i;
    d_b1_y2[idx] = -dy_i;
    d_b2_y1[idx] = dy_i;
    d_b2_y2[idx] = dy_i;
  }
}

extern "C" int FusedGetCenterDistBprop(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                       void *stream, void *extra)
{
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  constexpr int OUTPUT_INDEX = 9;
  constexpr int TOTAL_PARAM_NUM = 17;
  if (nparam != TOTAL_PARAM_NUM)
    return 1;
  // check the data type is float32
  for (int i = 0; i < nparam; i++)
  {
    if (strcmp(dtypes[i], "float32") != 0)
    {
      return 2;
    }
  }
  // read input & output parameters
  void *b1_x1 = params[0];
  void *b1_x2 = params[1];
  void *b1_y1 = params[2];
  void *b1_y2 = params[3];
  void *b2_x1 = params[4];
  void *b2_x2 = params[5];
  void *b2_y1 = params[6];
  void *b2_y2 = params[7];
  void *d_out = params[8];
  void *d_b1_x1 = params[9];
  void *d_b1_x2 = params[10];
  void *d_b1_y1 = params[11];
  void *d_b1_y2 = params[12];
  void *d_b2_x1 = params[13];
  void *d_b2_x2 = params[14];
  void *d_b2_y1 = params[15];
  void *d_b2_y2 = params[16];
  // calculate the size to data to be processed
  size_t size = 1;
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++)
  {
    size *= shapes[OUTPUT_INDEX][i];
  }
  int n = size / THREADS;

  FusedGetCenterDistBpropKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<const float *>(b1_x1), static_cast<const float *>(b1_x2),
                                                                 static_cast<const float *>(b1_y1), static_cast<const float *>(b1_y2),
                                                                 static_cast<const float *>(b2_x1), static_cast<const float *>(b2_x2),
                                                                 static_cast<const float *>(b2_y1), static_cast<const float *>(b2_y2),
                                                                 static_cast<const float *>(d_out),
                                                                 static_cast<float *>(d_b1_x1), static_cast<float *>(d_b1_x2),
                                                                 static_cast<float *>(d_b1_y1), static_cast<float *>(d_b1_y2),
                                                                 static_cast<float *>(d_b2_x1), static_cast<float *>(d_b2_x2),
                                                                 static_cast<float *>(d_b2_y1), static_cast<float *>(d_b2_y2), size);
  return 0;
}
