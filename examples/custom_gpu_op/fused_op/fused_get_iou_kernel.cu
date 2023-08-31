#include <stdio.h>
#include "elementswise_op_impl.cu"

constexpr int THREADS = 256;
constexpr float EPS = 1e-7;

__global__ void FusedGetIouKernel(const float *w1, const float *h1, const float *w2, const float *h2, const float *inter,
                                  float *output, float *val_union, const size_t size)
{
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; idx < size; idx += step)
  {
    float val_union_i = w1[idx] * h1[idx] + w2[idx] * h2[idx] - inter[idx] + EPS;
    output[idx] = inter[idx] / val_union_i;
    val_union[idx] = val_union_i;
  }
}

extern "C" int FusedGetIou(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                           void *extra)
{
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  constexpr int OUTPUT_INDEX = 5;
  constexpr int TOTAL_PARAM_NUM = 7;
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
  void *w1 = params[0];
  void *h1 = params[1];
  void *w2 = params[2];
  void *h2 = params[3];
  void *inter = params[4];
  void *out = params[5];
  void *val_union = params[6];
  size_t size = 1;
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++)
  {
    size *= shapes[OUTPUT_INDEX][i];
  }

  int n = size / THREADS;
  FusedGetIouKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<const float *>(w1), static_cast<const float *>(h1),
                                                     static_cast<const float *>(w2), static_cast<const float *>(h2),
                                                     static_cast<const float *>(inter), static_cast<float *>(out),
                                                     static_cast<float *>(val_union), size);
  return 0;
}

__global__ void FusedGetIouBpropKernel(const float *w1, const float *h1, const float *w2, const float *h2,
                                       const float *inter, const float *d_out, const float *d_val_union,
                                       float *d_w1, float *d_h1, float *d_w2, float *d_h2, float *d_inter,
                                       size_t size)
{

  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < size;
       i += step)
  {
    const float w1_i = w1[i], h1_i = h1[i], w2_i = w2[i], h2_i = h2[i], inter_i = inter[i],
                diou_i = d_out[i], dunion_i = d_val_union[i];
    const float w_h_eps = w1_i * h1_i + w2_i * h2_i + EPS;
    const float w_h_eps_inter_diff = w_h_eps - inter_i;
    const float w_h_eps_inter_diff_square = w_h_eps_inter_diff * w_h_eps_inter_diff;
    const float common_for_dwh = -inter_i * diou_i / w_h_eps_inter_diff_square;
    d_inter[i] = w_h_eps * diou_i / w_h_eps_inter_diff_square - dunion_i;
    d_w1[i] = h1_i * (common_for_dwh + dunion_i);
    d_h1[i] = w1_i * (common_for_dwh + dunion_i);
    d_w2[i] = h2_i * (common_for_dwh + dunion_i);
    d_h2[i] = w2_i * (common_for_dwh + dunion_i);
  }
}

extern "C" int FusedGetIouBprop(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                                void *extra)
{
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  constexpr int OUTPUT_INDEX = 7;
  constexpr int TOTAL_PARAM_NUM = 12;
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
  void *w1 = params[0];
  void *h1 = params[1];
  void *w2 = params[2];
  void *h2 = params[3];
  void *inter = params[4];
  void *d_out = params[5];
  void *d_val_union = params[6];
  void *d_w1 = params[7];
  void *d_h1 = params[8];
  void *d_w2 = params[9];
  void *d_h2 = params[10];
  void *d_inter = params[11];
  // calculate the size to data to be processed
  size_t size = 1;
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++)
  {
    size *= shapes[OUTPUT_INDEX][i];
  }
  int n = size / THREADS;

  FusedGetIouBpropKernel<<<n + 1, THREADS, 0, custream>>>(static_cast<float *>(w1), static_cast<float *>(h1),
                                                          static_cast<float *>(w2), static_cast<float *>(h2),
                                                          static_cast<float *>(inter), static_cast<float *>(d_out),
                                                          static_cast<float *>(d_val_union), static_cast<float *>(d_w1),
                                                          static_cast<float *>(d_h1), static_cast<float *>(d_w2),
                                                          static_cast<float *>(d_h2), static_cast<float *>(d_inter), size);
  return 0;
}
